"""E056 independent repeated-RF Bloch and complex-image physics audit.

No Koma simulation, sorting, interpolation, or reconstruction helper is imported.
The exported waveform is an input, not an independently specified scanner pulse.
The spatial/motion inputs are analytical controls, not a substitute CFD dataset.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tomllib

import h5py
import numpy as np


ENCODINGS = ("REF", "X", "Y", "Z")
MODELS = ("static", "uniform_osc", "parabolic_osc")


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def complex_array(dataset):
    a = dataset[()]
    if np.iscomplexobj(a):
        return np.asarray(a, np.complex128)
    if a.dtype.names:
        names = a.dtype.names
        real = next((n for n in ("r", "re", "real") if n in names), names[0])
        imag = next((n for n in ("i", "im", "imag") if n in names), names[1])
        return a[real].astype(float) + 1j * a[imag].astype(float)
    raise TypeError("Expected native or compound complex dataset")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def interpolate(t, knots, values):
    if values.ndim == 1:
        return np.interp(t, knots, values)
    return np.column_stack([np.interp(t, knots, values[:, c]) for c in range(values.shape[1])])


def rhs(m, rf, bz, t1, t2, gamma):
    field = np.empty_like(m)
    field[:, 0], field[:, 1], field[:, 2] = rf.real, rf.imag, bz
    result = 2 * np.pi * gamma * np.cross(m, field)
    result[:, :2] -= m[:, :2] / t2
    result[:, 2] += (1 - m[:, 2]) / t1
    return result


def rf_affine(template, z, scale, t1, t2, gamma, step=0.5e-6):
    """Finite RF map, affine in pre-RF longitudinal magnetization.

    Every spin starts a pulse with transverse magnetization zero, matching the
    explicitly declared perfect-spoiling simulation method. Two initial states
    recover the affine map, including relaxation during finite RF excitation.
    """
    knots, b1, grad = template["t"], template["b1"], template["g"]
    nonzero = np.flatnonzero(np.abs(b1) > 0)
    if not len(nonzero):
        raise ValueError("Missing physical RF")
    end_index = min(int(nonzero[-1]) + 1, len(knots) - 1)
    end = float(knots[end_index])
    # Reject rather than silently ignore RF frequency/frame modulation.
    if np.max(np.abs(template["frequency"])) > 1e-9:
        raise ValueError("Nonzero RF frequency needs explicit rotating-frame treatment")
    before = knots <= end + 1e-15
    if np.max(np.abs(grad[before, :2])) > 1e-12:
        raise ValueError("Transverse RF gradient invalidates the z-only affine factorization")
    cuts = np.unique(np.r_[0., knots[before], end])
    fine = [cuts[0]]
    for left, right in zip(cuts[:-1], cuts[1:]):
        fine.extend(np.linspace(left, right, max(1, int(np.ceil((right-left)/step)))+1)[1:])
    time = np.asarray(fine)
    sample = interpolate(time, knots, b1)
    gz = interpolate(time, knots, grad[:, 2])
    out = []
    for initial in (0., 1.):
        m = np.zeros((len(z), 3)); m[:, 2] = initial
        for j, dt in enumerate(np.diff(time)):
            f0, f1 = scale * sample[j], scale * sample[j+1]
            z0, z1 = z * gz[j], z * gz[j+1]
            k1 = rhs(m, f0, z0, t1, t2, gamma)
            k2 = rhs(m+dt*k1/2, (f0+f1)/2, (z0+z1)/2, t1, t2, gamma)
            k3 = rhs(m+dt*k2/2, (f0+f1)/2, (z0+z1)/2, t1, t2, gamma)
            k4 = rhs(m+dt*k3, f1, z1, t1, t2, gamma)
            m += dt * (k1+2*k2+2*k3+k4)/6
        out.append(m)
    zero, one = out
    return dict(end=end, az=one[:, 2]-zero[:, 2], bz=zero[:, 2],
                axy=(one[:, 0]+1j*one[:, 1])-(zero[:, 0]+1j*zero[:, 1]),
                bxy=zero[:, 0]+1j*zero[:, 1])


def phase_moments(template, rf_start, rf_end, path_time, path_factor, period):
    """Exact Gauss integration for piecewise-linear G and saved path.

    The saved cyclic path, not a fitted sinusoid, drives the numerical reference.
    Both path knots and gradient/ADC knots split every integration interval.
    """
    adc = template["adc"]
    finish = float(adc[-1])
    first_cycle = int(np.floor((rf_start+rf_end)/period))
    last_cycle = int(np.floor((rf_start+finish)/period))
    motion_knots = np.concatenate([path_time+k*period-rf_start
                                   for k in range(first_cycle, last_cycle+1)])
    cuts = np.unique(np.r_[rf_end, finish, adc, template["t"], motion_knots])
    cuts = cuts[(cuts >= rf_end-1e-15) & (cuts <= finish+1e-15)]
    mid, half = (cuts[:-1]+cuts[1:])/2, np.diff(cuts)/2
    points = np.column_stack([mid-half/np.sqrt(3), mid+half/np.sqrt(3)]).ravel()
    g = interpolate(points, template["t"], template["g"])
    unit = np.interp(np.remainder(rf_start+points, period), path_time, path_factor)
    ordinary = np.cumsum((g.reshape(-1, 2, 3) * half[:, None, None]).sum(axis=1), axis=0)
    moving = np.cumsum((g[:, 0]*unit).reshape(-1, 2).sum(axis=1)*half)
    ordinary = np.vstack([np.zeros(3), ordinary]); moving = np.r_[0., moving]
    indices = np.searchsorted(cuts, adc)
    if np.max(np.abs(cuts[indices]-adc)) > 1e-12:
        raise AssertionError("ADC integration endpoint missing")
    return ordinary[indices], moving[indices]


def load_group(group, nx, ny, nz):
    metadata = {key: group[key][()] for key in
                ("rf_start_s", "rf_scale", "rf_phase_rad", "profile_rf_index0", "adc_first_s", "dwell_s", "labels0")}
    labels = np.asarray(metadata["labels0"], int)
    if labels.shape[1] != 3 and labels.shape[0] == 3:
        labels = labels.T
    metadata["labels"] = labels
    metadata["compensation"] = complex_array(group["adc_phase_compensation"])
    if metadata["compensation"].shape != (len(labels), nx):
        raise ValueError("Unexpected ADC compensation shape")
    templates = []
    for number in range(ny*nz):
        s = group["shots"][f"{number:04d}"]
        g = np.asarray(s["g_T_m"], float)
        if g.shape[1] != 3 and g.shape[0] == 3:
            g = g.T
        templates.append(dict(t=np.asarray(s["t_s"], float), g=g,
                              b1=complex_array(s["b1_T"]),
                              frequency=np.asarray(s["rf_frequency_Hz"], float),
                              adc=np.asarray(s["adc_time_s"], float)))
    metadata["templates"] = templates
    metadata["gamma"] = float(group["gamma_Hz_T"][()]) if "gamma_Hz_T" in group else 42.577478518e6
    if not 42e6 < metadata["gamma"] < 43e6:
        raise ValueError("Invalid gamma")
    return metadata


def predict(metadata, positions, rho, u, t1, t2, path_time, factor, period, shape, require_complete=True):
    nx, ny, nz, bins = shape
    yz, inverse = np.unique(positions[:, 1:3], axis=0, return_inverse=True)
    z, z_inverse = np.unique(yz[:, 1], return_inverse=True)
    groups = [np.flatnonzero(inverse == j) for j in range(len(yz))]
    group_u = np.asarray([u[g[0]] for g in groups])
    for j, ids in enumerate(groups):
        if not np.all(u[ids] == group_u[j]):
            raise ValueError("Motion is not invariant along the factored readout lattice")
    # Factor identical x-coordinate populations without assuming equal weights.
    x = positions[groups[0], 0]
    if not all(np.array_equal(positions[g, 0], x) for g in groups):
        raise ValueError("Readout quadrature differs between transverse populations")
    weights = np.asarray([rho[g] for g in groups])
    starts = np.asarray(metadata["rf_start_s"], float).ravel()
    scales = np.round(np.asarray(metadata["rf_scale"], float).ravel(),12)
    phases = np.asarray(metadata["rf_phase_rad"], float).ravel()
    template = metadata["templates"][0]
    operators = {float(scale): rf_affine(template, z, float(scale), t1, t2, metadata["gamma"])
                 for scale in np.unique(scales)}
    state = np.ones(len(z)); previous_end = 0.
    xy_after = np.empty((len(starts), len(z)), np.complex128)
    for j, start in enumerate(starts):
        operator = operators[float(scales[j])]
        delay = float(start-previous_end)
        if delay < -1e-10:
            raise ValueError("Overlapping RF pulses")
        state = 1+(state-1)*np.exp(-max(0., delay)/t1)
        xy_after[j] = (operator["axy"]*state+operator["bxy"])*np.exp(1j*phases[j])
        state = operator["az"]*state+operator["bz"]
        previous_end = float(start+operator["end"])
    labels = metadata["labels"]
    k = np.zeros((bins, nz, ny, nx), np.complex128)
    filled = np.zeros((bins, nz, ny), bool)
    raw = np.empty((len(labels), nx), np.complex128)
    for p, (ky, kz, cardiac) in enumerate(labels):
        if filled[cardiac, kz, ky]:
            raise ValueError("Duplicate independently sorted profile")
        filled[cardiac, kz, ky] = True
        rf = int(np.asarray(metadata["profile_rf_index0"]).ravel()[p])
        shot = metadata["templates"][int(ky+ny*kz)]
        operator = operators[float(scales[rf])]
        ordinary, moving = phase_moments(shot, starts[rf], operator["end"], path_time, factor, period)
        phase = -2*np.pi*metadata["gamma"]
        x_sum = np.exp(1j*phase*ordinary[:, 0, None]*x[None]) @ weights.T
        transverse = (ordinary[:, 1, None]*yz[None, :, 0] +
                      ordinary[:, 2, None]*yz[None, :, 1] + moving[:, None]*group_u[None])
        signal = np.sum(x_sum*np.exp(1j*phase*transverse)*xy_after[rf, z_inverse][None], axis=1)
        signal *= np.exp(-(shot["adc"]-operator["end"])/t2)
        signal *= metadata["compensation"][p]
        raw[p] = signal; k[cardiac, kz, ky] = signal
    if require_complete and not filled.all():
        raise ValueError("Incomplete independent k-space")
    image = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(k, axes=(1, 2, 3)), axes=(1, 2, 3)), axes=(1, 2, 3))
    return raw.ravel(), image


def error_metrics(actual, expected, mask):
    if not np.any(mask):
        return dict(voxels=0, vector_rmse_m_s=None, axial_bias_m_s=None)
    difference = actual[:, mask]-expected[:, mask]
    return dict(voxels=int(mask.sum()), vector_rmse_m_s=float(np.sqrt(np.mean(np.sum(difference**2, axis=-1)))),
                axial_bias_m_s=float(np.mean(difference[..., 0])))


def analyze(repo, run, output):
    if output.exists() or output.with_name(output.name+".building").exists():
        raise FileExistsError(output)
    stage = output.with_name(output.name+".building"); stage.mkdir(parents=True)
    source = run/"inputs_v1/controls.h5"
    wave = run/"inputs/sequences_v1/waveforms.h5"
    if not wave.exists():
        wave = run/"sequences_v1/waveforms.h5"
    acquisition=tomllib.loads((run/'acquisition_v1/report.toml').read_text())
    sequence=tomllib.loads((run/'sequences_v1/report.toml').read_text())
    input_registration=json.loads((run/'inputs_v1/registration.json').read_text())
    if acquisition['status']!='COMPLETE_E056_SYNTHETIC_3D_ACQUISITIONS':
        raise ValueError('Incomplete acquisition')
    if sha(source)!=acquisition['input_sha256'] or sha(source)!=input_registration['input_sha256']:
        raise ValueError('Control input changed')
    if sha(wave)!=acquisition['waveform_sha256'] or sha(wave)!=sequence['waveforms_sha256']:
        raise ValueError('Waveform export changed')
    with h5py.File(source) as f:
        positions = np.asarray(f["positions"], float)
        if positions.shape[1] != 3 and positions.shape[0] == 3:
            positions = positions.T
        rho = np.asarray(f["rho"], float).ravel()
        t1, t2 = [np.asarray(f[k], float).ravel() for k in ("T1", "T2")]
        if not np.all(t1 == t1[0]) or not np.all(t2 == t2[0]):
            raise ValueError("This reference requires declared uniform T1/T2")
        omega, phi, echo, period = [float(f[k][()]) for k in
                                  ("omega_s_inv", "phi_rad", "echo_baseline_s", "motion_period_s")]
        path_time = np.asarray(f["path_time_s"], float)
        model_u = {m: np.asarray(f["u_"+m], float).ravel() for m in MODELS}
    factor = (np.sin(omega*(path_time-echo)+phi)-np.sin(phi))/omega
    if not np.isclose(factor[0], factor[-1], atol=1e-12, rtol=0):
        raise ValueError("Control path is not continuous at its seam")
    nx, ny, nz, bins = shape = (80, 16, 32, 15)
    spacing = np.array([.00125, .00125, .0025])
    x = (np.arange(nx)-nx//2)*spacing[0]
    y = (np.arange(ny)-ny//2)*spacing[1]
    z = (np.arange(nz)-nz//2)*spacing[2]
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    radius = .004579049785927147
    radial = yy**2+zz**2
    rows, raw_checks, fields, sources = [], [], {}, [source, wave, Path(__file__).resolve()]
    with h5py.File(wave) as f, h5py.File(stage/"independent_fields.h5", "w") as saved:
        arms = [name for name in f if isinstance(f[name],h5py.Group)]
        if len(arms) != 3:
            raise ValueError("Exactly three preregistered acquisition arms required")
        references = {}
        for arm in arms:
            references[arm] = {}
            for enc in ENCODINGS:
                group = f[arm][enc] if enc in f[arm] else f[arm][enc.lower()]
                references[arm][enc] = load_group(group, nx, ny, nz)
                references[arm][enc]['gamma']=float(f['gamma_Hz_T'][()])
        for model in MODELS:
            fields[model] = {}
            for arm in arms:
                images, predicted_images = {}, {}
                for enc in ENCODINGS:
                    print(f"E056 independent reference {model} {arm} {enc}", flush=True)
                    metadata = references[arm][enc]
                    predicted_raw, predicted_image = predict(metadata, positions, rho, model_u[model],
                        float(t1[0]), float(t2[0]), path_time, factor, period, shape)
                    result = run/"acquisition_v1"/f"{model}_{arm}_{enc.lower()}.h5"
                    sources.append(result)
                    key=f'{model}_{arm}_{enc.lower()}'
                    if sha(result)!=acquisition['result_sha256'][key]:
                        raise ValueError('Acquired result changed: '+key)
                    with h5py.File(result) as actual:
                        raw = complex_array(actual["raw_signal_f64"]).ravel()
                        images[enc] = complex_array(actual["image"])
                    if images[enc].shape != (bins, nz, ny, nx) or raw.shape != predicted_raw.shape:
                        raise ValueError("Unexpected scientific result shape")
                    if not np.isfinite(raw).all() or not np.isfinite(images[enc]).all():
                        raise ValueError('Nonfinite raw signal or image')
                    actual_k=np.zeros((bins,nz,ny,nx),np.complex128)
                    lab=metadata['labels'];actual_k[lab[:,2],lab[:,1],lab[:,0]]=raw.reshape(-1,nx)
                    independent_fft=np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(actual_k,axes=(1,2,3)),axes=(1,2,3)),axes=(1,2,3))
                    fft_error=float(np.max(abs(independent_fft-images[enc]))/max(np.max(abs(independent_fft)),1e-30))
                    if fft_error>3e-6:raise ValueError('Independent reconstruction identity failed')
                    scale = np.linalg.norm(predicted_raw)
                    l2 = float(np.linalg.norm(raw-predicted_raw)/scale)
                    image_l2 = float(np.linalg.norm(images[enc]-predicted_image)/np.linalg.norm(predicted_image))
                    raw_checks.append(dict(model=model, arm=arm, encoding=enc, raw_relative_l2=l2,
                        image_relative_l2=image_l2,independent_fft_relative_max=fft_error,
                        passed_numerical_gate=bool(l2 <= .01 and image_l2 <= .01)))
                    g = saved.require_group(f"{model}/{arm}/{enc}")
                    g["raw_signal_independent"] = predicted_raw
                    g["image_independent"] = predicted_image
                    predicted_images[enc] = predicted_image
                velocity = np.stack([np.angle(images[e]*images["REF"].conj())*5/np.pi for e in ENCODINGS[1:]], axis=-1)
                independent_velocity = np.stack([np.angle(predicted_images[e]*predicted_images["REF"].conj())*5/np.pi
                                                 for e in ENCODINGS[1:]], axis=-1)
                magnitude = np.stack([abs(images[e]) for e in ENCODINGS])
                support = np.all(magnitude >= .05*magnitude.max(axis=(2, 3, 4), keepdims=True), axis=(0, 1))
                metadata = references[arm]["REF"]
                rf_index = int(np.asarray(metadata["profile_rf_index0"]).ravel()[0])
                shot = metadata["templates"][0]
                actual_echo = float(np.asarray(metadata["rf_start_s"]).ravel()[rf_index]+shot["adc"][nx//2])
                angle = omega*(actual_echo-echo)+phi
                if model == "static":
                    speed = np.zeros_like(xx)
                elif model == "uniform_osc":
                    speed = np.full_like(xx, 2.2)
                else:
                    speed = 2.2*np.maximum(0., 1-radial/radius**2)
                displacement = speed*(np.sin(angle)-np.sin(phi))/omega
                geometry = (radial <= radius**2) & (abs(xx-displacement) < .005)
                core = (radial <= (radius-.00125)**2) & (abs(xx-displacement) <= .0025)
                ideal = np.zeros((bins, nz, ny, nx, 3))
                ideal[..., 0] = speed*np.cos(angle)
                fields[model][arm] = dict(velocity=velocity, independent=independent_velocity,
                    support=support, geometry=geometry, core=core, ideal=ideal, echo=actual_echo,
                    magnitude=images["REF"])
            common_core = np.logical_and.reduce([fields[model][a]["core"] for a in arms])
            common_geometry = np.logical_and.reduce([fields[model][a]["geometry"] for a in arms])
            paired = common_core & np.logical_and.reduce([fields[model][a]["support"] for a in arms])
            for arm in arms:
                data = fields[model][arm]
                own = data["core"] & data["support"]
                row = dict(model=model, arm=arm, actual_echo_s=data["echo"],
                    geometric_voxels=int(data["geometry"].sum()), predefined_core_voxels=int(data["core"].sum()),
                    own_geometry_coverage=float(np.sum(data["support"] & data["geometry"])/data["geometry"].sum()),
                    own_core_coverage=float(own.sum()/data["core"].sum()),
                    identical_paired_core_coverage=float(paired.sum()/common_core.sum()),
                    Koma_vs_instantaneous_analytic=error_metrics(data["velocity"], data["ideal"], paired),
                    independent_image_vs_instantaneous_analytic=error_metrics(data["independent"], data["ideal"], paired),
                    Koma_vs_independent_complex_model=error_metrics(data["velocity"], data["independent"], paired))
                rows.append(row)
                g = saved.require_group(f"{model}/{arm}/analysis")
                for key in ("support", "geometry", "core"):
                    g[key] = data[key].astype(np.uint8)
                g["paired_core"] = paired.astype(np.uint8)
                g["velocity_Koma"] = data["velocity"]
                g["velocity_independent"] = data["independent"]
                g["velocity_instantaneous_analytic"] = data["ideal"]
            saved.require_group(model)["common_geometry"] = common_geometry.astype(np.uint8)
    numerical_pass = all(r["passed_numerical_gate"] for r in raw_checks)
    report = dict(experiment="E-056", created_utc=datetime.now(timezone.utc).isoformat(),
        status="PASS_INDEPENDENT_REPEATED_RF_COMPLEX_MODEL" if numerical_pass else "FAIL_INDEPENDENT_COMPLEX_MODEL_GATE",
        numerical_checks=raw_checks, comparisons=rows, new_1m_started=False,
        mask="Unchanged 5% all-bin/all-encoding magnitude support; all-arm paired predefined analytical interior",
        limitations=["Analytical stroboscopic accelerating/sheared controls, not CFD-driven stenosis accuracy or a physiological waveform.",
            "Independent numerical Bloch and Fourier implementation shares exported physical waveform inputs; not independent physical validation.",
            "Actual prescribed path is piecewise-linear and continuous; no inlet, recycling, cycle mapping or wall interaction is introduced.",
            "A shorter-TE arm samples a slightly different actual motion phase; analytic instantaneous reference follows its own actual TE.",
            "No fitted complex gain, velocity scale, phase shift, mask relaxation or missing-phase imputation."])
    write_json(stage/"report.json", report)
    write_json(stage/"validation.json", dict(status="PASS" if numerical_pass else "FAIL", checks=len(raw_checks)*2,
        raw_and_image_checks=raw_checks, report_sha256=sha(stage/"report.json"), no_new_1m=True))
    (stage/"README.md").write_text("# Acquisition-physics control\n\n"+report["status"]+"\n\n"
        "Numerical Koma/independent-complex-model agreement is separate from error against the instantaneous analytical field. "
        "These are synthetic mechanism controls, not a corrected CFD phantom run or independent physical validation. "
        "See report.json for identical-support errors and retained coverage. No 1M run was started.\n", encoding="utf-8")
    write_json(stage/"MANIFEST.json", dict(inputs={str(p): dict(bytes=p.stat().st_size, sha256=sha(p)) for p in sources},
        outputs={p.name: dict(bytes=p.stat().st_size, sha256=sha(p)) for p in stage.iterdir() if p.is_file()}))
    stage.rename(output)
    print(json.dumps(dict(status=report["status"], rows=len(rows), numerical_checks=len(raw_checks), new_1m_started=False)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    analyze(arguments.repo, arguments.run, arguments.output)
