"""E-065 S0: section flow waveforms of the converted dp122 CFD (inlet, 30 mm upstream, throat, outlet). Read-only."""
import json, sys, hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np, pyvista as pv
PVD = Path('D:/MRSim/outputs/dp122_native_e065/conversion_v1/flow/dp122_flow.pvd')
ZS = {'inlet': -0.10610 + 5e-4, 'up30': -0.030, 'throat': 0.0, 'outlet': 0.36010 - 5e-4}
target = Path(sys.argv[1]); assert not target.exists(); target.mkdir()
ds = ET.parse(PVD).findall('.//DataSet'); t = [float(d.attrib['timestep']) for d in ds]
q = {k: [] for k in ZS}
for d in ds:
    m = pv.read(PVD.parent/d.attrib['file'])
    for k, z in ZS.items():
        s = m.slice(normal=(0, 0, 1), origin=(0, 0, z)); s.point_data['w'] = np.asarray(s.point_data['Velocity'])[:, 2]
        q[k].append(float(s.integrate_data().point_data['w'][0])*1e6)
rep = {'times_s': t, 'flow_ml_s': q, 'summary': {}}
for k, v in q.items():
    a = np.array(v); i = int(a.argmax())
    rep['summary'][k] = dict(peak_ml_s=float(a[i]), peak_time_s=t[i], mean_ml_s=float(a[:-1].mean()), first_ml_s=float(a[0]), last_ml_s=float(a[-1]))
    print(f'{k:7s} peak {a[i]:6.1f} mL/s at {t[i]:.2f} s | mean {a[:-1].mean():6.1f} | t0 {a[0]:5.1f} | t_end {a[-1]:5.1f}')
rep['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(target/'section_flows.json').write_text(json.dumps(rep, indent=1))
