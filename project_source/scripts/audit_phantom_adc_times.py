"""E049 compare saved ADC timestamps with every Pulseq dwell center."""
import argparse
import json
from pathlib import Path
import tomllib
import uuid
import h5py
import numpy as np
from scripts.aorta_hemodynamic_reference import read_pulseq
from scripts.build_aorta_inlet_picker import sha
from scripts.diagnose_phantom_signal import E047,ROOT


def audit(output):
    if output.exists():raise FileExistsError(output)
    stage=output.with_name(output.name+'.building-'+uuid.uuid4().hex[:8]);stage.mkdir(parents=True)
    info=tomllib.loads((E047/'phantom_inputs_v1/case_info.toml').read_text())
    seqdir=Path(info['sequence_directory']);timing=E047/'phantom_inputs_v1/legacy_timing.h5'
    rows=[];sources=[Path(__file__).resolve(),timing]
    with h5py.File(timing) as f,h5py.File(stage/'timing_checks.h5','w') as out:
        for e in ('REF','X','Y','Z'):
            source=seqdir/(e.lower()+'.seq');sources.append(source)
            parsed=read_pulseq(source)
            defs={r[0]:r[1:] for r in parsed['DEFINITIONS']}
            raster=float(defs['BlockDurationRaster'][0])
            adcs={int(r[0]):r for r in parsed['ADC']}
            first=[];dwells=[];counts=[];start=0.
            for b in parsed['BLOCKS']:
                if int(b[6]):
                    adc=adcs[int(b[6])];d=float(adc[2])*1e-9
                    first.append(start+float(adc[3])*1e-6+.5*d)
                    dwells.append(d);counts.append(int(adc[1]))
                start+=int(b[1])*raster
            actual=np.array(first);saved=f[e+'/adc_first_s'][:]
            delta=saved-actual
            maximum=float(np.max(abs(delta)))
            if maximum>1e-7 or not np.allclose(dwells,f[e+'/dwell_s'][:],atol=1e-14,rtol=0):
                raise AssertionError('Saved ADC timestamps do not match actual Pulseq dwell centers')
            if not np.array_equal(counts,f[e+'/n'][:]):raise AssertionError('Sample count mismatch')
            out[e+'_saved_minus_Pulseq_actual_s']=delta
            rows.append(dict(encoding=e,profiles=len(actual),maximum_timestamp_delta_s=maximum,
                             first_saved_s=float(saved[0]),first_Pulseq_actual_s=float(actual[0]),
                             half_dwell_already_in_saved_first_time=True))
    report=dict(experiment='E-049',status='PASS_SAVED_ADC_TIMING_AUDIT',encodings=rows,
                apply_extra_half_dwell=False,
                conclusion='The saved E047/E048 acquisition timestamps already include half-dwell centering. Only the raw Pulseq proxy parser omitted it. Do not add another offset.',
                earlier_offset_bound='adc_bound_v1 is a hypothetical additional-offset sensitivity, NOT evidence of an error in the saved ADC-averaged CFD comparator.')
    (stage/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (stage/'MANIFEST.json').write_text(json.dumps(dict(inputs={str(p):sha(p) for p in sources},
        outputs={p.name:sha(p) for p in stage.iterdir() if p.is_file()}),indent=2)+'\n')
    stage.rename(output);print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    audit(p.parse_args().output)
