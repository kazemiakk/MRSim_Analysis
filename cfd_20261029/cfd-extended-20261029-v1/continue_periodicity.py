"""Bounded second-cycle continuation through the project STDIO MCP, never initialize."""
from pathlib import Path
from datetime import datetime, timezone
import asyncio, copy, csv, hashlib, json, os, re, sys, time, traceback

OLD = Path(r'D:\FluentAgent\automation\20261022-systole-first-three-case-v1')
BASE = Path(r'D:\FluentAgent\runs\20261023-070441-568099-systole-first-checkpoint-recovery-v1\patient_specific')
sys.path.insert(0, str(OLD))
from run_study import Study
from support import save, update, sha, utc, BudgetReached, licensing, sources_unchanged


def prepare():
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    out = Path(r'D:\FluentAgent\runs') / (stamp + '-patient-specific-cycle2-bounded-v1')
    out.mkdir(exist_ok=False)
    directory = out / 'patient_specific'
    directory.mkdir()
    cfg = json.loads((BASE / 'input-configuration.json').read_text())
    prior = json.loads((BASE / 'progress.json').read_text())
    period = cfg['period_s']
    assert period == .868 and len(prior['accepted_steps']) == 95
    original_profile = Path(cfg['profile_file'])
    pattern = re.compile(r'\(\((\S+) transient (\d+) 1\)\s*\(time\s+([^)]*)\)\s*\((\S+)\s+([^)]*)\)\s*\)')
    text = original_profile.read_text()
    blocks = list(pattern.finditer(text))
    assert len(blocks) == 5 and not pattern.sub('', text).strip()
    expanded = []
    for match in blocks:
        name, size, times, field, values = match.groups()
        ts = list(map(float, times.split())); vs = list(map(float, values.split()))
        assert len(ts) == len(vs) == int(size) and ts == cfg['time_s']
        assert ts[0] == 0 and ts[-1] == period and abs(vs[-1]-vs[0]) < 1e-14
        tt = ts + [period+t for t in ts[1:]]
        vv = vs + vs[1:]
        expanded.append('((%s transient %d 1)\n (time\n %s\n )\n (%s\n %s\n )\n)\n' %
                        (name, len(tt), ' '.join(format(v,'.17g') for v in tt), field, ' '.join(format(v,'.17g') for v in vv)))
    profile = directory / 'exact-waveform-repeat-through-cycle2.prof'
    with profile.open('x', encoding='utf-8') as f: f.write(''.join(expanded))
    original = copy.deepcopy(cfg)
    cfg.pop('continuation', None)
    cfg.pop('startup_iteration_limit', None)
    cfg['source_case'] = str(BASE / 'systole-first-patient_specific-t868000us-CHECKPOINT.cas.h5')
    cfg['source_data'] = str(BASE / 'systole-first-patient_specific-t868000us-CHECKPOINT.dat.h5')
    cfg['source_case_sha256'] = sha(cfg['source_case'])
    cfg['source_data_sha256'] = sha(cfg['source_data'])
    cfg['profile_file'] = str(profile); cfg['profile_sha256'] = sha(profile)
    cfg['time_s'] = original['time_s'] + [period+t for t in original['time_s'][1:]]
    for group in ('inlets','outlets'):
        for key, item in cfg[group].items():
            for field, value in list(item.items()):
                if isinstance(value, list) and len(value) == len(original['time_s']):
                    assert abs(value[-1]-value[0]) < 1e-12
                    item[field] = value + value[1:]
    cfg['initial_time_s'] = period
    cfg['target_time_s'] = 2*period
    cfg['target_times_s'] = [round(period+t, 12) for t in original['target_times_s']]
    cfg['initialized_case'] = str(directory / 'NEVER-INITIALIZE.cas.h5')
    save(directory / 'input-configuration.json', cfg)
    payload = (OLD / 'live_solver.py').read_text()
    payload = payload.replace("ACTION['target_time_s']==.868", "ACTION['target_time_s']==CONFIG['target_time_s']")
    # Deliberately make the old initialization branch unreachable, even by a bad caller.
    payload = "assert ACTION['kind'] != 'initialize', 'Continuation must never initialize'\n" + payload
    with (out / 'continuation-payload.py').open('x', encoding='utf-8') as f: f.write(payload)
    save(out / 'source-preservation.json', {
        'prior_run':str(BASE),'source_case':cfg['source_case'],'source_case_sha256':cfg['source_case_sha256'],
        'source_data':cfg['source_data'],'source_data_sha256':cfg['source_data_sha256'],
        'original_profile':str(original_profile),'original_profile_sha256':sha(original_profile),
        'new_profile_sha256':sha(profile),'waveform_change':'Exact repetition only; first-cycle samples unchanged',
        'starting_flow_time_s':period,'target_flow_time_s':2*period,
        'maximum_total_seconds':6900,'cleanup_reserve_seconds':480,
        'user_authorization':'less than 2 hours total; patient-specific continuation priority',
        'mesh_or_physical_changes':False})
    plan = {'output_directory':str(out),'combined_runtime_limit_s':6900,'cleanup_audit_reserve_s':480,
            'pilot_steps':3,'batch_steps':1,'payload_sha256':sha(OLD/'live_solver.py'),
            'cases':[{'kind':'patient_specific','configuration':str(directory/'input-configuration.json'),'directory':str(directory)}]}
    save(out/'plan.json', plan)
    return out, plan, payload, prior


class Continuation(Study):
    def __init__(self, out, plan, payload, prior):
        super().__init__(out/'plan.json')
        self.payload = payload
        self.repeat_loaded = False
        case = self.cases[0]; cfg = case['cfg']; st = case['state']
        st.update(initialized=True,last_accepted_time_s=.868,last_accepted_case=cfg['source_case'],last_accepted_data=cfg['source_data'])
        case['signature'] = json.loads((BASE/'physical-model-signature.json').read_text())
        self.state.update(cycles_per_case=2,additional_cycles_target=1,
                          source_completed_cycles=1,periodicity_validated=False,
                          scope='Patient-specific continuation only; earlier three-case comparisons unchanged')
        self.prior = prior
        self.progress()

    async def load(self, case):
        await super().load(case)
        if self.repeat_loaded: return
        code = """
import json
assert session.get_fluent_version().value == '26.1.0'
s = session.settings
assert abs(s.solution.run_calculation.transient_controls.flow_time()-.868) < 1e-12
before=s.setup.get_state()
s.file.read_profile(file_name=PROFILE)
assert before == s.setup.get_state(), 'Repeated profile changed setup settings'
assert s.setup.general.solver.time()=='transient'
assert s.setup.models.viscous.model()=='laminar' and not s.setup.models.energy.enabled()
assert s.setup.general.operating_conditions.operating_pressure()==100000
blood=s.setup.materials.fluid['blood-mimicking']
assert blood.density.value()==1060 and abs(blood.viscosity.value()-.0035)<1e-15
print('ADAPTIVE_RESULT='+json.dumps({'version':session.get_fluent_version().value,'time_s':s.solution.run_calculation.transient_controls.flow_time(),'repeat_profiles_loaded':True,'setup_unchanged':True,'initialized':False}))
"""
        from support import unpack
        response = await self.client.call_tool('run_code',arguments={'code':'PROFILE='+repr(case['cfg']['profile_file'])+'\n'+code},read_timeout_seconds=240)
        save(self.out/'profile-repeat-live-audit.json',response.model_dump(mode='json'))
        value = unpack(response)
        assert '"repeat_profiles_loaded": true' in value.get('stdout','')
        self.repeat_loaded = True

    async def batch(self, case, count):
        await super().batch(case, count)
        st = case['state']; steps = st['accepted_steps']
        if len(steps)==3:
            historical = self.prior['accepted_steps'][-20:]
            baseline = sum(s['wall_seconds'] for s in historical)/len(historical)
            pilot = sum(s['wall_seconds'] for s in steps)/len(steps)
            estimate = 1.1 * max(baseline,pilot) * (95-len(steps))
            available = self.deadline-time.time()-self.plan['cleanup_audit_reserve_s']
            receipt = {'accepted_pilot_steps':3,'pilot_mean_seconds':pilot,
                       'historical_late_cycle_mean_seconds':baseline,'remaining_cycle_estimate_seconds':estimate,
                       'available_before_cleanup_seconds':available,'completion_appears_to_fit':estimate<=available}
            save(self.out/'three-step-runtime-estimate.json',receipt)
            print('RUNTIME_ESTIMATE '+json.dumps(receipt),flush=True)
            if estimate > available:
                raise BudgetReached('A complete second cycle does not conservatively fit the approved total budget; pilot checkpoints preserved')

    def finish(self):
        case = self.cases[0]; st = case['state']
        self.state['license_issues'] = licensing(self.out)
        self.state['original_files_unchanged'] = sources_unchanged(case['cfg'])
        receipt = json.loads((self.out/'source-preservation.json').read_text())
        self.state['original_profile_unchanged'] = sha(receipt['original_profile'])==receipt['original_profile_sha256']
        self.state['elapsed_wall_s'] = time.time()-self.started
        self.state['finished_utc'] = utc()
        self.state['additional_cycle_completed'] = st['full_cycle_completed']
        self.state['accepted_additional_steps'] = len(st['accepted_steps'])
        self.state['periodicity_validated'] = False
        self.progress()
        save(case['directory']/'run-result.json',st)
        save(self.out/'study-result.json',self.state)
        with (self.out/'phase-matched-native-reports.csv').open('x',newline='',encoding='utf-8') as f:
            writer=csv.writer(f)
            writer.writerow(['phase_s','cycle2_time_s','cycle1_sac_mean_WSS_Pa','cycle2_sac_mean_WSS_Pa',
                             'cycle1_sac_max_WSS_Pa','cycle2_sac_max_WSS_Pa','relative_sac_mean_difference','step_seconds'])
            for old,new in zip(self.prior['accepted_steps'],st['accepted_steps']):
                a=old['comparison_wall_reports']['aneurysm_sac'];b=new['comparison_wall_reports']['aneurysm_sac']
                writer.writerow([old['time_s'],new['time_s'],a['area_mean_wss_pa'],b['area_mean_wss_pa'],a['max_wss_pa'],b['max_wss_pa'],
                                 abs(b['area_mean_wss_pa']-a['area_mean_wss_pa'])/max(abs(a['area_mean_wss_pa']),1e-12),new['wall_seconds']])
        with (self.out/'README.md').open('x',encoding='utf-8') as f:
            f.write('# Bounded patient-specific second-cycle test\n\n')
            f.write('Status: '+self.state['status']+'\n\n')
            f.write('New accepted steps: '+str(len(st['accepted_steps']))+' / 95.\n\n')
            f.write('Exact repetition of the existing 868-ms waveform, no initialization or changed geometry/physics. ')
            f.write('Existing residual, mass-balance, outlet split and boundary-direction gates retained.\n\n')
            f.write('Partial second-cycle results do not establish periodicity. A completed second cycle still requires phase-matched field analysis, not only native wall reports.\n\n')
            f.write('The previous three-condition comparison and videos have NOT been replaced by these continuation results.\n\n')
            f.write('Elapsed seconds: '+str(self.state['elapsed_wall_s'])+'; owned session closed: '+str(self.state['owned_session_closed'])+'\n')
        print('CONTINUATION_FINISHED '+str(self.out)+' '+self.state['status'],flush=True)


async def main():
    out, plan, payload, prior = prepare()
    print('CONTINUATION_DIRECTORY '+str(out),flush=True)
    study = Continuation(out, plan, payload, prior)
    await study.solve()
    study.finish()


if __name__=='__main__': asyncio.run(main())
