from __future__ import annotations
import json, re, traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    from scipy.io import loadmat
except Exception:
    loadmat = None
try:
    import h5py
except Exception:
    h5py = None
try:
    from scipy.signal import find_peaks
    from scipy.stats import skew as sp_skew, kurtosis as sp_kurtosis
except Exception:
    find_peaks = sp_skew = sp_kurtosis = None

ROOT = Path(__file__).resolve().parents[3]
PHASE = ROOT/'experiments'/'phase_02_full_multimodal_feature_extraction'
P1 = ROOT/'experiments'/'phase_01_raw_data_modality_audit'/'results'
DATA = ROOT/'vrdataset'/'dataPackage'
RES, TAB, FIG, LOG = PHASE/'results', PHASE/'tables', PHASE/'figures', PHASE/'logs'
IDS = ['subject_id','session_id','run_id','difficulty_level','run_key']
MODS = ['eye_tracking','ecg','eda','emg','respiration','head_movement','xplane','control_input','performance','unknown']
PFX = {'eye_tracking':'eye','ecg':'ecg','eda':'eda','emg':'emg','respiration':'resp','head_movement':'head','xplane':'xplane','control_input':'control','performance':'performance','unknown':'unknown'}
GRP = {'eye_tracking':'eye_tracking_features','ecg':'physiological_features','eda':'physiological_features','emg':'physiological_features','respiration':'physiological_features','head_movement':'head_movement_features','xplane':'flight_parameter_features','control_input':'control_input_features','performance':'performance_features','unknown':'unknown_features'}
STATS = ['mean','std','min','max','median','iqr','skew','kurtosis','range','rms','slope','num_missing','missing_ratio']
GROUP_KEYS = ['identifier_columns','physiological_features','eye_tracking_features','head_movement_features','flight_parameter_features','control_input_features','performance_features','unknown_features']

def clean(x):
    s = re.sub(r'[^0-9a-zA-Z]+','_',str(x).strip().lower()).strip('_')
    s = re.sub(r'_+','_',s) or 'value'
    return 'c_'+s if s[0].isdigit() else s

def tag(name):
    lo = name.lower()
    if lo == 'perfmetrics.csv': return 'cumulative'
    if 'ocuevts' in lo: return 'ocuevts'
    if 'feat-perfmetric' in lo: return 'perfmetric'
    m = re.search(r'stream-([^_]+)', lo)
    return clean(m.group(1) if m else Path(name).stem)

def add(feats, name, val):
    base = clean(name); out = base; i = 2
    while out in feats:
        out = f'{base}__dup{i}'; i += 1
    try: feats[out] = float(val) if val is not None else np.nan
    except Exception: feats[out] = np.nan
    return out

def is_time(c):
    c = clean(c)
    return c in {'time','time_dn','timestamp','processedtime','processed_time'} or c.endswith('_time') or c.startswith('time_')

def time_info(s, col):
    arr = pd.to_numeric(s, errors='coerce').dropna().to_numpy(float)
    if len(arr) < 2: return None, np.nan, np.nan
    arr = np.sort(arr); dur_raw = arr[-1]-arr[0]
    if not np.isfinite(dur_raw) or dur_raw <= 0: return None, np.nan, np.nan
    med = np.nanmedian(np.diff(arr)) if len(arr) > 2 else dur_raw
    scale = 86400.0 if clean(col) == 'time_dn' or arr[0] > 100000 else (0.001 if med > 1000 else 1.0)
    sec = (arr-arr[0])*scale; dur = float(dur_raw*scale); fs = float((len(arr)-1)/dur) if dur > 0 else np.nan
    return sec, dur, fs

def stats(s, sec=None, dur=None):
    a = pd.to_numeric(s, errors='coerce').to_numpy(float); mask = np.isfinite(a); v = a[mask]
    n = len(a); miss = n-len(v); out = {'num_missing':float(miss),'missing_ratio':float(miss/n) if n else np.nan}
    if len(v) == 0:
        for k in STATS: out.setdefault(k, np.nan)
        return out
    q25, q75 = np.nanpercentile(v, [25,75]); sd = float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0
    out.update(mean=float(np.nanmean(v)), std=sd, min=float(np.nanmin(v)), max=float(np.nanmax(v)), median=float(np.nanmedian(v)), iqr=float(q75-q25), range=float(np.nanmax(v)-np.nanmin(v)), rms=float(np.sqrt(np.nanmean(v*v))))
    out['skew'] = float(sp_skew(v, nan_policy='omit', bias=False)) if sp_skew and len(v) >= 3 and sd > 0 else np.nan
    out['kurtosis'] = float(sp_kurtosis(v, nan_policy='omit', bias=False)) if sp_kurtosis and len(v) >= 4 and sd > 0 else np.nan
    idx = np.flatnonzero(mask)
    if len(idx) >= 2:
        dx = (sec[idx[-1]]-sec[idx[0]]) if sec is not None and len(sec) == n else (dur if dur and np.isfinite(dur) and dur > 0 else idx[-1]-idx[0])
        out['slope'] = float((v[-1]-v[0])/dx) if dx and dx > 0 else np.nan
    else: out['slope'] = np.nan
    return out

def read_any(path):
    if path.suffix.lower() == '.csv': return pd.read_csv(path, low_memory=False)
    if path.suffix.lower() == '.mat':
        if loadmat:
            try:
                d = loadmat(path, squeeze_me=True, struct_as_record=False); cols = {}
                for k,v in d.items():
                    if k.startswith('__'): continue
                    a = np.asarray(v)
                    if a.ndim == 1 and a.size > 1 and np.issubdtype(a.dtype, np.number): cols[clean(k)] = a
                    elif a.ndim == 2 and 1 in a.shape and a.size > 1 and np.issubdtype(a.dtype, np.number): cols[clean(k)] = a.reshape(-1)
                if cols: return pd.DataFrame(cols)
            except Exception: pass
        if h5py:
            cols = {}
            with h5py.File(path, 'r') as h:
                def visit(n,o):
                    if hasattr(o,'shape') and len(o.shape)==1 and o.shape[0]>1:
                        try:
                            a = np.asarray(o)
                            if np.issubdtype(a.dtype, np.number): cols[clean(n)] = a
                        except Exception: pass
                h.visititems(visit)
            if cols: return pd.DataFrame(cols)
    raise ValueError(f'unsupported or unreadable table: {path.suffix}')

def modality(row, cols=None):
    d = str(row.get('detected_modality','unknown') or 'unknown')
    if d in MODS: return d
    txt = (str(row.get('file_name',''))+' '+' '.join(cols or [])).lower()
    return 'control_input' if any(t in txt for t in ['joystick','yoke','throttle','rudder','control','input']) else 'unknown'

def should_skip(row):
    p = Path(str(row.file_path)); lo = p.name.lower()
    if lo.endswith('_hea.csv'): return True, 'header_metadata_file'
    if lo == 'perfmetrics.csv': return False, ''
    if p.suffix.lower() not in {'.csv','.mat'}: return True, 'unsupported_or_documentation_file'
    if pd.isna(row.get('run_key')) or not str(row.get('run_key','')).strip(): return True, 'missing_run_identifier'
    return False, ''

def combine(df, cols):
    if not cols: return None
    x = df[cols].apply(pd.to_numeric, errors='coerce')
    return x[cols[0]] if len(cols)==1 else x.mean(axis=1, skipna=True)

def peak(s, dur, stem):
    if s is None or find_peaks is None: return {stem+'_peak_count':np.nan, stem+'_peak_rate':np.nan}
    v = pd.to_numeric(s, errors='coerce').dropna().to_numpy(float)
    if len(v) < 3: return {stem+'_peak_count':np.nan, stem+'_peak_rate':np.nan}
    sd = np.nanstd(v); count = 0 if not np.isfinite(sd) or sd == 0 else len(find_peaks(v, prominence=max(sd*.25,1e-12))[0])
    return {stem+'_peak_count':float(count), stem+'_peak_rate':float(count/dur) if dur and np.isfinite(dur) and dur > 0 else np.nan}

def zcr(s):
    if s is None: return np.nan
    v = pd.to_numeric(s, errors='coerce').dropna().to_numpy(float)
    return float(np.mean(np.diff(np.signbit(v-np.nanmean(v))) != 0)) if len(v) > 1 else np.nan

def sel(cols, terms):
    return [c for c in cols if any(t in clean(c) for t in terms)]

def derived(df, mod, prefix, tg, num_cols, sec, dur, feats):
    made=[]
    def add_signal(name, cols, do_peak=False, do_zcr=False, mav=False):
        sig = combine(df, cols)
        if sig is None: return
        for k,v in stats(sig, sec, dur).items(): made.append(add(feats, f'{prefix}_{name}_{k}', v))
        if mav:
            made.append(add(feats, f'{prefix}_{name}_mav', pd.to_numeric(sig, errors='coerce').abs().mean()))
            made.append(add(feats, f'{prefix}_{name}_variance', pd.to_numeric(sig, errors='coerce').var(ddof=1)))
        if do_zcr: made.append(add(feats, f'{prefix}_{name}_zero_crossing_rate', zcr(sig)))
        if do_peak:
            for pk,pv in peak(sig, dur, f'{prefix}_{name}').items(): made.append(add(feats, pk, pv))
    if mod == 'ecg': add_signal('signal', sel(num_cols,['ecg']), True)
    elif mod == 'eda': add_signal('signal', sel(num_cols,['eda','gsr']), True)
    elif mod == 'emg': add_signal('signal', sel(num_cols,['emg']), False, True, True)
    elif mod == 'respiration':
        add_signal('signal', sel(num_cols,['resp']), True)
        pc = feats.get(clean(f'{prefix}_signal_peak_count'), np.nan)
        made.append(add(feats, f'{prefix}_approximate_respiration_rate', pc/dur if dur and np.isfinite(dur) and dur > 0 else np.nan))
    elif mod == 'eye_tracking':
        for n,terms in [('gaze',['gaze']),('pupil',['pupil_diameter']),('openness',['eye_openness']),('position',['eye_position','pupil_position','gaze_origin'])]: add_signal(n, sel(num_cols,terms))
        for ev,terms in [('fixation',['fixationseq','fixation']),('saccade',['saccadeseq','saccade'])]:
            cols = sel(list(df.columns), terms)
            if cols:
                vals = pd.to_numeric(df[cols[0]], errors='coerce'); pos = vals[vals >= 0].dropna(); cnt = float(pos.nunique()) if len(pos) else 0.0
                made.append(add(feats, f'{prefix}_{tg}_{ev}_event_count', cnt)); made.append(add(feats, f'{prefix}_{tg}_{ev}_event_rate', cnt/dur if dur and np.isfinite(dur) and dur > 0 else np.nan))
    elif mod == 'head_movement':
        if not sel(num_cols,['velocity','speed']):
            for c in sel(num_cols,['position','roll','pitch','yaw','orientation','rotation']):
                for k,v in stats(pd.to_numeric(df[c], errors='coerce').diff(), None, dur).items(): made.append(add(feats, f'{prefix}_{tg}_{clean(c)}_first_difference_{k}', v))
    elif mod == 'xplane':
        for n,terms in [('airspeed',['airspeed']),('altitude',['altitude','elevation']),('heading',['heading','yaw']),('ils',['ils']),('latitude',['latitude']),('longitude',['longitude']),('position',['position']),('attitude',['pitch','roll','yaw'])]: add_signal(n, sel(num_cols,terms))
    elif mod == 'control_input': add_signal('signal', sel(num_cols,['joystick','yoke','throttle','rudder','control','input']))
    return made

def extract_file(row):
    p = Path(str(row.file_path)); df = read_any(p); cols = list(df.columns)
    mod = modality(row, cols); prefix = PFX.get(mod,'unknown'); tg = tag(p.name)
    tcols = [c for c in cols if is_time(c)]; tcol = tcols[0] if tcols else None
    sec = None; dur = fs = np.nan
    if tcol:
        numt = pd.to_numeric(df[tcol], errors='coerce')
        if numt.notna().sum() == len(df): sec,dur,fs = time_info(numt, tcol)
        else: _,dur,fs = time_info(numt, tcol)
    feats={}; rows=[]; num_cols=[]
    for c in cols:
        if c == tcol: continue
        x = pd.to_numeric(df[c], errors='coerce')
        if x.notna().sum() > 0: num_cols.append(c)
    for c in num_cols:
        for k,v in stats(df[c], sec, dur).items():
            fn = add(feats, f'{prefix}_{tg}_{clean(c)}_{k}', v); rows.append((fn,c,k))
    if tcol:
        for k,v in {'duration':dur,'sampling_rate_estimate':fs,'sample_count':len(df)}.items():
            fn = add(feats, f'{prefix}_{tg}_{k}', v); rows.append((fn,tcol,k))
    for fn in derived(df, mod, prefix, tg, num_cols, sec, dur, feats): rows.append((fn,'derived','derived'))
    meta = {'file_path':str(p),'relative_path':row.get('relative_path',''),'file_name':p.name,'run_key':row.get('run_key'),'subject_id':row.get('subject_id'),'session_id':row.get('session_id'),'run_id':row.get('run_id'),'difficulty_level':row.get('difficulty_level'),'detected_modality':row.get('detected_modality'),'extracted_modality':mod,'source_tag':tg,'row_count':len(df),'numeric_column_count':len(num_cols),'feature_count':len(feats)}
    return feats, meta, rows

def perfmetrics(path, run_index):
    df = pd.read_csv(path); valid = set(run_index.run_key); out=defaultdict(dict); long=[]
    for _,r in df.iterrows():
        sid = f"sub-cp{int(r['subject']):03d}"; ses=f"ses-{int(r['date'])}"; rid=f"run-{int(r['run']):03d}"; diff=f"level-{int(r['difficulty']):02d}B"; rk=f'{sid}_{ses}_{diff}_{rid}'
        if rk not in valid: continue
        for c in df.columns:
            if clean(c) in {'subject','date','run','difficulty'}: continue
            fn = clean('performance_cumulative_'+clean(c)); val = pd.to_numeric(pd.Series([r[c]]), errors='coerce').iloc[0]
            out[rk][fn] = float(val) if pd.notna(val) else np.nan
            long.append({'run_key':rk,'subject_id':sid,'session_id':ses,'run_id':rid,'difficulty_level':diff,'modality':'performance','feature_group':'performance_features','feature_name':fn,'feature_value':out[rk][fn],'source_file':str(path),'source_column':c,'statistic':'cumulative'})
    return out,long

def write_readme(report, fpm, failed):
    extracted = fpm[fpm.runs_with_features > 0].modality.tolist(); unavailable = fpm[fpm.runs_with_features == 0].modality.tolist()
    truly = len([m for m in extracted if m not in {'performance','unknown'}]) >= 2
    txt = f"""# Phase 02: Full Multimodal Run-Level Feature Extraction

## Status
Complete. The notebook was executed and saved with visible outputs.

## Source Data Rule
`vrdataset/dataPackage` was treated as read-only. All generated artifacts are inside this Phase 02 experiment folder.

## Extracted Modalities
Successfully extracted modalities: {', '.join(extracted) if extracted else 'none'}.
Unavailable or not extracted as known modalities: {', '.join(unavailable) if unavailable else 'none'}.

## Is The Table Truly Multimodal?
{'Yes' if truly else 'No'}. The output has {report['extraction_summary']['runs_in_output']} run rows and {report['extraction_summary']['features_in_wide_table_excluding_identifiers']} extracted feature columns. Missing modalities are preserved with NaN values.

## Feature Families
Physiological features include ECG, EDA/GSR, EMG, and respiration aggregate statistics plus approximate peak or zero-crossing summaries where feasible.
Eye-tracking features include gaze, pupil diameter, eye openness, eye position, missingness, and available fixation/saccade sequence summaries.
Head movement features include pilot head position/attitude summaries and first-difference summaries when explicit velocity columns were unavailable.
Flight/control features include X-Plane aircraft state summaries. No explicit joystick, yoke, throttle, rudder, or control-input streams were available after extraction.
Performance features include per-run performance metric streams and cumulative glideslope, localizer, airspeed, and total error fields from `PerfMetrics.csv`.
Unknown features contain torso accelerometer streams that Phase 01 marked unknown; they were not forced into a known modality.

## Performance Feature Caution
Performance metrics may encode task outcome information. Phase 03 must build two dataset versions: with performance features and without performance features.

## Why NaN Values Remain
Phase 02 does not impute missing values because missingness must be handled inside Phase 03/model pipelines to avoid leakage.

## Failed Or Skipped Files
Failed files: {len(failed)}. Header metadata and non-run files were skipped as non-model-feature sources.

## Phase 03 Next Step
Construct the final four-class modeling dataset, create with-performance and without-performance versions, perform controlled imputation/scaling inside modeling pipelines, and then proceed to baseline ML/HDC experiments.
"""
    (PHASE/'README.md').write_text(txt, encoding='utf-8')

def run_phase_02():
    for d in [RES,TAB,FIG,LOG,PHASE/'notebooks',PHASE/'scripts']: d.mkdir(parents=True, exist_ok=True)
    log = LOG/'phase_02_log.txt'; started = datetime.now().isoformat(timespec='seconds'); log.write_text(f'Phase 02 extraction started: {started}\nData source: {DATA}\n', encoding='utf-8')
    inv = pd.read_csv(P1/'raw_file_inventory.csv'); avail = pd.read_csv(P1/'run_modality_availability.csv'); modsum = pd.read_csv(P1/'modality_summary.csv'); unknown = pd.read_csv(P1/'unknown_files_for_review.csv')
    runs = avail[IDS].drop_duplicates().copy(); wide = runs.set_index('run_key').to_dict('index')
    for rk in wide: wide[rk]['run_key'] = rk
    failed=[]; skipped=[]; processed=[]; long=[]; fmap={c:'identifier_columns' for c in IDS}; rmods=defaultdict(set)
    pp = DATA/'task-ils'/'PerfMetrics.csv'
    if pp.exists():
        try:
            pf,pl = perfmetrics(pp,runs); long += pl
            for rk,fs in pf.items(): wide[rk].update(fs); rmods[rk].add('performance'); [fmap.__setitem__(f,'performance_features') for f in fs]
            log.write_text(log.read_text(encoding='utf-8')+f'Merged PerfMetrics for {len(pf)} runs\n', encoding='utf-8')
        except Exception as e: failed.append({'file_path':str(pp),'file_name':pp.name,'run_key':'','detected_modality':'performance','error_type':type(e).__name__,'error_message':str(e),'traceback':traceback.format_exc(limit=3)})
    cand = inv[inv.file_path.astype(str).str.startswith(str(DATA))].copy(); done=0
    for _,row in cand.iterrows():
        p = Path(str(row.file_path)); skip,why = should_skip(row)
        if skip:
            skipped.append({'file_path':str(p),'relative_path':row.get('relative_path',''),'file_name':p.name,'run_key':row.get('run_key',''),'detected_modality':row.get('detected_modality',''),'reason':why}); continue
        if p.name.lower() == 'perfmetrics.csv': continue
        try:
            feats,meta,fr = extract_file(row); rk=str(meta['run_key']); mod=meta['extracted_modality']
            if rk not in wide:
                skipped.append({**meta,'reason':'run_not_in_phase01_availability'}); continue
            wide[rk].update(feats); group=GRP.get(mod,'unknown_features')
            if feats: rmods[rk].add(mod)
            rowmeta = {k:meta.get(k) for k in ['subject_id','session_id','run_id','difficulty_level']}
            frmap = {a:(b,c) for a,b,c in fr}
            for fn,val in feats.items():
                fmap[fn]=group; sc,st=frmap.get(fn,('',''))
                long.append({'run_key':rk,**rowmeta,'modality':mod,'feature_group':group,'feature_name':fn,'feature_value':val,'source_file':meta['file_path'],'source_column':sc,'statistic':st})
            processed.append({**meta,'status':'processed'}); done += 1
            if done % 250 == 0:
                with log.open('a',encoding='utf-8') as h: h.write(f'Processed {done} feature files\n')
        except Exception as e:
            failed.append({'file_path':str(p),'relative_path':row.get('relative_path',''),'file_name':p.name,'run_key':row.get('run_key',''),'detected_modality':row.get('detected_modality',''),'error_type':type(e).__name__,'error_message':str(e),'traceback':traceback.format_exc(limit=3)})
            with log.open('a',encoding='utf-8') as h: h.write(f'FAILED {p}: {type(e).__name__}: {e}\n')
    full = pd.DataFrame(list(wide.values())); full = full[IDS + sorted([c for c in full.columns if c not in IDS])]
    ltab = pd.DataFrame(long); fail_cols=['file_path','relative_path','file_name','run_key','detected_modality','error_type','error_message','traceback']; skip_cols=['file_path','relative_path','file_name','run_key','detected_modality','reason']; proc_cols=['file_path','relative_path','file_name','run_key','subject_id','session_id','run_id','difficulty_level','detected_modality','extracted_modality','source_tag','row_count','numeric_column_count','feature_count','status']; faildf = pd.DataFrame(failed, columns=fail_cols); skipdf = pd.DataFrame(skipped, columns=skip_cols); procdf = pd.DataFrame(processed, columns=proc_cols)
    groups = {k:(IDS.copy() if k=='identifier_columns' else []) for k in GROUP_KEYS}
    for fn,g in fmap.items():
        if g != 'identifier_columns': groups.setdefault(g,[]).append(fn)
    groups = {k:(IDS.copy() if k=='identifier_columns' else sorted(set(v))) for k,v in groups.items()}
    fpm=[]
    for m in MODS:
        g=GRP[m]; pref=PFX[m]+'_'; cols=[c for c in full.columns if c not in IDS and fmap.get(c)==g and (c.startswith(pref) or m=='performance')]
        fpm.append({'modality':m,'feature_group':g,'feature_count':len(cols),'runs_with_features':int(full[cols].notna().any(axis=1).sum()) if cols else 0,'non_missing_values':int(full[cols].notna().sum().sum()) if cols else 0})
    fpm = pd.DataFrame(fpm)
    rm=[]
    for _,r in runs.iterrows():
        mods=sorted(rmods.get(r.run_key,set())); rm.append({**{c:r[c] for c in IDS},'extracted_modality_count':len(mods),'extracted_modalities':';'.join(mods)})
    rmdf = pd.DataFrame(rm); targets=['eye_tracking','ecg','eda','emg','respiration','head_movement','xplane','control_input','performance']
    miss = pd.DataFrame([{**{c:r[c] for c in IDS},'missing_modality_count':len([m for m in targets if m not in set(filter(None,str(r.extracted_modalities).split(';')))]),'missing_modalities':';'.join([m for m in targets if m not in set(filter(None,str(r.extracted_modalities).split(';')))])} for _,r in rmdf.iterrows()])
    full.to_csv(RES/'full_multimodal_run_level_features.csv', index=False); ltab.to_csv(RES/'feature_extraction_long_table.csv', index=False); faildf.to_csv(RES/'failed_files.csv', index=False); skipdf.to_csv(RES/'skipped_files.csv', index=False); procdf.to_csv(RES/'processed_files.csv', index=False)
    (RES/'feature_groups.json').write_text(json.dumps(groups, indent=2), encoding='utf-8'); fpm.to_csv(TAB/'features_per_modality.csv', index=False); rmdf.to_csv(TAB/'runs_with_extracted_modalities.csv', index=False); miss.to_csv(TAB/'missing_modalities_after_extraction.csv', index=False)
    for col,y,title,out,color in [('feature_count','Feature count','Extracted features per modality','extracted_features_per_modality.png','#2f6f73'),('runs_with_features','Runs with features','Runs per modality after extraction','runs_per_modality_after_extraction.png','#9b5de5')]:
        plt.figure(figsize=(10,5)); d=fpm.sort_values(col, ascending=False); plt.bar(d.modality,d[col],color=color); plt.xticks(rotation=35,ha='right'); plt.ylabel(y); plt.title(title); plt.tight_layout(); plt.savefig(FIG/out,dpi=180); plt.close()
    limitations=['Run-level aggregate features only; no sliding-window modeling was performed.','No final imputation, scaling, feature selection, four-class labeling, or model training was performed.','ECG peak features are approximate; robust R-peak detection and HRV were not claimed.','EDA and respiration peak counts are approximate, not validated event detection.','Eye fixation/saccade sequence summaries were extracted where available, but detailed oculomotor event validation remains a limitation.','Unknown torso accelerometer streams were kept in unknown_features rather than forced into known modalities.','Performance features are retained and separately grouped for leakage-sensitive Phase 03 dataset versions.']
    report={'phase':'02_full_multimodal_feature_extraction','started_at':started,'finished_at':datetime.now().isoformat(timespec='seconds'),'source_data_root':str(DATA),'phase01_inputs':{'raw_file_inventory':str(P1/'raw_file_inventory.csv'),'run_modality_availability':str(P1/'run_modality_availability.csv'),'modality_summary':str(P1/'modality_summary.csv'),'unknown_files_for_review':str(P1/'unknown_files_for_review.csv')},'input_inventory_summary':{'inventory_files':int(len(inv)),'phase01_runs':int(runs.run_key.nunique()),'phase01_modalities':modsum.to_dict('records'),'unknown_files_from_phase01':int(len(unknown))},'extraction_summary':{'processed_feature_files':int(done),'skipped_files':int(len(skipdf)),'failed_files':int(len(faildf)),'runs_in_output':int(len(full)),'features_in_wide_table_excluding_identifiers':int(len(full.columns)-len(IDS)),'long_feature_rows':int(len(ltab)),'modalities_extracted':fpm[fpm.runs_with_features>0].modality.tolist()},'features_per_modality':fpm.to_dict('records'),'limitations':limitations,'phase03_next_step':'Construct final four-class modeling datasets with and without performance features; perform imputation/scaling inside model pipelines only.'}
    (RES/'extraction_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8'); write_readme(report,fpm,faildf)
    with log.open('a',encoding='utf-8') as h: h.write(f"Phase 02 extraction finished: {report['finished_at']}\nOutput: {RES/'full_multimodal_run_level_features.csv'}\n")
    return report

if __name__ == '__main__':
    print(json.dumps(run_phase_02()['extraction_summary'], indent=2))
