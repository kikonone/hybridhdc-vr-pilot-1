"""Candidate-level, atomic inner-CV checkpoint utilities for Phase 04A."""
from itertools import product
from pathlib import Path
import os
import tempfile
import numpy as np
import pandas as pd

COLUMNS=["outer_fold","candidate_id","k","n_estimators","learning_rate","max_depth","inner_fold_1_macro_f1","inner_fold_2_macro_f1","inner_fold_3_macro_f1","mean_inner_macro_f1","std_inner_macro_f1","status"]
STATUSES={"NOT_STARTED","IN_PROGRESS","COMPLETE","FAILED"}

def candidate_manifest(outer_fold: int=2) -> pd.DataFrame:
    rows=[]
    for candidate_id,(k,n,lr,depth) in enumerate(product([100,200],[100,200],[0.05,0.1],[2]),1):
        rows.append({"outer_fold":outer_fold,"candidate_id":candidate_id,"k":k,"n_estimators":n,"learning_rate":lr,"max_depth":depth})
    return pd.DataFrame(rows)

def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w",suffix=".csv",dir=path.parent,delete=False,encoding="utf-8",newline="") as handle:
        temp=Path(handle.name); frame.to_csv(handle,index=False); handle.flush(); os.fsync(handle.fileno())
    os.replace(temp,path)

def validate_results(frame: pd.DataFrame, manifest: pd.DataFrame) -> None:
    if list(frame.columns)!=COLUMNS: raise ValueError("invalid candidate result schema")
    if frame.candidate_id.duplicated().any(): raise ValueError("duplicate candidate_id")
    lookup=manifest.set_index("candidate_id")
    for _,row in frame.iterrows():
        if row.status not in STATUSES or row.candidate_id not in lookup.index: raise ValueError("invalid status/candidate")
        expected=lookup.loc[row.candidate_id]
        if any(row[key]!=expected[key] for key in ["outer_fold","k","n_estimators","learning_rate","max_depth"]): raise ValueError("manifest mismatch")
        if row.status=="COMPLETE":
            scores=np.array([row[f"inner_fold_{i}_macro_f1"] for i in range(1,4)],dtype=float)
            if not np.isfinite(scores).all() or not ((scores>=0)&(scores<=1)).all(): raise ValueError("invalid F1")
            if not np.isclose(row.mean_inner_macro_f1,scores.mean()) or not np.isclose(row.std_inner_macro_f1,scores.std()): raise ValueError("invalid summary")

def load_or_initialize(path: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    if path.exists(): result=pd.read_csv(path)
    else: result=pd.DataFrame(columns=COLUMNS); atomic_csv(result,path)
    validate_results(result,manifest); return result

def eligible(manifest: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    done=set(results.loc[results.status=="COMPLETE","candidate_id"])
    return manifest.loc[~manifest.candidate_id.isin(done)].copy()
