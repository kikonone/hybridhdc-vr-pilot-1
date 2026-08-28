import sys
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPTS=Path(__file__).resolve().parents[1]/"scripts"; sys.path.insert(0,str(SCRIPTS))
from consolidate_analyze_and_freeze_phase07 import assign_shared_ranks, canonicalize, holm, rank_biserial

def test_holm_monotone_and_bounded():
    out=holm([.01,.04,.02,.5]); assert all(0<=x<=1 for x in out); assert out[0] <= out[1] <= out[3]

def test_rank_biserial_all_zero_not_estimable():
    assert rank_biserial(np.zeros(5)) is None

def test_shared_rank_tolerance():
    d=pd.DataFrame({"modality":["b","a","c"],"x":[1.,1.+5e-13,2.]})
    out=assign_shared_ranks(d,[("x",True)]).set_index("modality"); assert out.loc["a","rank"]==out.loc["b","rank"]==1; assert out.loc["c","rank"]==3

def test_canonical_rules_use_mean_scores_and_clip():
    rows=[]; regs=[]
    for seed in [42,43,44,45,46]:
        rows.append({"modality":"m","run_key":"r","subject_id":"s","outer_fold":1,"target_class":3,"modality_available":True,"seed":seed,"class_score_0":1.,"class_score_1":1.,"class_score_2":0.,"class_score_3":0.})
        regs.append({"modality":"m","run_key":"r","subject_id":"s","outer_fold":1,"target_score":4.,"modality_available":True,"seed":seed,"prediction_raw":5.})
    c,r=canonicalize(pd.DataFrame(rows),pd.DataFrame(regs)); assert c.iloc[0].predicted_class==0; assert r.iloc[0].prediction_raw==5.; assert r.iloc[0].prediction_bounded==4.
