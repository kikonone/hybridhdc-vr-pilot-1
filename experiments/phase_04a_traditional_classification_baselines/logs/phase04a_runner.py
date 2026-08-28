from pathlib import Path
import hashlib, importlib.util, json, time, warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, make_scorer, recall_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

SEED = 42
EXPECTED_SHA256 = "e4dc943af21851bace49345f6336f9c88f82613ca3a26b3c433efe7dfb041f6f"

class SafeSelectKBest(BaseEstimator, TransformerMixin):
    def __init__(self, k="all"): self.k = k
    def fit(self, X, y):
        self.effective_k_ = "all" if self.k == "all" else min(int(self.k), X.shape[1])
        self.selector_ = SelectKBest(f_classif, k=self.effective_k_).fit(X, y)
        return self
    def transform(self, X): return self.selector_.transform(X)
    def get_support(self): return self.selector_.get_support()

def _root():
    root = Path.cwd().resolve()
    for candidate in [root, *root.parents]:
        if (candidate / "CODEX_NOTEBOOK_RULES.md").exists() and (candidate / "vrdataset").is_dir():
            return candidate
    raise RuntimeError("PROJECT ROOT NOT VERIFIED")

def setup():
    root = _root()
    phase03 = root / "experiments" / "phase_03_multimodal_dataset_labeling"
    out = root / "experiments" / "phase_04a_traditional_classification_baselines"
    for d in ["results/oof", "results/fold_metrics", "results/summaries", "results/predictions", "configs", "figures", "audits", "reports", "logs"]: (out / d).mkdir(parents=True, exist_ok=True)
    paths = {"data": phase03 / "data/primary_without_performance.csv", "folds": phase03 / "data/fold_assignments.csv", "primary_manifest": phase03 / "manifests/primary_feature_manifest.json", "fold_manifest": phase03 / "manifests/fold_manifest.json", "feature_groups": phase03 / "manifests/feature_group_manifest.json"}
    if not all(p.is_file() for p in paths.values()): raise FileNotFoundError("Required Phase 03 artifact absent")
    sha = hashlib.sha256(paths["folds"].read_bytes()).hexdigest()
    if sha != EXPECTED_SHA256: raise RuntimeError("FROZEN FOLD CHECKSUM MISMATCH")
    data, folds = pd.read_csv(paths["data"]), pd.read_csv(paths["folds"])
    features = json.loads(paths["primary_manifest"].read_text(encoding="utf-8"))["features"]
    if len(data) != 419 or data.subject_id.nunique() != 35 or len(features) != 1176 or any(f not in data for f in features): raise RuntimeError("PRIMARY DATASET VALIDATION FAILED")
    if data.target_class.value_counts().sort_index().to_dict() != {0:104,1:106,2:104,3:105}: raise RuntimeError("TARGET DISTRIBUTION INVALID")
    merged = data.merge(folds, on="run_key", how="outer", suffixes=("_data", "_fold"), indicator=True, validate="one_to_one")
    if len(merged) != 419 or not (merged._merge == "both").all() or not (merged.subject_id_data == merged.subject_id_fold).all() or not (merged.target_class_data == merged.target_class_fold).all(): raise RuntimeError("DATA/FOLD ALIGNMENT FAILED")
    if "outer_fold" in data.columns:
        existing = data[["run_key", "outer_fold"]].merge(folds[["run_key", "outer_fold"]], on="run_key", suffixes=("_data", "_fold"), validate="one_to_one")
        if not (existing.outer_fold_data == existing.outer_fold_fold).all(): raise RuntimeError("DATA/FOLD OUTER FOLD MISMATCH")
    data = data.drop(columns=["outer_fold"], errors="ignore").merge(folds[["run_key", "outer_fold"]], on="run_key", validate="one_to_one")
    return {"root":root,"out":out,"paths":paths,"sha":sha,"data":data,"features":features,"folds":sorted(data.outer_fold.unique())}

def pipeline(estimator, scaled):
    steps=[("imputer",SimpleImputer(strategy="median",add_indicator=True,keep_empty_features=True)),("variance",VarianceThreshold())]
    if scaled: steps.append(("scaler",StandardScaler()))
    return Pipeline(steps+[("selector",SafeSelectKBest()),("classifier",estimator)])

def metric(y, p):
    r=recall_score(y,p,labels=[0,1,2,3],average=None,zero_division=0)
    return {"macro_f1":f1_score(y,p,average="macro",zero_division=0),"balanced_accuracy":balanced_accuracy_score(y,p),"accuracy":accuracy_score(y,p),"weighted_f1":f1_score(y,p,average="weighted",zero_division=0),**{f"recall_class_{i}":r[i] for i in range(4)}}

def models():
    return {
      "dummy_most_frequent":("DummyClassifier",DummyClassifier(strategy="most_frequent"),False,{"selector__k":["all"]},None),
      "dummy_stratified":("DummyClassifier",DummyClassifier(strategy="stratified",random_state=SEED),False,{"selector__k":["all"]},SEED),
      "logistic_regression":("Logistic Regression",LogisticRegression(solver="lbfgs",max_iter=5000,random_state=SEED),True,{"classifier__C":[.1,1.0,10.0],"selector__k":[100,200]},SEED),
      "linear_svm":("Linear SVM",LinearSVC(max_iter=10000,random_state=SEED),True,{"classifier__C":[.1,1.0,10.0],"selector__k":[100,200]},SEED),
      "rbf_svm":("RBF SVM",SVC(kernel="rbf", probability=False),True,{"classifier__C":[.1,1.0,10.0],"classifier__gamma":["scale","auto"],"selector__k":[100,200]},None),
      "random_forest":("Random Forest",RandomForestClassifier(random_state=SEED,n_jobs=1),False,{"classifier__n_estimators":[200,500],"classifier__max_depth":[None,20],"classifier__max_features":["sqrt"],"selector__k":[100,200]},SEED),
      "knn":("K-Nearest Neighbors",KNeighborsClassifier(metric="minkowski"),True,{"classifier__n_neighbors":[3,5,7,11],"classifier__weights":["uniform","distance"],"classifier__p":[1,2],"selector__k":[100,200]},None),
      "gradient_boosting":("Gradient Boosting",GradientBoostingClassifier(random_state=SEED),False,{"classifier__n_estimators":[100,200],"classifier__learning_rate":[.05,.1],"classifier__max_depth":[2],"selector__k":[100,200]},SEED)}

def smoke(ctx):
    d, fs, fold = ctx["data"],ctx["features"],ctx["folds"][0]
    tr, te=d[d.outer_fold!=fold],d[d.outer_fold==fold]
    inner=GroupKFold(3)
    for a,b in inner.split(tr[fs],tr.target_class,tr.subject_id): assert not(set(tr.iloc[a].subject_id)&set(tr.iloc[b].subject_id))
    g=GridSearchCV(pipeline(LogisticRegression(max_iter=5000,random_state=SEED),True),{"classifier__C":[.1],"selector__k":[50]},scoring=make_scorer(f1_score,average="macro",zero_division=0),cv=inner,n_jobs=1,error_score="raise")
    g.fit(tr[fs],tr.target_class,groups=tr.subject_id); pred=g.predict(te[fs])
    if not set(pred).issubset({0,1,2,3}): raise RuntimeError("SMOKE PREDICTIONS INVALID")
    result={"REAL_DATA_SMOKE_TEST":"PASS","status":"PASS","model":"logistic_regression","outer_fold":int(fold),"best_params":g.best_params_,"metrics_not_final":metric(te.target_class,pred),"preprocessing":"training-fold pipeline: median imputation with indicators, variance filtering, scaling, SelectKBest"}
    (ctx["out"] / "audits/smoke_test.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result

def run(ctx, only=None, folds_only=None):
    d,fs,out=ctx["data"],ctx["features"],ctx["out"]
    pred_frames=[]; rows=[]; selected={}; failures=[]; scorer=make_scorer(f1_score,average="macro",zero_division=0)
    progress={name:"NOT STARTED" for name in models()}
    progress_path=out/"logs/phase04a_progress.json"
    for name,(family,est,scaled,grid,seed) in models().items():
        if only is not None and name not in only: continue
        progress[name]="RUNNING"; progress_path.write_text(json.dumps(progress,indent=2),encoding="utf-8")
        selected[name]={}
        for fold in ctx["folds"]:
            if folds_only is not None and fold not in folds_only: continue
            tr,te=d[d.outer_fold!=fold].reset_index(drop=True),d[d.outer_fold==fold].reset_index(drop=True)
            if set(tr.subject_id)&set(te.subject_id): raise RuntimeError("OUTER SUBJECT LEAKAGE")
            inner=GroupKFold(3)
            if name in {"logistic_regression", "linear_svm", "rbf_svm", "random_forest", "knn", "gradient_boosting"}:
                checkpoint_dir=out/f"results/checkpoints/{name}"; checkpoint_dir.mkdir(parents=True,exist_ok=True)
                (checkpoint_dir/f"fold_{fold}_started.json").write_text(json.dumps({"frozen_fold_sha256":EXPECTED_SHA256,"dataset_version":"PRIMARY_WITHOUT_PERFORMANCE","model":"Logistic Regression","outer_fold":int(fold),"train_rows":len(tr),"test_rows":len(te),"train_subjects":tr.subject_id.nunique(),"test_subjects":te.subject_id.nunique(),"search_grid":grid,"decision":"COMPUTE_CONSTRAINED_COMPACT_GRID","preprocessing":"SimpleImputer median+indicator -> VarianceThreshold -> StandardScaler -> SelectKBest(f_classif) -> LogisticRegression","n_jobs":1},indent=2),encoding="utf-8")
            for a,b in inner.split(tr[fs],tr.target_class,tr.subject_id): assert not(set(tr.iloc[a].subject_id)&set(tr.iloc[b].subject_id))
            search=GridSearchCV(pipeline(est,scaled),grid,scoring=scorer,cv=inner,n_jobs=1,error_score=np.nan)
            t=time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always"); search.fit(tr[fs],tr.target_class,groups=tr.subject_id)
            train_s=time.perf_counter()-t
            for w in caught:
                if "ConvergenceWarning" in w.category.__name__ or "FitFailedWarning" in w.category.__name__: failures.append({"model":name,"fold":fold,"parameters":"GridSearchCV","error_warning":str(w.message),"action":"recorded; predefined neutral max_iter/grid retained"})
            t=time.perf_counter(); p=search.best_estimator_.predict(te[fs]); infer_s=time.perf_counter()-t
            sel=int(search.best_estimator_.named_steps["selector"].get_support().sum()); params={k:(v.item() if isinstance(v,np.generic) else v) for k,v in search.best_params_.items()}|{"effective_selected_k":sel}; selected[name][str(fold)]=params
            rows.append({"model":name,"model_family":family,"outer_fold":fold,"test_run_count":len(te),"test_subject_count":te.subject_id.nunique(),"selected_hyperparameters":json.dumps(params,sort_keys=True),"selected_feature_count":sel,"tuning_training_seconds":train_s,"inference_seconds":infer_s,**metric(te.target_class,p)})
            frame=te[["subject_id","session_id","run_id","run_key","outer_fold","target_class"]].rename(columns={"target_class":"true_class"}).copy(); frame["predicted_class"]=p; frame["model"]=name; frame["model_family"]=family; frame["selected_k"]=sel; frame["seed"]=seed
            pr=np.full((len(te),4),np.nan); ds=np.full((len(te),4),np.nan); cl=search.best_estimator_.named_steps["classifier"]
            if hasattr(search.best_estimator_,"predict_proba"):
                v=search.best_estimator_.predict_proba(te[fs]); [pr.__setitem__((slice(None),int(c)),v[:,i]) for i,c in enumerate(cl.classes_)]
            elif hasattr(search.best_estimator_,"decision_function"):
                v=np.asarray(search.best_estimator_.decision_function(te[fs])); [ds.__setitem__((slice(None),int(c)),v[:,i]) for i,c in enumerate(cl.classes_)]
            for i in range(4): frame[f"probability_class_{i}"]=pr[:,i]; frame[f"decision_score_class_{i}"]=ds[:,i]
            pred_frames.append(frame)
            if name in {"logistic_regression", "linear_svm", "rbf_svm", "random_forest", "knn", "gradient_boosting"}:
                checkpoint_dir=out/f"results/checkpoints/{name}"; checkpoint_dir.mkdir(parents=True,exist_ok=True)
                frame.to_csv(checkpoint_dir/f"{name}_fold_{fold}_predictions.csv",index=False)
                (checkpoint_dir/f"{name}_fold_{fold}_metrics.json").write_text(json.dumps(rows[-1],default=float,indent=2),encoding="utf-8")
                (checkpoint_dir/f"{name}_fold_{fold}_best_params.json").write_text(json.dumps(params,indent=2),encoding="utf-8")
        progress[name]="COMPLETE"; progress_path.write_text(json.dumps(progress,indent=2),encoding="utf-8")
        checkpoint_oof=pd.concat(pred_frames,ignore_index=True); checkpoint_metrics=pd.DataFrame(rows)
        checkpoint_oof.to_csv(out/"results/oof/classification_oof_predictions.csv",index=False)
        checkpoint_metrics.to_csv(out/"results/fold_metrics/classification_fold_results.csv",index=False)
        for checkpoint_name,checkpoint_frame in checkpoint_oof.groupby("model"): checkpoint_frame.to_csv(out/f"results/predictions/{checkpoint_name}_oof.csv",index=False)
        (out/"configs/classification_best_params_by_fold.json").write_text(json.dumps(selected,indent=2),encoding="utf-8")
    oof=pd.concat(pred_frames,ignore_index=True); fold_metrics=pd.DataFrame(rows)
    oof.to_csv(out/"results/oof/classification_oof_predictions.csv",index=False); fold_metrics.to_csv(out/"results/fold_metrics/classification_fold_results.csv",index=False)
    for name,frame in oof.groupby("model"): frame.to_csv(out/f"results/predictions/{name}_oof.csv",index=False)
    (out/"configs/classification_best_params_by_fold.json").write_text(json.dumps(selected,indent=2),encoding="utf-8")
    pd.DataFrame(failures,columns=["model","fold","parameters","error_warning","action"]).to_csv(out/"audits/failed_configurations.csv",index=False)
    return oof,fold_metrics,failures
