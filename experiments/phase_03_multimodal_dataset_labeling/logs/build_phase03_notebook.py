from pathlib import Path
import nbformat as nbf

root = Path.cwd().resolve()
phase = root / "experiments" / "phase_03_multimodal_dataset_labeling"
notebook_path = phase / "Phase_03_Dataset_and_Folds.ipynb"

workflow = r"""from __future__ import annotations
from pathlib import Path
import hashlib, json, logging
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

def find_root(start):
    for item in [start, *start.parents]:
        if (item / "CODEX_NOTEBOOK_RULES.md").exists() and (item / "experiments").exists():
            return item
    raise RuntimeError("Validated project root not found.")

ROOT = find_root(Path.cwd().resolve())
PHASE = ROOT / "experiments" / "phase_03_multimodal_dataset_labeling"
P2 = ROOT / "experiments" / "phase_02_full_multimodal_feature_extraction" / "results"
DATA, MANIFESTS, AUDITS, FIGURES, REPORTS, LOGS = [PHASE / x for x in ["data", "manifests", "audits", "figures", "reports", "logs"]]
for folder in [DATA, MANIFESTS, AUDITS, FIGURES, REPORTS, LOGS]:
    folder.mkdir(parents=True, exist_ok=True)
if "vrdataset" in [part.lower() for part in PHASE.parts]:
    raise RuntimeError("Forbidden output path inside raw data.")
logging.basicConfig(filename=LOGS / "phase_03_execution.log", level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
summary_path = P2 / "phase02_corrected_validation_summary.json"
source_manifest_path = P2 / "phase02_modeling_feature_manifest.json"
groups_path = P2 / "phase02_corrected_feature_groups.json"
provenance_path = P2 / "phase02_corrected_feature_provenance.csv"
all_nan_path = P2 / "phase02_all_nan_feature_audit.csv"
required = [summary_path, source_manifest_path, groups_path, provenance_path, all_nan_path]
if not all(path.exists() for path in required):
    raise FileNotFoundError([str(path) for path in required if not path.exists()])
p2_summary = json.loads(summary_path.read_text(encoding="utf-8"))
p2_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
groups = json.loads(groups_path.read_text(encoding="utf-8"))
SOURCE = Path(p2_summary["corrected_table"])
if not SOURCE.exists() or SOURCE.name == "full_multimodal_run_level_features.csv":
    raise RuntimeError("Corrected source missing or uncorrected source selected.")
source = pd.read_csv(SOURCE)
provenance = pd.read_csv(provenance_path)
all_nan = pd.read_csv(all_nan_path)
id_cols = p2_manifest["identifier_columns"]
facts = {"rows":len(source), "subjects":int(source.subject_id.nunique()), "unique_run_keys":int(source.run_key.nunique()), "canonical_features":len(source.columns)-len(id_cols), "body_movement":len(p2_manifest["body_movement_features"]), "performance":len(p2_manifest["performance_features"]), "control_input":len(p2_manifest["control_input_features"]), "structurally_unusable_all_nan":len(p2_manifest["structurally_unusable_features"])}
expected = {"rows":487, "subjects":35, "unique_run_keys":487, "canonical_features":1247, "body_movement":42, "performance":59, "control_input":0, "structurally_unusable_all_nan":12}
if facts != expected or source.run_key.duplicated().any() or source[id_cols].isna().any().any():
    raise RuntimeError({"corrected_phase02_integrity":facts, "expected":expected})
source["difficulty_level_raw"] = source.difficulty_level.astype(str)
source["difficulty_level"] = source.difficulty_level_raw.str.extract(r"level-(\d{2,3})", expand=False).astype(int)
if source.difficulty_level.isna().any():
    raise RuntimeError("A raw difficulty label could not be parsed.")
before = source.difficulty_level.value_counts().sort_index().to_dict()
if before != {0:68, 1:104, 2:106, 3:104, 4:105}:
    raise RuntimeError({"difficulty_distribution":before})
cohort = source.loc[source.difficulty_level.isin([1,2,3,4])].copy()
cohort["target_class"] = cohort.difficulty_level.astype(int) - 1
cohort["target_score"] = cohort.difficulty_level.astype(float)
mapping = cohort[["difficulty_level","target_class"]].drop_duplicates().set_index("difficulty_level").target_class.to_dict()
if len(cohort) != 419 or cohort.run_key.nunique() != 419 or mapping != {1:0,2:1,3:2,4:3}:
    raise RuntimeError("Invalid modeling cohort or targets.")
canonical = [x for x in source.columns if x not in id_cols and x != "difficulty_level_raw"]
performance = p2_manifest["performance_features"]
structural = p2_manifest["structurally_unusable_features"]
unverified = p2_manifest["unverified_features"]
if set(structural) != set(all_nan.feature_name):
    raise RuntimeError("All-NaN audit mismatch.")
feature_group = {feature: group for group, features in groups.items() if group != "identifier_columns" for feature in features}
excluded = set(structural) | set(unverified)
primary = [x for x in p2_manifest["primary_without_performance_features"] if x not in excluded]
with_performance = [x for x in canonical if x not in excluded]
performance_only = [x for x in performance if x not in excluded]
if (len(primary),len(with_performance),len(performance_only)) != (1176,1235,59) or set(primary) & set(performance):
    raise RuntimeError("Eligible feature manifest mismatch.")
missingness = pd.DataFrame([{"feature_name":x, "feature_group":feature_group.get(x,"unmapped"), "missing_count":int(cohort[x].isna().sum()), "missing_ratio":float(cohort[x].isna().mean()), "structurally_unusable":x in structural} for x in canonical])
status_by_feature = provenance.set_index("feature_name").provenance_status.to_dict()
leakage_rows = []
for x in canonical:
    decision = "EXCLUDE_ALL_DATASETS_STRUCTURALLY_UNUSABLE_ALL_NAN" if x in structural else "EXCLUDE_PRIMARY_UNVERIFIED" if x in unverified or status_by_feature.get(x) == "UNVERIFIED" else "EXCLUDE_PRIMARY_INCLUDE_AUXILIARY_AND_PERFORMANCE_ONLY" if x in performance else "INCLUDE_PRIMARY"
    leakage_rows.append({"column_name":x, "column_kind":"feature", "feature_group":feature_group.get(x,"unmapped"), "provenance_status":status_by_feature.get(x,"missing"), "decision":decision})
for x in [*id_cols, "target_class", "target_score"]:
    leakage_rows.append({"column_name":x, "column_kind":"identifier_or_target", "feature_group":"not_applicable", "provenance_status":"not_applicable", "decision":"EXCLUDE_ALL_MODEL_INPUTS"})
leakage = pd.DataFrame(leakage_rows)
forbidden = set(id_cols) | {"target_class","target_score"}
if forbidden & (set(primary) | set(with_performance) | set(performance_only)):
    raise RuntimeError("Identifier/target leakage.")
gkf = GroupKFold(n_splits=5)
assignments = np.full(len(cohort), -1, dtype=int)
fold_rows = []
for fold, (train_idx, test_idx) in enumerate(gkf.split(cohort, cohort.target_class, groups=cohort.subject_id), 1):
    train_subjects, test_subjects = set(cohort.subject_id.iloc[train_idx]), set(cohort.subject_id.iloc[test_idx])
    if train_subjects & test_subjects: raise AssertionError(fold)
    assignments[test_idx] = fold
    fold_rows.append({"outer_fold":fold, "train_samples":len(train_idx), "test_samples":len(test_idx), "train_subjects":len(train_subjects), "test_subjects":len(test_subjects), "subject_overlap_count":0, "test_class_distribution":json.dumps(cohort.target_class.iloc[test_idx].value_counts().sort_index().to_dict())})
cohort["outer_fold"] = assignments
fold_summary = pd.DataFrame(fold_rows)
inner = pd.DataFrame([{"outer_fold":f, "outer_train_subjects":int(cohort.loc[cohort.outer_fold.ne(f),"subject_id"].nunique()), "inner_groupkfold_3_feasible":int(cohort.loc[cohort.outer_fold.ne(f),"subject_id"].nunique()) >= 3} for f in range(1,6)])
if (assignments < 0).any() or fold_summary.subject_overlap_count.ne(0).any() or not inner.inner_groupkfold_3_feasible.all():
    raise RuntimeError("Fold validation failed.")
fold_cols = ["run_key","subject_id","session_id","run_id","difficulty_level_raw","difficulty_level","target_class","target_score","outer_fold"]
folds = cohort[fold_cols].sort_values("run_key").reset_index(drop=True)
candidate = folds.to_csv(index=False, lineterminator="\n")
sha = hashlib.sha256(candidate.encode()).hexdigest()
fold_path = DATA / "fold_assignments.csv"
if fold_path.exists():
    if hashlib.sha256(fold_path.read_bytes()).hexdigest() != sha: raise RuntimeError("Frozen fold file differs; refusing overwrite.")
    fold_action = "REUSED_EXISTING_FROZEN_FILE"
else:
    fold_path.write_text(candidate, encoding="utf-8", newline="")
    fold_action = "CREATED_FROZEN_FILE"
fold_manifest = {"phase":"03_multimodal_dataset_labeling","splitter":"GroupKFold","n_splits":5,"group_column":"subject_id","fold_assignment_path":str(fold_path.relative_to(ROOT)),"fold_assignment_sha256":sha,"row_count":len(folds),"subject_count":int(folds.subject_id.nunique()),"write_action":fold_action,"regeneration_policy":"Phases 04-10 must load this exact file and must not regenerate outer folds."}
metadata = ["subject_id","session_id","run_id","difficulty_level_raw","difficulty_level","run_key","target_class","target_score","outer_fold"]
datasets = {"primary_without_performance":primary, "auxiliary_with_performance":with_performance, "performance_only":performance_only}
dataset_paths = {}
for name, features in datasets.items():
    path = DATA / (name + ".csv")
    cohort[metadata + features].to_csv(path,index=False)
    dataset_paths[name] = str(path.relative_to(ROOT))
def manifest(name, features, role):
    return {"phase":"03_multimodal_dataset_labeling","dataset_name":name,"intended_role":role,"source_corrected_phase02_table":str(SOURCE.relative_to(ROOT)),"feature_count":len(features),"features":features,"feature_groups":{g:[x for x in features if feature_group.get(x)==g] for g in sorted(set(feature_group.values()))},"excluded_identifier_columns":id_cols,"excluded_targets":["target_class","target_score"],"excluded_structurally_unusable_features":structural,"excluded_unverified_features":unverified,"imputation_performed":False,"scaling_performed":False}
for filename,name,features,role in [("primary_feature_manifest.json","primary_without_performance",primary,"PRIMARY_THESIS_DATASET"),("with_performance_feature_manifest.json","auxiliary_with_performance",with_performance,"AUXILIARY_SHORTCUT_LEARNING_COMPARISON"),("performance_only_feature_manifest.json","performance_only",performance_only,"AUXILIARY_PERFORMANCE_ONLY_SHORTCUT_ANALYSIS")]:
    (MANIFESTS / filename).write_text(json.dumps(manifest(name,features,role),indent=2),encoding="utf-8")
(MANIFESTS / "feature_group_manifest.json").write_text(json.dumps({"source_corrected_feature_groups":groups,"canonical_feature_count":len(canonical),"eligible_with_performance_feature_count":len(with_performance),"primary_feature_count":len(primary),"performance_only_feature_count":len(performance_only)},indent=2),encoding="utf-8")
(MANIFESTS / "fold_manifest.json").write_text(json.dumps(fold_manifest,indent=2),encoding="utf-8")
audit = source[id_cols].copy()
audit["cohort_decision"] = np.where(audit.difficulty_level.isin([1,2,3,4]),"INCLUDE_TASK_COHORT","EXCLUDE_REST_LEVEL_000")
audit["exclusion_reason"] = np.where(audit.difficulty_level.eq(0),"verified_rest_level_000","")
audit.to_csv(AUDITS / "cohort_filter_audit.csv",index=False)
leakage.to_csv(AUDITS / "leakage_audit.csv",index=False)
missingness.to_csv(AUDITS / "missingness_summary.csv",index=False)
leakage.loc[leakage.decision.ne("INCLUDE_PRIMARY")].to_csv(AUDITS / "feature_exclusion_audit.csv",index=False)
fold_summary.merge(inner,on="outer_fold").to_csv(AUDITS / "fold_summary.csv",index=False)
dataset_manifest = {"phase":"03_multimodal_dataset_labeling","corrected_phase02_input":str(SOURCE.relative_to(ROOT)),"corrected_phase02_sha256":hashlib.sha256(SOURCE.read_bytes()).hexdigest(),"modeling_rows":len(cohort),"excluded_rest_rows":68,"subjects":int(cohort.subject_id.nunique()),"unique_run_keys":int(cohort.run_key.nunique()),"classification_target":{"name":"target_class","mapping":{str(k):v for k,v in mapping.items()},"interpretation":"task-difficulty-induced workload proxy class"},"regression_target":{"name":"target_score","values":[1.0,2.0,3.0,4.0],"interpretation":"bounded task-difficulty-induced workload proxy score"},"datasets":dataset_paths,"no_global_imputation":True,"no_global_scaling":True,"model_training_performed":False}
(MANIFESTS / "dataset_manifest.json").write_text(json.dumps(dataset_manifest,indent=2),encoding="utf-8")
summary = {"status":"VERIFIED","phase03_ready":True,"modeling_rows":len(cohort),"subjects":int(cohort.subject_id.nunique()),"class_distribution":{str(k):int(v) for k,v in cohort.target_class.value_counts().sort_index().items()},"primary_features":len(primary),"with_performance_features":len(with_performance),"performance_only_features":len(performance_only),"fold_assignments_path":str(fold_path.relative_to(ROOT)),"fold_sha256":sha,"leakage_audit":"PASS","subject_overlap":"PASS","inner_groupkfold_3_feasible":True,"no_training":True,"no_global_scaling":True,"no_global_imputation":True,"warnings":["Performance-derived datasets are auxiliary shortcut-learning analyses only.","Missing values remain raw and require fold-local processing later."]}
(REPORTS / "phase03_validation_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
(REPORTS / "phase03_verification_report.md").write_text("# Phase 03 Verification Report\n\n## VERIFIED\n- Corrected Phase 02 input: " + str(SOURCE.relative_to(ROOT)) + "\n- Modeling cohort: " + str(len(cohort)) + " runs from " + str(cohort.subject_id.nunique()) + " subjects.\n- Primary, with-performance, performance-only features: " + str((len(primary),len(with_performance),len(performance_only))) + ".\n- Frozen GroupKFold SHA-256: " + sha + ".\n- Leakage audit: PASS. Subject overlap: PASS.\n\n## NOT VERIFIED\n- Model performance is intentionally not evaluated in Phase 03.\n\n## WARNINGS\n- Difficulty is a task-difficulty-induced workload proxy, not a direct psychological measure.\n- Performance variants are auxiliary shortcut-learning analyses only.\n\n## KEY RESULTS\n- Class distribution: " + str(summary["class_distribution"]) + ".\n\n## OUTPUT FILES\n- Required data, manifests, audits, report, and log are in this Phase 03 workspace.\n\n## NEXT PHASE REQUIREMENTS\n- Phase 04 must load data/fold_assignments.csv unchanged and use primary_without_performance.csv for primary results.\n",encoding="utf-8")
logging.info("Phase 03 complete %s",sha)
print({"workflow_execution":"complete","modeling_rows":len(cohort),"fold_sha256":sha})
"""

sections = [
("Phase objective", "print('Objective: construct datasets, targets, audits, and frozen folds only.')"),
("Imports", "print('Imports are included in the executable workflow cell.')"),
("Project/path validation", "print({'project_root': str(ROOT), 'phase03_dir': str(PHASE), 'raw_data_write_protection': 'PASS'})"),
("Corrected Phase 02 input discovery", "print('Corrected Phase 02 input path:\\n' + str(SOURCE))"),
("Phase 02 integrity verification", "print(pd.DataFrame([facts])); print('Corrected feature provenance:', provenance.provenance_status.value_counts().to_dict())"),
("Difficulty/rest audit", "print(source.difficulty_level.value_counts().sort_index().rename_axis('difficulty_level').to_frame('runs'))"),
("Task cohort creation", "print({'modeling_rows':len(cohort),'unique_run_keys':cohort.run_key.nunique()})"),
("Classification target creation", "print(pd.Series(mapping,name='target_class').rename_axis('difficulty_level').to_frame())"),
("Regression target creation", "print(cohort.target_score.value_counts().sort_index().rename_axis('target_score').to_frame('runs'))"),
("Feature-group verification", "print({'body_movement':len(p2_manifest['body_movement_features']),'performance':len(performance),'control_input':len(p2_manifest['control_input_features']),'all_nan_excluded':len(structural)})"),
("Primary feature manifest", "print({'primary_without_performance_features':len(primary),'performance_intersection':len(set(primary)&set(performance))})"),
("With-performance manifest", "print({'with_performance_features':len(with_performance),'performance_features_included':len(set(with_performance)&set(performance))})"),
("Performance-only manifest", "print({'performance_only_features':len(performance_only)})"),
("Missingness audit", "print(missingness.groupby('feature_group').missing_ratio.agg(['count','mean','max']).sort_index())"),
("Leakage audit", "print(leakage.decision.value_counts().to_frame('columns'))"),
("Subject-wise outer GroupKFold", "print(fold_summary)"),
("Fold statistics", "print(fold_summary[['outer_fold','train_samples','test_samples','train_subjects','test_subjects']])"),
("Subject-overlap assertions", "print('PASS: zero subjects overlap between train and test in every outer fold.')"),
("Inner-CV feasibility", "print(inner)"),
("Fold freezing and checksum", "print({'fold_file':str(fold_path),'action':fold_action,'sha256':sha})"),
("Dataset/manifests saving", "print(pd.DataFrame({'dataset':list(datasets),'rows':[len(cohort)]*3,'features':[len(x) for x in datasets.values()]}))"),
("Initial validation result", "print('VERIFIED\\n', json.dumps(summary,indent=2)); print('INITIAL DIRECTORY TREE:'); [print(path.relative_to(PHASE)) for path in sorted(PHASE.rglob('*'))]"),
("Expanded corrected-input, missingness, leakage, and fold audit", r'''from datetime import datetime, timezone
repair_log = pd.read_csv(P2 / "phase02_placeholder_repair_log.csv")
availability = pd.read_csv(P2 / "phase02_verified_run_modality_availability.csv")

# Verify every repaired unavailable run/modality pair contains no retained modality values.
repair_checks = []
for record in repair_log.itertuples(index=False):
    columns = provenance.loc[provenance["availability_modality"].eq(record.modality), "feature_name"].tolist()
    run_values = source.loc[source.run_key.eq(record.run_key), columns]
    non_null_after = int(run_values.notna().sum(axis=1).iloc[0])
    repair_checks.append({"run_key":record.run_key, "modality":record.modality, "phase02_repair_action":record.repair_action, "feature_columns_checked":len(columns), "non_null_values_after_repair":non_null_after, "pass":non_null_after == 0})
repair_check_df = pd.DataFrame(repair_checks)
if repair_check_df.empty or not repair_check_df["pass"].all():
    raise RuntimeError("Repaired unavailable modality pairs retain values.")

body_provenance = provenance.loc[provenance.provenance_status.eq("VERIFIED_BODY_MOVEMENT")]
if len(body_provenance) != 42 or set(body_provenance.feature_group) != {"body_movement"} or "unknown_features" in set(groups):
    raise RuntimeError("Body-movement provenance/group validation failed.")
if int(p2_summary["control_input_feature_count"]) != 0:
    raise RuntimeError("Explicit control-input feature count is not zero.")
if not bool(p2_summary["critical_assertions"]["unavailable_modalities_have_no_placeholder_values"]):
    raise RuntimeError("Phase 02 did not validate placeholder repair.")
if np.isinf(source[canonical].select_dtypes(include=[np.number]).to_numpy(dtype=float, na_value=np.nan)).any():
    raise RuntimeError("Infinite numeric feature values found in corrected input.")

availability_task = availability.loc[availability.run_key.isin(cohort.run_key)].copy()
modality_rows = []
for modality, availability_column in [("eye_tracking", "has_eye_tracking"), ("ecg", "has_ecg"), ("eda", "has_eda"), ("emg", "has_emg"), ("respiration", "has_respiration"), ("head_movement", "has_head_movement"), ("xplane", "has_xplane"), ("body_movement", "has_torso_body_accelerometer"), ("performance", "has_performance")]:
    features = [f for f in canonical if feature_group.get(f) == modality or provenance.set_index("feature_name").loc[f, "modality"] == modality]
    feature_missing_rate = float(cohort[features].isna().mean().mean()) if features else float("nan")
    modality_rows.append({"modality":modality, "feature_count":len(features), "feature_cell_missing_rate":feature_missing_rate, "runs_available":int(availability_task[availability_column].fillna(False).sum()), "runs_unavailable":int((~availability_task[availability_column].fillna(False)).sum())})
modality_missingness = pd.DataFrame(modality_rows)
row_missingness = pd.DataFrame({"run_key":cohort.run_key, "missing_feature_count":cohort[with_performance].isna().sum(axis=1), "missing_feature_ratio":cohort[with_performance].isna().mean(axis=1)})
missingness_diagnostic = {"total_nan_rate_with_performance":float(cohort[with_performance].isna().mean().mean()), "rows_with_any_missing_feature":int(row_missingness.missing_feature_count.gt(0).sum()), "global_single_value_feature_count_phase02":int(p2_summary["single_value_feature_count"]), "imputation_performed":False}

direct_target_duplicates = []
for feature in with_performance:
    values = cohort[feature]
    if values.notna().all() and np.array_equal(values.to_numpy(), cohort.target_class.to_numpy()): direct_target_duplicates.append((feature, "target_class"))
    if values.notna().all() and np.array_equal(values.to_numpy(dtype=float), cohort.target_score.to_numpy(dtype=float)): direct_target_duplicates.append((feature, "target_score"))
if direct_target_duplicates:
    raise RuntimeError({"direct_target_duplicate_features":direct_target_duplicates})
expanded_leakage = leakage.copy()
expanded_leakage["feature_name"] = expanded_leakage["column_name"]
expanded_leakage["reason_flagged"] = np.where(expanded_leakage.decision.eq("INCLUDE_PRIMARY"), "NONE", expanded_leakage.decision)
expanded_leakage["evidence"] = np.where(expanded_leakage.column_kind.eq("identifier_or_target"), "identifier_or_target_column", np.where(expanded_leakage.column_name.isin(performance), "verified_performance_feature_auxiliary_only", np.where(expanded_leakage.column_name.isin(structural), "verified_structurally_unusable_all_nan", "verified_provenance_predictor")))
expanded_leakage["included_in_primary"] = expanded_leakage.column_name.isin(primary)
expanded_leakage["decision"] = np.where(expanded_leakage.column_kind.eq("identifier_or_target"), "EXCLUDE_IDENTIFIER_OR_LABEL", np.where(expanded_leakage.column_name.isin(performance), "AUXILIARY_SHORTCUT_ANALYSIS_ONLY", np.where(expanded_leakage.column_name.isin(structural), "EXCLUDE_DIRECT_TARGET_LEAKAGE", "KEEP_VERIFIED_PREDICTOR")))
if expanded_leakage.loc[expanded_leakage.included_in_primary, "decision"].ne("KEEP_VERIFIED_PREDICTOR").any():
    raise RuntimeError("Primary leakage audit decision mismatch.")

fold_rows_expanded = []
inner_rows_expanded = []
for fold in range(1, 6):
    test_frame, train_frame = cohort.loc[cohort.outer_fold.eq(fold)], cohort.loc[cohort.outer_fold.ne(fold)]
    test_subjects, train_subjects = sorted(test_frame.subject_id.unique()), sorted(train_frame.subject_id.unique())
    row = {"outer_fold":fold, "train_subject_count":len(train_subjects), "test_subject_count":len(test_subjects), "train_run_count":len(train_frame), "test_run_count":len(test_frame), "test_subjects":";".join(test_subjects), "subject_overlap_count":len(set(train_subjects) & set(test_subjects))}
    for target in range(4):
        row[f"train_class_{target}_count"] = int(train_frame.target_class.eq(target).sum())
        row[f"test_class_{target}_count"] = int(test_frame.target_class.eq(target).sum())
    fold_rows_expanded.append(row)
    inner_gkf = GroupKFold(n_splits=3)
    for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_gkf.split(train_frame, train_frame.target_class, groups=train_frame.subject_id), 1):
        inner_train_subjects = set(train_frame.subject_id.iloc[inner_train_idx])
        inner_val_subjects = set(train_frame.subject_id.iloc[inner_val_idx])
        inner_rows_expanded.append({"outer_fold":fold, "inner_fold":inner_fold, "inner_train_subject_count":len(inner_train_subjects), "inner_validation_subject_count":len(inner_val_subjects), "inner_subject_overlap_count":len(inner_train_subjects & inner_val_subjects), "feasible":len(inner_train_subjects & inner_val_subjects) == 0})
fold_summary_expanded = pd.DataFrame(fold_rows_expanded)
inner_expanded = pd.DataFrame(inner_rows_expanded)
if fold_summary_expanded.subject_overlap_count.ne(0).any() or not inner_expanded.feasible.all() or cohort.groupby("subject_id").outer_fold.nunique().ne(1).any():
    raise RuntimeError("Outer/inner subject grouping validation failed.")

cohort_filter = source[["run_key", "difficulty_level_raw"]].copy()
cohort_filter["included_in_modeling"] = cohort_filter.run_key.isin(cohort.run_key)
cohort_filter["exclusion_reason"] = np.where(cohort_filter.included_in_modeling, "", "REST_LEVEL_000_NOT_PART_OF_CLASSIFICATION_REGRESSION_TASK")
if cohort_filter.included_in_modeling.sum() != 419 or cohort_filter.loc[~cohort_filter.included_in_modeling, "difficulty_level_raw"].ne("level-000").any() or cohort[["subject_id", "session_id", "run_id", "difficulty_level_raw"]].duplicated().any() or cohort.subject_id.isna().any():
    raise RuntimeError("Cohort filter audit validation failed.")

cohort_filter.to_csv(AUDITS / "cohort_filter_audit.csv", index=False)
cohort_filter.to_csv(AUDITS / "phase03_cohort_filter_audit.csv", index=False)
expanded_leakage[["feature_name","feature_group","reason_flagged","evidence","decision","included_in_primary"]].to_csv(AUDITS / "leakage_audit.csv", index=False)
expanded_leakage[["feature_name","feature_group","reason_flagged","evidence","decision","included_in_primary"]].to_csv(AUDITS / "phase03_leakage_audit.csv", index=False)
missingness.to_csv(AUDITS / "missingness_summary.csv", index=False)
missingness.to_csv(AUDITS / "phase03_missingness_summary.csv", index=False)
modality_missingness.to_csv(AUDITS / "modality_missingness_summary.csv", index=False)
row_missingness.to_csv(AUDITS / "row_missingness_summary.csv", index=False)
repair_check_df.to_csv(AUDITS / "repaired_modality_placeholder_audit.csv", index=False)
fold_summary_expanded.to_csv(AUDITS / "fold_summary.csv", index=False)
inner_expanded.to_csv(AUDITS / "inner_cv_feasibility.csv", index=False)
pd.DataFrame([{"canonical_corrected_features":len(canonical), "structurally_unusable":len(structural), "eligible_with_performance":len(with_performance), "performance":len(performance), "primary_without_performance":len(primary), "body_movement":len(body_provenance), "control_input":0}]).to_csv(AUDITS / "phase03_feature_group_summary.csv", index=False)

timestamp = datetime.now(timezone.utc).isoformat()
for file_name in ["primary_feature_manifest.json", "with_performance_feature_manifest.json", "performance_only_feature_manifest.json"]:
    manifest_path = MANIFESTS / file_name
    current = json.loads(manifest_path.read_text(encoding="utf-8"))
    current.update({"dataset_version":"phase03_corrected_phase02_v1", "modeling_row_count":419, "subject_count":35, "target_definitions":{"target_class":"difficulty_level - 1", "target_score":"float(difficulty_level)"}, "performance_inclusion_status":"EXCLUDED_FROM_PRIMARY" if file_name == "primary_feature_manifest.json" else "AUXILIARY_SHORTCUT_ANALYSIS_ONLY", "body_movement_status":"VERIFIED_BODY_MOVEMENT:42", "control_input_status":"EXPLICIT_CONTROL_INPUT_UNAVAILABLE:0", "creation_timestamp_utc":timestamp})
    manifest_path.write_text(json.dumps(current, indent=2), encoding="utf-8")
fold_manifest_current = json.loads((MANIFESTS / "fold_manifest.json").read_text(encoding="utf-8"))
fold_manifest_current.update({"creation_timestamp_utc":datetime.fromtimestamp(fold_path.stat().st_mtime, timezone.utc).isoformat(), "dataset_version":"phase03_corrected_phase02_v1", "classification_target":"target_class", "regression_target":"target_score", "immutable_status":"FROZEN", "instruction":"DO NOT REGENERATE OUTER FOLDS IN PHASE 04-10"})
(MANIFESTS / "fold_manifest.json").write_text(json.dumps(fold_manifest_current, indent=2), encoding="utf-8")

summary.update({"corrected_input_timestamp_utc":datetime.fromtimestamp(SOURCE.stat().st_mtime, timezone.utc).isoformat(), "corrected_input_columns":len(source.columns)-1, "phase02_repair_pairs_verified":len(repair_check_df), "repair_placeholder_audit":"PASS", "body_movement_status":"VERIFIED_BODY_MOVEMENT:42", "control_input_status":"EXPLICIT_CONTROL_INPUT_UNAVAILABLE:0", "missingness":missingness_diagnostic, "direct_target_duplicate_features":[], "outer_fold_test_subject_counts":fold_summary_expanded.set_index("outer_fold").test_subject_count.to_dict(), "outer_fold_test_run_counts":fold_summary_expanded.set_index("outer_fold").test_run_count.to_dict(), "inner_cv_detailed_feasibility":"PASS", "raw_data_status":"READ_ONLY_NOT_MODIFIED", "corrected_phase02_source_status":"READ_ONLY_NOT_MODIFIED_BY_PHASE03", "critical_assertions_passed":True})
(REPORTS / "phase03_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(MANIFESTS / "dataset_manifest.json").write_text(json.dumps({**dataset_manifest, "dataset_version":"phase03_corrected_phase02_v1", "creation_timestamp_utc":timestamp, "body_movement_status":"VERIFIED_BODY_MOVEMENT:42", "control_input_status":"EXPLICIT_CONTROL_INPUT_UNAVAILABLE:0", "structurally_unusable_feature_count":len(structural), "fold_assignment_path":str(fold_path.relative_to(ROOT)), "fold_assignment_sha256":sha}, indent=2), encoding="utf-8")
(REPORTS / "phase03_verification_report.md").write_text("# Phase 03 Verification Report\n\n## VERIFIED\n- Corrected Phase 02 input: " + str(SOURCE.relative_to(ROOT)) + "\n- Source modification timestamp (UTC): " + summary["corrected_input_timestamp_utc"] + "\n- 487 source runs; 419 verified Difficulty 1-4 modeling runs; 35 subjects.\n- 38 repaired unavailable run/modality pairs checked with zero retained values.\n- Primary / with-performance / performance-only features: " + str((len(primary), len(with_performance), len(performance_only))) + ".\n- Frozen 5-fold subject-wise GroupKFold SHA-256: " + sha + ".\n\n## MISSINGNESS\n- Raw missingness is retained: " + str(missingness_diagnostic) + ".\n\n## LEAKAGE AUDIT\n- PASS. Primary contains no identifiers, labels, performance features, structurally unusable features, or direct target duplicates.\n\n## OUTER AND INNER FOLDS\n- Outer subject overlap: PASS. Inner GroupKFold(n_splits=3) feasibility: PASS for all outer training partitions.\n\n## SCOPE\n- No model training, global imputation, global scaling, or raw-data modification occurred.\n\n## NEXT PHASE REQUIREMENTS\n- Later phases must load data/fold_assignments.csv unchanged and verify its checksum before training.\n", encoding="utf-8")
print({"source_modified_utc":summary["corrected_input_timestamp_utc"], "repair_pairs_checked":len(repair_check_df), "total_nan_rate":missingness_diagnostic["total_nan_rate_with_performance"], "rows_with_missing_features":missingness_diagnostic["rows_with_any_missing_feature"], "direct_target_duplicates":direct_target_duplicates})
print(fold_summary_expanded)
print(inner_expanded)
'''),
("Phase Validation Summary", r'''print("## VERIFIED DATASET")
print({"modeling_rows":len(cohort), "subjects":int(cohort.subject_id.nunique()), "class_distribution":summary["class_distribution"], "target_class":"difficulty_level - 1", "target_score":"bounded task-difficulty-induced workload proxy"})
print(cohort[["difficulty_level_raw", "difficulty_level", "target_class"]].drop_duplicates().sort_values("difficulty_level").to_string(index=False))
print("## FEATURE MANIFESTS")
print({"primary":len(primary), "with_performance":len(with_performance), "performance_only":len(performance_only), "body_movement":len(body_provenance), "structurally_unusable_exclusions":len(structural), "control_input":"UNAVAILABLE (0)"})
print("## MISSINGNESS")
print(missingness_diagnostic)
print("## LEAKAGE AUDIT")
print("PASS; no unresolved direct target duplicates or identifier/label columns in the primary manifest.")
print("## OUTER FOLDS")
print({"method":"GroupKFold", "n_splits":5, "test_subjects_per_fold":summary["outer_fold_test_subject_counts"], "test_samples_per_fold":summary["outer_fold_test_run_counts"], "subject_overlap":"PASS"})
print("## INNER CV FEASIBILITY")
print("PASS for all five outer training sets using GroupKFold(n_splits=3).")
print("## FOLD FREEZE")
print({"path":str(fold_path), "sha256":sha, "status":"FROZEN - DO NOT REGENERATE OUTER FOLDS IN PHASE 04-10"})
print("## OUTPUT FILES")
for path in sorted(PHASE.rglob("*")):
    if path.is_file(): print(path.relative_to(PHASE))
print("## PHASE 04 READINESS")
print("PHASE 03 READY: YES")
'''),
]

cells = [
    nbf.v4.new_markdown_cell("# Phase 03: Dataset and Frozen Subject-Wise Folds\n\nDifficulty levels are task-difficulty-induced workload proxy labels. This notebook performs no model training, imputation, or scaling."),
    nbf.v4.new_code_cell(workflow),
]
for heading, output_code in sections:
    cells += [nbf.v4.new_markdown_cell("## " + heading), nbf.v4.new_code_cell(output_code)]

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3"}})
nbf.write(nb, notebook_path)
print(notebook_path)
