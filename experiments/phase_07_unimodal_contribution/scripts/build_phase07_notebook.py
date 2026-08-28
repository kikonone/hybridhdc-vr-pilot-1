from __future__ import annotations

from pathlib import Path

import nbformat as nbf


PHASE_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PHASE_DIR / "Phase_07_Unimodal_Contribution.ipynb"


def markdown(title: str, body: str = ""):
    return nbf.v4.new_markdown_cell(f"## {title}\n\n{body}".strip())


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


def build() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["metadata"]["language_info"] = {"name": "python", "version": "3"}
    notebook["metadata"]["phase07_scope"] = "INITIALIZATION_ONLY_NO_MODELING"
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            "# Phase 07 — Unimodal Contribution Analysis\n\n"
            "Executed initialization and read-only upstream audit. This notebook performs no model training, "
            "hypervector generation, prediction, OOF generation, row deletion, or fitted preprocessing."
        ),
        markdown("1. Environment information"),
        code(
            "from pathlib import Path\n"
            "import json, platform, sys\n"
            "PHASE_DIR = Path.cwd().resolve()\n"
            "assert PHASE_DIR.name == 'phase_07_unimodal_contribution'\n"
            "sys.path.insert(0, str(PHASE_DIR / 'scripts'))\n"
            "from initialize_phase07 import collect_audit\n"
            "audit = collect_audit()\n"
            "print(json.dumps({'python': sys.version, 'platform': platform.platform(), 'phase_dir': str(PHASE_DIR)}, indent=2))"
        ),
        markdown("2. Required upstream file existence"),
        code("print(json.dumps(audit['upstream_existence'], indent=2)); assert all(audit['upstream_existence'].values())"),
        markdown("3. Primary data SHA-256"),
        code("print(audit['hashes']['primary_data']); assert audit['data_checks']['primary_data_checksum_pass']"),
        markdown("4. Frozen fold SHA-256"),
        code("print(audit['hashes']['frozen_folds']); assert audit['data_checks']['frozen_fold_checksum_pass']"),
        markdown("5. Phase 06 core configuration SHA-256"),
        code(
            "keys = ['phase06_freeze', 'phase06_classification', 'phase06_regression']\n"
            "print(json.dumps({key: audit['hashes'][key] for key in keys}, indent=2))\n"
            "assert audit['phase06_checks']['freeze_checksum_pass']\n"
            "assert audit['phase06_checks']['classification_checksum_pass']\n"
            "assert audit['phase06_checks']['regression_checksum_pass']"
        ),
        markdown("6. Modeling rows, subjects, and Primary features"),
        code(
            "checks = audit['data_checks']\n"
            "print(json.dumps({key: checks[key] for key in ['modeling_rows','subjects','primary_predictive_features','unique_run_key']}, indent=2))\n"
            "assert checks['modeling_rows'] == 419 and checks['subjects'] == 35\n"
            "assert checks['primary_predictive_features'] == 1176 and checks['unique_run_key'] == 419"
        ),
        markdown("7. Classification and regression targets"),
        code(
            "print(json.dumps({key: checks[key] for key in ['target_class_values','target_class_missing','target_score_values','target_score_missing']}, indent=2))\n"
            "assert checks['target_class_values'] == [0,1,2,3] and checks['target_class_missing'] == 0\n"
            "assert checks['target_score_values'] == [1.0,2.0,3.0,4.0] and checks['target_score_missing'] == 0"
        ),
        markdown("8. run_key uniqueness and fold alignment"),
        code(
            "print(json.dumps({key: checks[key] for key in ['fold_assignment_rows','fold_assignment_unique_run_key','run_key_one_to_one_alignment']}, indent=2))\n"
            "assert checks['fold_assignment_rows'] == checks['fold_assignment_unique_run_key'] == 419\n"
            "assert checks['run_key_one_to_one_alignment']"
        ),
        markdown("9. Outer-fold subject isolation"),
        code("print(json.dumps(audit['outer_fold_audits'], indent=2)); assert checks['outer_subject_isolation_pass']"),
        markdown("10. Inner GroupKFold feasibility"),
        code("print(json.dumps(audit['inner_groupkfold_audits'], indent=2)); assert checks['inner_groupkfold_3_feasibility_pass']"),
        markdown("11. Frozen five-modality membership and counts"),
        code("print(json.dumps(audit['modality_checks']['feature_counts'], indent=2)); assert audit['modality_checks']['expected_counts_pass']"),
        markdown("12. Modality disjointness and complete Primary coverage"),
        code(
            "mapping = audit['modality_checks']\n"
            "print(json.dumps({key: mapping[key] for key in ['feature_union_count','disjointness_pass','union_coverage_pass','duplicate_memberships']}, indent=2))\n"
            "assert mapping['feature_union_count'] == 1176 and mapping['disjointness_pass'] and mapping['union_coverage_pass']"
        ),
        markdown("13. Modality-level missingness and availability audit"),
        code(
            "summary = [{\n"
            "  'modality': item['modality'], 'feature_count': item['feature_count'],\n"
            "  'fully_missing_rows': item['fully_missing_rows'],\n"
            "  'subjects_with_available_data': item['subjects_with_available_data'],\n"
            "  'target_class_distribution_available_rows': item['target_class_distribution_available_rows'],\n"
            "  'target_score_distribution_available_rows': item['target_score_distribution_available_rows'],\n"
            "  'outer_fold_details': item['outer_fold_details']} for item in audit['modality_audits']]\n"
            "print(json.dumps(summary, indent=2))\n"
            "assert all(all(f['train_has_available_modality_data'] and f['test_has_available_modality_data'] for f in item['outer_fold_details']) for item in audit['modality_audits'])"
        ),
        markdown("14. Performance, control-input, and unverified exclusions"),
        code(
            "exclusions = {key: mapping[key] for key in ['performance_primary_intersection_count','control_input_feature_count','unverified_feature_count','body_movement_status','body_movement_verified']}\n"
            "print(json.dumps(exclusions, indent=2))\n"
            "assert mapping['performance_primary_intersection_count'] == 0\n"
            "assert mapping['control_input_feature_count'] == mapping['unverified_feature_count'] == 0\n"
            "assert mapping['body_movement_verified']"
        ),
        markdown("15. Frozen Phase 06 classification interface"),
        code(
            "classification = audit['phase06_classification']\n"
            "print(json.dumps({key: classification[key] for key in ['selected_variant','selected_variant_name','selected_fixed_dimension','levels','feature_k','structure_selection_policy','selection_evidence','single_seed_selected']}, indent=2))\n"
            "assert audit['phase06_checks']['classification_interface_pass']"
        ),
        markdown("16. Frozen Phase 06 regression interface"),
        code(
            "regression = audit['phase06_regression']\n"
            "print(json.dumps({key: regression[key] for key in ['selected_variant','selected_regression_head','selected_fixed_dimension','levels','feature_k','parameter_policy','selection_evidence','single_seed_selected']}, indent=2))\n"
            "print('Target: bounded difficulty-induced workload proxy regression')\n"
            "assert audit['phase06_checks']['regression_interface_pass']\n"
            "assert not audit['phase06_checks']['outer_oof_read_for_selection']"
        ),
        markdown("17. Initialization artifact existence and JSON parseability"),
        code(
            "required = [\n"
            " 'configs/phase07_experiment_contract.json','configs/phase07_environment.json','configs/phase07_upstream_interface.json',\n"
            " 'manifests/phase07_input_manifest.json','manifests/phase07_modality_feature_manifest.json',\n"
            " 'audits/phase07_input_and_fold_audit.json','audits/phase07_modality_mapping_audit.json',\n"
            " 'audits/phase07_phase06_freeze_interface_audit.json','audits/phase07_initialization_artifact_audit.json',\n"
            " 'audits/phase07_notebook_persistence_audit.json']\n"
            "artifact_checks = {}\n"
            "for relative in required:\n"
            "    path = PHASE_DIR / relative\n"
            "    artifact_checks[relative] = {'exists': path.is_file(), 'json_parseable': False}\n"
            "    if path.is_file():\n"
            "        json.loads(path.read_text(encoding='utf-8'))\n"
            "        artifact_checks[relative]['json_parseable'] = True\n"
            "print(json.dumps(artifact_checks, indent=2))\n"
            "assert all(item['exists'] and item['json_parseable'] for item in artifact_checks.values())"
        ),
        markdown("18. Final initialization summary"),
        code(
            "summary = {\n"
            " 'initialization_gates_without_notebook_persistence_pass': audit['initialization_gates_without_notebook_persistence_pass'],\n"
            " 'phase_status': 'PENDING_CONTRACT_FREEZE',\n"
            " 'hdc_training_executed': audit['hdc_training_executed'],\n"
            " 'ready_for_contract_freeze_subject_to_notebook_persistence': audit['initialization_gates_without_notebook_persistence_pass'],\n"
            " 'ready_for_modeling': False}\n"
            "print(json.dumps(summary, indent=2))\n"
            "assert summary['initialization_gates_without_notebook_persistence_pass']\n"
            "assert not summary['hdc_training_executed'] and not summary['ready_for_modeling']"
        ),
        markdown(
            "Phase Validation Summary",
            "**VERIFIED**: frozen inputs, checksums, row/subject/feature/target counts, run-key alignment, outer isolation, inner feasibility, modality mapping, exclusions, and Phase 06 interfaces.\n\n"
            "**NOT VERIFIED**: the intentionally unresolved Phase 07 Contract Freeze decisions.\n\n"
            "**WARNINGS**: control input is unavailable; performance features are excluded; no modeling is authorized.\n\n"
            "**KEY RESULTS**: five disjoint modalities cover all 1176 Primary features.\n\n"
            "**OUTPUT FILES**: Phase 07 configs, manifests, audits, documentation, and this executed notebook.\n\n"
            "**NEXT PHASE REQUIREMENTS**: Phase 07 Contract Freeze. Do not start modeling."
        ),
    ]
    nbf.write(notebook, NOTEBOOK_PATH)
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    build()
