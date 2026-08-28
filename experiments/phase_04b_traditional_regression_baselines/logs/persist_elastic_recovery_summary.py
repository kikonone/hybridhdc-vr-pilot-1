from pathlib import Path
import json
import nbformat
from nbclient import NotebookClient

p = Path(r"E:\hdc-vr-pilot\experiments\phase_04b_traditional_regression_baselines")
n = p / "Phase_04B_Regression_Baselines.ipynb"
nb = nbformat.read(n, as_version=4)
cell = nbformat.v4.new_code_cell(
    "from pathlib import Path\n"
    "import pandas as pd\n"
    "phase = Path(r'E:\\hdc-vr-pilot\\experiments\\phase_04b_traditional_regression_baselines')\n"
    "summary = pd.read_csv(phase / 'results' / 'summaries' / 'elastic_net_summary.csv')\n"
    "assert len(summary) == 1 and summary.loc[0, 'status'] == 'COMPLETE'\n"
    "print('ELASTIC NET RECOVERY V1: COMPLETE')\n"
    "print(phase / 'results' / 'predictions' / 'elastic_net_oof.csv')\n"
    "print(summary.to_string(index=False))\n"
)
cell.metadata['tags'] = ['elastic_net_recovery_v1_official']
nb.cells.append(nbformat.v4.new_markdown_cell('## Elastic Net Recovery V1 — Syntax Fix and Foldwise Resume'))
nb.cells.append(cell)
nbformat.write(nb, n)
client = NotebookClient(nb, timeout=180, kernel_name='python3')
with client.setup_kernel():
    client.execute_cell(nb.cells[-1], len(nb.cells) - 1, store_history=True)
nbformat.write(nb, n)
nb = nbformat.read(n, as_version=4)
last = nb.cells[-1]
audit = {
    'file_exists': n.is_file(), 'parseable': True,
    'recovery_cell_execution_count': last.execution_count,
    'recovery_cell_has_outputs': bool(last.outputs),
    'recovery_tag_present': 'elastic_net_recovery_v1_official' in last.metadata.get('tags', []),
    'oof_path_in_outputs': 'elastic_net_oof.csv' in ''.join(o.get('text', '') for o in last.outputs),
}
audit['pass'] = all([audit['file_exists'], audit['parseable'], audit['recovery_cell_execution_count'] is not None, audit['recovery_cell_has_outputs'], audit['recovery_tag_present'], audit['oof_path_in_outputs']])
(p / 'audits' / 'elastic_net_notebook_persistence_audit.json').write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')
config = {'recovery_version':'V1','fold_1_2_max_iter_used':20000,'fold_1_2_naturally_converged':True,'fold_3_5_max_iter_used':100000,'target':'target_score = difficulty_level','interpretation':'bounded difficulty-induced workload proxy regression','scientific_grid_unchanged':True}
(p / 'configs' / 'elastic_net_configuration.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
print(audit)
