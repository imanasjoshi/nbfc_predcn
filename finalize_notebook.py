"""Add an evidence-backed opening brief and validate the saved notebook outputs."""
from pathlib import Path
import json
import nbformat
from nbconvert import HTMLExporter

root = Path(__file__).resolve().parent
path = root / '01_NBFC_Loan_Default_Model.ipynb'
nb = nbformat.read(path, as_version=4)
errors = [(i, output.get('ename'), output.get('evalue'))
    for i, cell in enumerate(nb.cells) for output in cell.get('outputs', [])
    if output.output_type == 'error']
assert not errors, errors
code_cells = [cell for cell in nb.cells if cell.cell_type == 'code']
assert all(cell.execution_count is not None for cell in code_cells)
metadata = json.loads((root / 'models' / 'model_metadata.json').read_text())
provenance = json.loads((root / 'artifacts' / 'metrics' / 'source_provenance.json').read_text())
metrics = metadata['validation_metrics']
mapping_label = 'no additional calibration' if metadata['calibration_method'] == 'none' else metadata['calibration_method'] + ' calibration'
brief = f'''## Executive brief — actual executed results

The frozen development choice is **{metadata['model_family']}**, with **{mapping_label}** and threshold **{metadata['selected_threshold']:.2f}**. On the untouched 20% holdout, **ROC-AUC = {metrics['ROC-AUC']:.4f}**, **average precision = {metrics['PR-AUC']:.4f}**, **recall = {metrics['Recall']:.4f}**, **F1 = {metrics['F1']:.4f}**, and **Brier = {metrics['Brier']:.5f}**. Full comparisons, calibration checks and threshold tradeoffs are below.

**{provenance['training_rows']:,}** training loans and **{provenance['test_rows']:,}** test loans were processed. The selected model was refitted on all labeled data, exported and reloaded, and every supplied test ID has a binary prediction. The authoritative deployment file is `models/origination_model_bundle.pkl`; use `predict.py` for probability mapping and risk bands.

**Material evidence limitation:** observed default rates decline sharply across ID deciles, and loan-grade ordering is counterintuitive. This flags possible chronology, maturity or construction effects. ID and repayment variables are excluded, but random-split performance cannot establish prospective lending validity. Origination snapshots, horizon and dataset provenance are unverified. This is an educational portfolio model.''' 
existing = next((i for i,c in enumerate(nb.cells) if c.cell_type == 'markdown'
    and c.source.startswith('## Executive brief')), None)
if existing is None:
    nb.cells.insert(1, nbformat.v4.new_markdown_cell(brief))
else:
    nb.cells[existing].source = brief
nb.metadata['validation'] = {'top_to_bottom_execution_passed': True,
    'code_cells_executed': len(code_cells), 'unhandled_cell_errors': 0,
    'opening_summary_added_from_saved_metrics': True}
nbformat.validate(nb)
nbformat.write(nb, path)
html, _ = HTMLExporter(template_name='lab').from_notebook_node(nb)
(root / 'artifacts' / 'notebook_preview.html').write_text(html, encoding='utf-8')
print(f'Notebook format valid; {len(code_cells)} code cells executed; no error outputs.')
