"""Execute and save the notebook with the project virtual environment kernel."""
from pathlib import Path
import json
import os
import sys
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

root = Path(__file__).resolve().parent
runtime = root / 'artifacts' / 'runtime'
kernel = runtime / 'jupyter' / 'kernels' / 'nbfc'
kernel.mkdir(parents=True, exist_ok=True)
(kernel / 'kernel.json').write_text(json.dumps({'argv': [sys.executable, '-m', 'ipykernel_launcher',
    '-f', '{connection_file}'], 'display_name': 'NBFC local', 'language': 'python'}))
os.environ['JUPYTER_PATH'] = str(runtime / 'jupyter')
os.environ['JUPYTER_RUNTIME_DIR'] = str(runtime)
os.environ['IPYTHONDIR'] = str(runtime / 'ipython')
os.environ['MPLCONFIGDIR'] = str(runtime / 'matplotlib')
os.environ['OMP_NUM_THREADS'] = '4'
path = root / '01_NBFC_Loan_Default_Model.ipynb'
nb = nbformat.read(path, as_version=4)
def progress(cell, cell_index, **kwargs):
    first = cell.source.splitlines()[0][:100]
    print(f'Cell {cell_index + 1}/{len(nb.cells)}: {first}', flush=True)
client = NotebookClient(nb, timeout=1800, kernel_name='nbfc',
    resources={'metadata': {'path': str(root)}}, on_cell_start=progress)
try:
    client.execute()
finally:
    nbformat.write(nb, path)
nbformat.validate(nb)
html, _ = HTMLExporter(template_name='lab').from_notebook_node(nb)
(root / 'artifacts' / 'notebook_preview.html').write_text(html, encoding='utf-8')
print('SUCCESS: executed notebook and HTML preview saved.', flush=True)
import finalize_notebook
