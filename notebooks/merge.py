import os
import nbformat
from nbformat.validator import normalize

dir_path = os.path.dirname(os.path.abspath(__file__))

# Collect all notebooks except output files
nb_files = sorted([
    os.path.join(dir_path, f) 
    for f in os.listdir(dir_path) 
    if f.endswith(".ipynb") and not f.startswith("merged")
])

merged_nb = nbformat.v4.new_notebook()

for file_path in nb_files:
    nb = nbformat.read(file_path, as_version=4)
    for cell in nb.cells:
        # Ensure proper schema structure for code vs markdown cells
        if cell.cell_type == 'code':
            if 'outputs' not in cell:
                cell['outputs'] = []
            if 'execution_count' not in cell:
                cell['execution_count'] = None
        else:
            cell.pop('outputs', None)
            cell.pop('execution_count', None)
            
        merged_nb.cells.append(cell)

# Run normalization to auto-generate missing cell IDs
_, merged_nb = normalize(merged_nb)

output_path = os.path.join(dir_path, "merged_notebook.ipynb")
with open(output_path, "w", encoding="utf-8") as f:
    nbformat.write(merged_nb, f)

print(f"Successfully merged {len(nb_files)} notebooks into merged_notebook.ipynb!")