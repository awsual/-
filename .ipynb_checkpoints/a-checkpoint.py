# run_as_ipy.py
import sys
from IPython.core.interactiveshell import InteractiveShell

def run_file_as_cells(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    cells = text.split('\n# %%') if '# %%' in text else text.split('\n\n')
    shell = InteractiveShell.instance()
    user_ns = {}
    for i, cell in enumerate(cells, 1):
        src = cell.strip()
        if not src:
            continue
        print(f'\n=== Cell {i} ===')
        result = shell.run_cell(src, store_history=True, silent=False)
        # If last expression produced a result, IPython already prints it; continue.

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python run_as_ipy.py target.py')
        sys.exit(1)
    run_file_as_cells(sys.argv[1])