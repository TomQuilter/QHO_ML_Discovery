import json
with open('ImaginaryTime_QHO_Anharmonic_warm_start.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
for i, cell in enumerate(nb['cells']):
    ct = cell['cell_type']
    src = ''.join(cell['source'][:2]).strip().replace('\n', ' ')[:120]
    print(f'Cell {i:2d} [{ct:8s}]: {src.encode("ascii", "replace").decode()}')
