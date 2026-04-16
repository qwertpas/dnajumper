import json

with open('modeling/tsa_opt2.ipynb', 'r') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if "opti.subject_to(y0_var < L_var)" in source:
            print(f"Found in cell index: {i}")
            print("Source snippet:")
            print(source[:200])
