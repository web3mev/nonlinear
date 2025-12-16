import os

try:
    with open('parameters.csv', 'r') as f:
        lines = f.readlines()

    header = lines[0].strip()
    # Replace header parts carefully to avoid partial matches if any (but here keys are distinct)
    header = header.replace('X1_Var', 'X1_Var_NM')
    header = header.replace('X2_Var', 'X2_Var_NM')
    # RiskFactor is substring of RiskFactor_NM, so replace carefully
    # Split by comma
    cols = header.split(',')
    mapping = {
        'X1_Var': 'X1_Var_NM',
        'X2_Var': 'X2_Var_NM',
        'RiskFactor': 'RiskFactor_VAL',
        'X1_Val': 'X1_Var_Val',
        'X2_Val': 'X2_Var_Val',
        'On_Off': 'On_Off_Flag'
    }
    new_cols = [mapping.get(c, c) for c in cols]
    if 'Notes' not in new_cols:
        new_cols.append('Notes')
        
    new_header = ",".join(new_cols) + "\n"
    
    new_lines = [new_header]
    for line in lines[1:]:
        # Append empty note if line doesn't have it
        # Just append comma
        new_lines.append(line.strip() + ',\n')

    with open('parameters.csv', 'w') as f:
        f.writelines(new_lines)
    print("DONE")
except Exception as e:
    print(e)
