import re
from functools import partial

def extract_three_groups(expr: str) -> list[str]:
    """
    Scan expr left-to-right and extract exactly three “groups,” where each group is either:
      - A bare term (no parentheses), e.g. "a31"
      - Or a parenthesized chunk "( ... )" (we return the contents without the outer parens).
    Raise ValueError if there aren't exactly three.
    """
    groups = []
    i = 0
    n = len(expr)
    while len(groups) < 3 and i < n:
        while i < n and expr[i].isspace():
            i += 1
        if i >= n:
            break
        if expr[i] == '(':  # parenthesized group
            depth = 1
            j = i + 1
            while j < n and depth > 0:
                if expr[j] == '(': depth += 1
                elif expr[j] == ')': depth -= 1
                j += 1
            if depth != 0:
                raise ValueError(f"Unmatched '(' in: {expr}")
            inside = expr[i+1 : j-1].strip()
            groups.append(inside)
            i = j
        else:
            j = i
            while j < n and not expr[j].isspace() and expr[j] != '(': j += 1
            bare = expr[i:j].strip()
            if bare != "":
                groups.append(bare)
            i = j
    if len(groups) != 3:
        raise ValueError(f"Expected exactly 3 groups, but found {len(groups)} in: {expr!r}")
    return groups

def build_term_string(terms: list[tuple[str,str,str]], matrix_name: str, coord_to_index) -> str:
    pieces = []
    for i, (sign, coef, var) in enumerate(terms):
        idx = coord_to_index(var)
        factor = f"{coef}*" if coef and coef != '1' else ''
        term = f"{factor}{matrix_name}[{idx}]"
        if i == 0:
            pieces.append(f"-{term}" if sign == '-' else term)
        else:
            pieces.append(f" {'-' if sign == '-' else '+'} {term}")
    return "".join(pieces)

def parse_terms(segment: str, prefix: str, dim_row: int, dim_col: int) -> list[tuple[str,str,str]]:
    seg = segment.strip()
    if not seg.startswith(('+', '-')):
        seg = '+' + seg
    pattern = re.compile(rf'([+-])\s*(\d*)\s*\*?\s*({prefix}[1-{dim_row}][1-{dim_col}])')
    return [(m[0], m[1] if m[1] else '1', m[2]) for m in pattern.findall(seg)]

def coord_to_index_row_major(coord: str, row_dim: int) -> int:
    row = int(coord[1]) - 1
    col = int(coord[2]) - 1
    return row * row_dim + col

def coord_to_index_col_major(coord: str, col_dim: int) -> int:
    row = int(coord[1]) - 1
    col = int(coord[2]) - 1
    return col * col_dim + row

def translate_expression_Cwrites_last(expr: str, param: str, dim_1: int, dim_2: int, dim_3: int):
    expr_norm = re.sub(r'\s+', ' ', expr.strip())
    negate_result = False
    outer_match = re.fullmatch(r'-\((.*)\)', expr_norm)
    if outer_match:
        negate_result = True
        expr_norm = outer_match.group(1).strip()
    try:
        groups = extract_three_groups(expr_norm)
    except Exception:
        return None, None
    A_part, B_part, C_part = groups
    termsA = parse_terms(A_part, 'a', dim_1, dim_2)
    termsB = parse_terms(B_part, 'b', dim_2, dim_3)
    termsC = parse_terms(C_part, 'c', dim_3, dim_1)
    termA_str = build_term_string(termsA, 'A', partial(coord_to_index_row_major, row_dim=dim_1))
    termB_str = build_term_string(termsB, 'B', partial(coord_to_index_row_major, row_dim=dim_2))
    code_lines = [
        f"// Expression: {expr}",
        f"float termA{param} = {termA_str};",
        f"float termB{param} = {termB_str};",
    ]
    result_expr = f"termA{param} * termB{param}"
    if negate_result:
        result_expr = f"-({result_expr})"
    code_lines.append(f"float result{param} = {result_expr};")
    c_writes = []
    for sign, coef, var in termsC:
        idx = coord_to_index_col_major(var, col_dim=dim_3)
        op = "+=" if sign == '+' else "-="
        mult = "" if coef == '1' else f"{coef} * "
        c_writes.append(f"C[{idx}] {op} {mult}result{param}; // {var}")
    return code_lines, c_writes

def process_expressions_file_Cwrites_last():
    input_file = '666/666m153_lifted.txt'
    output_file = 'cuda_code.cu'
    dims = (6, 6, 6)
    all_code = []
    c_accum = {}  # key: (idx, comment), value: list of (op, expr)
    with open(input_file, 'r') as fin:
        for i, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            code_lines, c_writes = translate_expression_Cwrites_last(line, str(i), *dims)
            if code_lines is None:
                continue
            all_code.extend(code_lines)
            all_code.append("")
            # Parse c_writes for accumulation
            for w in c_writes:
                # Example: C[6] -= result55; // c22
                m = re.match(r'C\[(\d+)\] (\+=|-=) (.*?); // (c\d\d)', w)
                if not m:
                    continue
                idx, op, expr, comment = m.groups()
                key = (int(idx), comment)
                if key not in c_accum:
                    c_accum[key] = []
                # Normalize op to sign
                expr = expr.strip()
                if op == '+=':
                    c_accum[key].append(f"+{expr}")
                else:
                    c_accum[key].append(f"-{expr}")
    # Output code and then merged C writes
    with open(output_file, 'w') as fout:
        for line in all_code:
            fout.write(line + "\n")
        fout.write("\n// All C writes at the end (merged):\n")
        for (idx, comment), exprs in sorted(c_accum.items()):
            expr_sum = ' '.join(exprs)
            # Remove leading '+' if present
            expr_sum = expr_sum.lstrip('+').strip()
            fout.write(f"C[{idx}] = {expr_sum}; // {comment}\n")

if __name__ == "__main__":
    process_expressions_file_Cwrites_last()
    print(f"CUDA code written to cuda_code.cu")
