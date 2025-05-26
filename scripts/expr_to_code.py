import re

def translate_expression(expr: str, param: str) -> str:
    """
    Translate a single Mathematica-style term of a 5×5×5 Winograd-like decomposition
    into CUDA-style code. 

    Args:
      expr:     A string like "(-a12 - a14 - a35) (b22 + b23 + b24 + b25 - b42 + b52) (c13 - c23 - c31 - c33 + c43)"
      param:    A short identifier (e.g. "x") to inject into variable names:
                termAx, termBx, resultx, etc.

    Returns:
      A multi-line spytring containing:
        float termA{param} = …;
        float termB{param} = …;
        float result{param} = termA{param} * termB{param};
        C[idx] += result{param}; // cXY
        C[idx] -= result{param}; // cXY
      for each c-update.
    """
    # 1) Normalize spacing so that "+" and "-" stick to their aXY / bXY / cXY tokens
    expr_norm = expr.replace(" - ", "-").replace(" + ", "+")

    # 2) Extract all A‐terms, B‐terms, C‐terms
    termsA = re.findall(r'([+-]?a[1-5][1-5])', expr_norm)
    termsB = re.findall(r'([+-]?b[1-5][1-5])', expr_norm)
    termsC = re.findall(r'([+-]?c[1-5][1-5])', expr_norm)

    # Helper: convert "aXY" → flat index = (X−1)*5 + (Y−1)
    def coord_to_index(coord: str) -> int:
        row = int(coord[1]) - 1
        col = int(coord[2]) - 1
        return row * 5 + col

    # 3) Build termA{param}
    termA_pieces = []
    for i, t in enumerate(termsA):
        sign = '+'
        raw = t
        if t[0] in '+-':
            sign = t[0]
            raw = t[1:]
        idx = coord_to_index(raw)  # flatten aXY → index
        if i == 0:
            # first term: if sign = '-', prefix "-A[idx]"; if "+", just "A[idx]"
            if sign == '-':
                termA_pieces.append(f"-A[{idx}]")
            else:
                termA_pieces.append(f"A[{idx}]")
        else:
            # subsequent terms always show " ± A[idx]"
            if sign == '-':
                termA_pieces.append(f" - A[{idx}]")
            else:
                termA_pieces.append(f" + A[{idx}]")
    termA_str = "".join(termA_pieces)

    # 4) Build termB{param}
    termB_pieces = []
    for i, t in enumerate(termsB):
        sign = '+'
        raw = t
        if t[0] in '+-':
            sign = t[0]
            raw = t[1:]
        idx = coord_to_index(raw)
        if i == 0:
            if sign == '-':
                termB_pieces.append(f"-B[{idx}]")
            else:
                termB_pieces.append(f"B[{idx}]")
        else:
            if sign == '-':
                termB_pieces.append(f" - B[{idx}]")
            else:
                termB_pieces.append(f" + B[{idx}]")
    termB_str = "".join(termB_pieces)

    # 5) Build result line
    result_line = f"float result{param} = termA{param} * termB{param};"

    # 6) Build C updates
    c_lines = []
    for t in termsC:
        sign = '+'
        raw = t
        if t[0] in '+-':
            sign = t[0]
            raw = t[1:]
        idx = coord_to_index(raw)
        op = "+=" if sign == '+' else "-="
        c_lines.append(f"C[{idx}] {op} result{param}; // {raw}")

    # 7) Combine into final code string
    code_lines = []
    code_lines.append(f"float termA{param} = {termA_str};")
    code_lines.append(f"float termB{param} = {termB_str};")
    code_lines.append(result_line)
    code_lines.extend(c_lines)

    return "\n".join(code_lines)


# ── Example Usage ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    expr = "(-a12 - a14 - a35) (b22 + b23 + b24 + b25 - b42 + b52) (c13 - c23 - c31 - c33 + c43)"
    param = "x"
    print(translate_expression(expr, param))
