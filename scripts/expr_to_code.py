import re

def build_term_string(terms: list, matrix_name: str, coord_to_index) -> str:
    """
    Build a string representation of terms from a matrix expression.

    Args:
        terms: List of tuples (sign, coefficient, variable), e.g., [('+', '2', 'b31'), ('-', '', 'b32')]
        matrix_name: Name of the matrix ('A' or 'B')
        coord_to_index: Function to convert coordinates to flat index

    Returns:
        String representation of the terms (e.g., "A[0] - 2*A[1] + ...")
    """
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

def parse_terms(segment: str, prefix: str):
    """
    Extracts terms like -2a11 or +a12 or 3b23 and returns a list of (sign, coef, var)
    """
    pattern = re.compile(rf'([+-]?)\s*(\d*)\s*({prefix}[1-5][1-5])')
    return [(m[0] if m[0] else '+', m[1] if m[1] else '1', m[2]) for m in pattern.findall(segment)]

def translate_expression(expr: str, param: str) -> str:
    """
    Translate a single Mathematica-style expression into CUDA code.
    """
    # Remove extra whitespace for consistent parsing
    expr_norm = re.sub(r'\s+', ' ', expr.strip())

    # Check for outer negation (either -((...)) or -(...) )
    negate_result = False
    outer_match = re.fullmatch(r'-\((.*)\)', expr_norm)
    if outer_match:
        negate_result = True
        expr_norm = outer_match.group(1).strip()

    # Step 2: Now split by matching the three components explicitly
    # Match: (a...) (b...) (c...) with optional whitespace
    inner_groups = re.findall(r'\(([^()]*)\)', expr_norm)

    if len(inner_groups) == 3:
        groups = inner_groups
    else:
        # Fallback: try to split on first two ')' characters
        parts = []
        count = 0
        i = 0
        while i < len(expr_norm) and count < 2:
            if expr_norm[i] == ')':
                count += 1
            i += 1
        if count < 2:
            raise ValueError("Could not split the expression into three parts")
        before = expr_norm[:i]
        after = expr_norm[i:]
        groups = re.findall(r'\(([^()]*)\)', before) + [after.strip()]
        if len(groups) != 3:
            print(f"Groups found: {groups}")
            raise ValueError("Expression must have exactly three parenthesized groups")


    termsA = parse_terms(groups[0], 'a')
    termsB = parse_terms(groups[1], 'b')
    termsC = parse_terms(groups[2], 'c')

    # Coordinate to flat index
    def coord_to_index(coord: str) -> int:
        row = int(coord[1]) - 1
        col = int(coord[2]) - 1
        return row * 5 + col

    # Build CUDA lines
    code_lines = []

    termA_str = build_term_string(termsA, 'A', coord_to_index)
    termB_str = build_term_string(termsB, 'B', coord_to_index)

    neg_char = '-' if negate_result else ''


    code_lines.append(f"float termA{param} = {termA_str};")
    code_lines.append(f"float termB{param} = {termB_str};")
    code_lines.append(f"float result{param} = {neg_char}termA{param} * termB{param};")

    for sign, coef, var in termsC:
        idx = coord_to_index(var)
        op = "+=" if sign == '+' else "-="
        code_lines.append(f"C[{idx}] {op} result{param}; // {var}")

    return "\n".join(code_lines)


# ── Example Usage ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    expr = "(-a41 + a44 + a45 + a51 - a54 - a55) (-b11 - b12 + b13 - 2 b31 - 2 b33) (c15 + c22 + c24 + c35 + c55)"
    param = "1"
    print(translate_expression(expr, param))
