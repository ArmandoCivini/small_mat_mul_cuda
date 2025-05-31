import re

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
        # 1) Skip whitespace
        while i < n and expr[i].isspace():
            i += 1
        if i >= n:
            break

        # 2) If we see '(', grab everything up to its matching ')'
        if expr[i] == '(':
            depth = 1
            j = i + 1
            while j < n and depth > 0:
                if expr[j] == '(':
                    depth += 1
                elif expr[j] == ')':
                    depth -= 1
                j += 1
            if depth != 0:
                raise ValueError(f"Unmatched '(' in: {expr}")
            # contents between i+1 and j-1 (exclusive of both parens)
            inside = expr[i+1 : j-1].strip()
            groups.append(inside)
            i = j  # move past ')'

        # 3) Otherwise, read a “bare” token up to next space or '('
        else:
            j = i
            while j < n and not expr[j].isspace() and expr[j] != '(':
                j += 1
            bare = expr[i:j].strip()
            if bare != "":
                groups.append(bare)
            i = j

    if len(groups) != 3:
        raise ValueError(f"Expected exactly 3 groups, but found {len(groups)} in: {expr!r}")
    return groups


def build_term_string(terms: list[tuple[str,str,str]], matrix_name: str, coord_to_index) -> str:
    """
    Given a list of (sign, coef, var) for either A or B,
    produce something like "-A[2] + 3*A[7] - A[13]".
    """
    pieces = []
    for i, (sign, coef, var) in enumerate(terms):
        idx = coord_to_index(var)  # e.g. 'a12' → 1
        factor = f"{coef}*" if coef and coef != '1' else ''
        term = f"{factor}{matrix_name}[{idx}]"
        if i == 0:
            pieces.append(f"-{term}" if sign == '-' else term)
        else:
            pieces.append(f" {'-' if sign == '-' else '+'} {term}")
    return "".join(pieces)


def parse_terms(segment: str, prefix: str) -> list[tuple[str,str,str]]:
    """
    Take something like "-a13 +2a25 - a31" (no outer parentheses) and return
    a list of (sign, coef, var) tuples. E.g. [('-', '1', 'a13'), ('+', '2','a25'), ('-','1','a31')].
    Always ensures an explicit '+' or '-' before each coordinate.
    """
    seg = segment.strip()
    if not seg.startswith(('+', '-')):
        seg = '+' + seg
    pattern = re.compile(rf'([+-])\s*(\d*)\s*({prefix}[1-5][1-5])')
    return [(m[0], m[1] if m[1] else '1', m[2]) for m in pattern.findall(seg)]


def translate_expression(expr: str, param: str) -> str:
    """
    Translate a single Mathematica-style expression into CUDA-style code for 5×5×5.
    We assume there are exactly three “groups” (A-part, B-part, C-part), 
    but they may be written with or without parentheses.

    Args:
      expr:  e.g. "a31 (-b13 - b15 + b35) (-c13 - c53)"
             or "-((a13 - a33 + a53) (-b11 + b14 + b15 + b31 - b34 - b35 - b41 + b44 + b45) c13)"
             or "( -a11 + a14 + a15 + a31 - a34 - a35 - a41 + a44 + a45 ) b13 ( c13 - c33 + c53 )"
      param: A short identifier (e.g. "1" or "x") used as termA{param}, termB{param}, result{param}.

    Returns:
      A string containing CUDA-style code, e.g.:

        // Expression: a31 (-b13 - b15 + b35) (-c13 - c53)
        float termA1 = A[10];
        float termB1 = -B[2] - B[4] + B[14];
        float result1 = termA1 * termB1;
        C[2]  -= result1; // c13
        C[12] -= result1; // c53
    """
    # 1) Normalize whitespace
    expr_norm = re.sub(r'\s+', ' ', expr.strip())

    # 2) Check for outer negation "-(...)"
    negate_result = False
    outer_match = re.fullmatch(r'-\((.*)\)', expr_norm)
    if outer_match:
        negate_result = True
        expr_norm = outer_match.group(1).strip()

    # 3) Extract exactly three groups (either parenthesized or bare).
    groups = extract_three_groups(expr_norm)
    A_part, B_part, C_part = groups  # exactly three

    # 4) Parse each part into (sign, coef, var) lists
    termsA = parse_terms(A_part, 'a')
    termsB = parse_terms(B_part, 'b')
    termsC = parse_terms(C_part, 'c')

    # 5) coord_to_index: 'a13' → (1−1)*5+(3−1) = 2, etc.
    def coord_to_index(coord: str) -> int:
        row = int(coord[1]) - 1
        col = int(coord[2]) - 1
        return row * 5 + col

    # 6) Build termA and termB expressions
    termA_str = build_term_string(termsA, 'A', coord_to_index)
    termB_str = build_term_string(termsB, 'B', coord_to_index)

    # 7) Assemble the CUDA lines
    code_lines = [
        f"// Expression: {expr}",
        f"float termA{param} = {termA_str};",
        f"float termB{param} = {termB_str};",
    ]

    result_expr = f"termA{param} * termB{param}"
    if negate_result:
        result_expr = f"-({result_expr})"
    code_lines.append(f"float result{param} = {result_expr};")

    # 8) Generate C‐updates from termsC
    for sign, coef, var in termsC:
        idx = coord_to_index(var)
        op = "+=" if sign == '+' else "-="
        code_lines.append(f"C[{idx}] {op} result{param}; // {var}")

    return "\n".join(code_lines)

def process_expressions_file(input_file: str, output_file: str):
    """
    Read expressions from input file and write translated CUDA code to output file.
    
    Args:
        input_file: Path to input file containing expressions
        output_file: Path to output file for CUDA code
    """
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for i, line in enumerate(fin, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            
            cuda_code = translate_expression(line, str(i))
            fout.write(cuda_code)
            fout.write("\n\n")  # Add blank line between expressions
# ── Example Usage ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    input_file = '555/555m93_lifted.txt'  # Replace with your input file path
    output_file = 'cuda_code.cu'     # Replace with your desired output file path
    process_expressions_file(input_file, output_file)
    print(f"CUDA code written to {output_file}")

    # Test expression
    # test_expr = "(-a13 - a15 + a35) (-b13 - b53) c31"
    # test_output = translate_expression(test_expr, 'test')
    # print("\nTest Output:")
    # print(test_output)


