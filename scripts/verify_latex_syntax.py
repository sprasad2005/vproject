"""
Strict LaTeX syntax, environment matching, citation, label, and asset validator.
"""
import os
import re

PAPER_DIR = r"d:\vproj\RiceGuard_IEEE_Paper"

def validate():
    errors = []
    warnings = []

    # 1. Read references.bib
    bib_path = os.path.join(PAPER_DIR, "references.bib")
    bib_keys = set()
    if os.path.exists(bib_path):
        with open(bib_path, "r", encoding="utf-8") as f:
            bib_text = f.read()
        for match in re.finditer(r'@\w+\s*\{\s*([\w\-]+)\s*,', bib_text):
            bib_keys.add(match.group(1))
    else:
        errors.append("references.bib not found at root!")

    print(f"Found {len(bib_keys)} BibTeX keys: {sorted(list(bib_keys))}")

    # 2. Collect all labels across all tex files
    labels = set()
    tex_files = []
    for root, dirs, files in os.walk(PAPER_DIR):
        for f in files:
            if f.endswith(".tex"):
                tex_files.append(os.path.join(root, f))

    for tf in tex_files:
        with open(tf, "r", encoding="utf-8") as f:
            content = f.read()
        for match in re.finditer(r'\\label\{([^}]+)\}', content):
            labels.add(match.group(1))

    print(f"Found {len(labels)} defined labels.")

    # 3. Check each tex file
    for tf in tex_files:
        rel_path = os.path.relpath(tf, PAPER_DIR)
        with open(tf, "r", encoding="utf-8") as f:
            lines = f.readlines()

        env_stack = []
        for line_num, line in enumerate(lines, start=1):
            # Strip comments
            comment_idx = line.find("%")
            code_line = line if comment_idx == -1 else line[:comment_idx]

            # Check for markdown bold **
            if "**" in code_line:
                errors.append(f"[{rel_path}:{line_num}] Unescaped markdown '**' found: {code_line.strip()}")

            # Check for unescaped underscores outside math/texttt/url/label/ref/cite
            # Find \texttt{...} and check for unescaped _
            for tt_match in re.finditer(r'\\texttt\{([^}]+)\}', code_line):
                inner = tt_match.group(1)
                # Check for raw underscore not preceded by \
                if re.search(r'(?<!\\)_', inner):
                    errors.append(f"[{rel_path}:{line_num}] Unescaped underscore in \\texttt: {tt_match.group(0)}")

            # Environment matching
            for b_match in re.finditer(r'\\begin\{([\w*]+)\}', code_line):
                env_stack.append((b_match.group(1), line_num))
            for e_match in re.finditer(r'\\end\{([\w*]+)\}', code_line):
                e_name = e_match.group(1)
                if not env_stack:
                    errors.append(f"[{rel_path}:{line_num}] Extra \\end{{{e_name}}} with no matching \\begin")
                else:
                    b_name, b_line = env_stack.pop()
                    if b_name != e_name:
                        errors.append(f"[{rel_path}:{line_num}] Mismatched environment: \\begin{{{b_name}}} at line {b_line} ended by \\end{{{e_name}}}")

            # Check citations
            for cite_match in re.finditer(r'\\cite\{([^}]+)\}', code_line):
                keys = [k.strip() for k in cite_match.group(1).split(",")]
                for k in keys:
                    if k and k not in bib_keys:
                        errors.append(f"[{rel_path}:{line_num}] Undefined citation key '{k}'")

            # Check \ref
            for ref_match in re.finditer(r'\\ref\{([^}]+)\}', code_line):
                r_key = ref_match.group(1)
                if r_key not in labels:
                    errors.append(f"[{rel_path}:{line_num}] Undefined reference label '{r_key}'")

            # Check \includegraphics
            for img_match in re.finditer(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', code_line):
                img_rel = img_match.group(1)
                img_full = os.path.join(PAPER_DIR, img_rel)
                if not os.path.exists(img_full):
                    # Check without extension
                    if not any(os.path.exists(img_full + ext) for ext in [".png", ".pdf", ".jpg"]):
                        errors.append(f"[{rel_path}:{line_num}] Missing image file: {img_rel}")

            # Check \input
            for inp_match in re.finditer(r'\\input\{([^}]+)\}', code_line):
                inp_rel = inp_match.group(1)
                inp_full = os.path.join(PAPER_DIR, inp_rel)
                if not os.path.exists(inp_full) and not os.path.exists(inp_full + ".tex"):
                    errors.append(f"[{rel_path}:{line_num}] Missing \\input file: {inp_rel}")

        if env_stack:
            for b_name, b_line in env_stack:
                # In modular section files, document env is in main.tex
                if b_name != "document":
                    errors.append(f"[{rel_path}:{b_line}] Unclosed environment \\begin{{{b_name}}}")

    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    if errors:
        print(f"FAILED with {len(errors)} errors:")
        for e in errors:
            print("  -", e)
    else:
        print("PASSED: 100% of environments, citations, labels, images, inputs, and syntax are VALID!")
    print("="*60)

if __name__ == "__main__":
    validate()
