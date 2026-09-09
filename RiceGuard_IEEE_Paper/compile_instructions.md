# Compilation Instructions

### Option 1: Overleaf (Recommended)
Upload `RiceGuard_IEEE_Paper.zip` directly to Overleaf. Compile with `pdfLaTeX`.

### Option 2: Local LaTeX Compilation (MiKTeX / TeX Live)
Run the following standard sequence in PowerShell or Terminal:
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
