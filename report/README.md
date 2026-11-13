# Final report

`final_report.pdf` is the definitive project report.

The `source/` directory contains the active LaTeX sections, bibliography,
preamble, metadata, and only the figures referenced by the final document.
Intermediate report versions, provenance notes used during drafting, and LaTeX
build products have been removed.

To build the report on Windows, run
`powershell -ExecutionPolicy Bypass -File report/source/build_unified_report.ps1`
from the repository root after ensuring that `pdflatex` and `bibtex` are
available.
