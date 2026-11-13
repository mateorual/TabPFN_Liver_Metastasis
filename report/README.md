# Final report

`final_report.pdf` is the definitive project report.

## Supporting data collection and staging reference

[`PanNET_Patient_Data_Collection_Tables.pdf`](PanNET_Patient_Data_Collection_Tables.pdf)
documents the SEER variables used to construct the pancreatic neuroendocrine
tumour cohorts, their permitted source values, and the clinical recodings used
in the analyses. It also explains an important limitation of the historical
staging information available to the project: the SEER tumour and nodal stage
fields follow an American Joint Committee on Cancer (AJCC) seventh-edition
framework, whereas current AJCC Version 9 and European Neuroendocrine Tumor
Society (ENETS) conventions use updated pancreatic neuroendocrine tumour
definitions.

These frameworks are not directly interchangeable. In particular, primary
tumour size boundaries and local-invasion criteria can assign different T
categories, while the historical SEER grade I--IV field is not equivalent to
the current World Health Organization neuroendocrine tumour grade based on
Ki-67 and mitotic activity. The supporting document should therefore be used
when interpreting the staging and grade variables in the report. It is a
technical data dictionary and contains no patient-level records.

The `source/` directory contains the active LaTeX sections, bibliography,
preamble, metadata, and only the figures referenced by the final document.
Intermediate report versions, provenance notes used during drafting, and LaTeX
build products have been removed.

To build the report on Windows, run
`powershell -ExecutionPolicy Bypass -File report/source/build_unified_report.ps1`
from the repository root after ensuring that `pdflatex` and `bibtex` are
available.
