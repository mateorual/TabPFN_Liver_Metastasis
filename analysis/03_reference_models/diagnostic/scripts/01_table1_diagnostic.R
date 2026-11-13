# scripts/01_table1_diagnostic.R
# Table 1: baseline characteristics by liver metastasis status.

cat("=== 01_table1_diagnostic.R ===\n")

source("config/config.R")

for (pkg in c("dplyr", "tableone", "readr", "openxlsx")) {
  if (!requireNamespace(pkg, quietly = TRUE))
    stop("Install '", pkg, "' with: install.packages('", pkg, "')")
}

suppressPackageStartupMessages({
  library(dplyr); library(tableone); library(readr); library(openxlsx)
})

df <- readRDS(file.path(MODELS_DIR, "diagnostic_data_clean.rds"))
cat("  Data:", nrow(df), "rows\n")

table_vars <- intersect(c(
  "age_at_diagnosis", "sex_male", "married", "race",
  "histology_paper", "grade", "primary_site_group",
  "ajcc_t", "ajcc_n", "n_stage_positive",
  "surgery_primary", "surgery_lymph_node", "radiotherapy", "chemotherapy",
  "tumor_size_group", "tumor_size_mm",
  "mets_bone", "mets_brain", "mets_lung"
), names(df))

continuous_vars <- intersect(c("age_at_diagnosis", "tumor_size_mm"), table_vars)
cat_vars <- setdiff(table_vars, continuous_vars)

df_tbl <- df |>
  mutate(
    mets_liver_f = factor(mets_liver, levels = c(0, 1),
                          labels = c("No Liver Mets", "Liver Mets"))
  )

tbl1 <- tableone::CreateTableOne(
  vars       = table_vars,
  strata     = "mets_liver_f",
  data       = df_tbl,
  factorVars = cat_vars,
  addOverall = TRUE
)

print(tbl1, showAllLevels = TRUE, nonnormal = continuous_vars,
      quote = FALSE, noSpaces = TRUE)

tbl_matrix <- print(tbl1, showAllLevels = TRUE, nonnormal = continuous_vars,
                    quote = FALSE, noSpaces = TRUE, printToggle = FALSE)

tbl_df <- as.data.frame(tbl_matrix, stringsAsFactors = FALSE)
tbl_df$variable <- rownames(tbl_df)
rownames(tbl_df) <- NULL
tbl_df <- tbl_df |> select(variable, everything())

csv_path <- file.path(TABLES_DIR, "table1_diagnostic.csv")
readr::write_csv(tbl_df, csv_path)
cat("  Saved CSV ->", csv_path, "\n")

xlsx_path <- file.path(TABLES_DIR, "table1_diagnostic.xlsx")
wb <- openxlsx::createWorkbook()
openxlsx::addWorksheet(wb, "Table1")
openxlsx::writeData(wb, "Table1", tbl_df, rowNames = FALSE)
hs <- openxlsx::createStyle(fontColour = "#FFFFFF", fgFill = "#4472C4",
                              halign = "CENTER", fontName = "Calibri",
                              fontSize = 11, textDecoration = "Bold",
                              border = "Bottom")
openxlsx::addStyle(wb, "Table1", style = hs, rows = 1, cols = 1:ncol(tbl_df))
openxlsx::setColWidths(wb, "Table1", cols = 1:ncol(tbl_df), widths = "auto")
openxlsx::saveWorkbook(wb, xlsx_path, overwrite = TRUE)
cat("  Saved XLSX ->", xlsx_path, "\n")

html_path <- file.path(TABLES_DIR, "table1_diagnostic.html")
html_rows <- apply(tbl_df, 1, function(r) {
  paste0("<tr>", paste0("<td>", r, "</td>", collapse = ""), "</tr>")
})
header <- paste0("<tr>", paste0("<th>", names(tbl_df), "</th>", collapse = ""), "</tr>")
html_body <- paste0(
  "<!DOCTYPE html><html><head><style>",
  "table{border-collapse:collapse;font-family:Arial;font-size:13px;}",
  "th{background:#4472C4;color:white;padding:6px 10px;}",
  "td{padding:4px 10px;border-bottom:1px solid #ddd;}",
  "tr:nth-child(even){background:#f2f2f2;}",
  "</style></head><body>",
  "<h2>Table 1. Baseline Characteristics by Liver Metastasis Status</h2>",
  "<table>", header, paste(html_rows, collapse = ""), "</table>",
  "</body></html>"
)
writeLines(html_body, html_path)
cat("  Saved HTML ->", html_path, "\n")

if (requireNamespace("flextable", quietly = TRUE) &&
    requireNamespace("officer", quietly = TRUE)) {
  library(flextable); library(officer)
  docx_path <- file.path(TABLES_DIR, "table1_diagnostic.docx")
  tryCatch({
    doc <- officer::read_docx() |>
      officer::body_add_par("Table 1. Baseline Characteristics by Liver Metastasis Status",
                             style = "heading 1") |>
      officer::body_add_flextable(
        flextable::flextable(tbl_df) |> flextable::autofit()
      )
    print(doc, target = docx_path)
    cat("  Saved DOCX ->", docx_path, "\n")
  }, error = function(e) cat("  [warn] DOCX export failed:", conditionMessage(e), "\n"))
} else {
  cat("  [info] flextable/officer not installed; DOCX skipped.\n")
}

cat("=== 01 complete ===\n\n")
