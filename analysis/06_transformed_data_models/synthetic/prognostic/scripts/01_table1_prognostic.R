# scripts/01_table1_prognostic.R
# Baseline characteristics table (Table 3 analogue): Training vs Validation.
# os_months / os_event are outcomes, not baseline covariates, and are excluded
# from the main table; a separate survival/event audit table is saved.

cat("=== 01_table1_prognostic.R ===\n")

source("config/config.R")

for (pkg in c("tableone", "dplyr", "readr")) {
  if (!requireNamespace(pkg, quietly = TRUE))
    stop("Install '", pkg, "' with: install.packages('", pkg, "')")
}
suppressPackageStartupMessages({
  library(tableone); library(dplyr); library(readr)
})

df <- readRDS(file.path(MODELS_DIR, "prognostic_data_clean.rds"))
cat("  Data:", nrow(df), "rows |",
    sum(df$cohort == "Training"), "training |",
    sum(df$cohort == "Internal validation"), "validation\n")

table1_vars <- c(
  "age_at_diagnosis", "sex_male", "married", "race",
  "histology_prognostic_group", "grade",
  "primary_site_group", "ajcc_t", "ajcc_n", "n_stage_positive",
  "surgery_primary", "surgery_lymph_node", "radiotherapy", "chemotherapy",
  "tumor_size_group", "tumor_size_mm",
  "mets_bone", "mets_brain", "mets_lung"
)
table1_vars  <- intersect(table1_vars, names(df))
nonnormal_v  <- intersect(c("age_at_diagnosis", "tumor_size_mm"), table1_vars)
factor_vars  <- setdiff(table1_vars, nonnormal_v)

tbl <- tableone::CreateTableOne(
  vars       = table1_vars,
  strata     = "cohort",
  data       = df,
  factorVars = factor_vars
)

cat("  Table 3 analogue (baseline characteristics):\n")
print(tbl, nonnormal = nonnormal_v, showAllLevels = TRUE,
      quote = FALSE, noSpaces = TRUE, printToggle = TRUE)

tbl_mat <- print(tbl, nonnormal = nonnormal_v, showAllLevels = TRUE,
                 quote = FALSE, noSpaces = TRUE, printToggle = FALSE)
write.csv(tbl_mat, file.path(TABLES_DIR, "table1_prognostic.csv"))
cat("  Saved table1_prognostic.csv\n")

if (requireNamespace("knitr", quietly = TRUE) &&
    requireNamespace("kableExtra", quietly = TRUE)) {
  html_tbl <- knitr::kable(as.data.frame(tbl_mat), format = "html",
    caption = "Table 3 (analogue): Baseline characteristics — Training vs Internal validation") |>
    kableExtra::kable_styling(bootstrap_options = c("striped", "hover"))
  writeLines(as.character(html_tbl),
             file.path(TABLES_DIR, "table1_prognostic.html"))
  cat("  Saved table1_prognostic.html\n")
} else {
  cat("  [info] knitr/kableExtra not available; HTML skipped.\n")
}

if (requireNamespace("openxlsx", quietly = TRUE)) {
  wb <- openxlsx::createWorkbook()
  openxlsx::addWorksheet(wb, "Table3")
  openxlsx::writeData(wb, "Table3", tbl_mat, rowNames = TRUE)
  openxlsx::saveWorkbook(wb,
    file.path(TABLES_DIR, "table1_prognostic.xlsx"), overwrite = TRUE)
  cat("  Saved table1_prognostic.xlsx\n")
} else {
  cat("  [info] openxlsx not available; XLSX skipped.\n")
}

# ---- Survival outcome audit table (separate from Table 3) -----------------------
survival_audit <- df |>
  group_by(cohort) |>
  summarise(
    n              = n(),
    n_events       = sum(.data[[EVENT_VAR]] == 1),
    n_censored     = sum(.data[[EVENT_VAR]] == 0),
    event_rate_pct = round(mean(.data[[EVENT_VAR]] == 1) * 100, 1),
    median_os_months = round(median(.data[[TIME_VAR]], na.rm = TRUE), 1),
    min_os_months    = round(min(.data[[TIME_VAR]], na.rm = TRUE), 1),
    max_os_months    = round(max(.data[[TIME_VAR]], na.rm = TRUE), 1),
    n_zero_os_original = sum(.data[["os_months_original"]] == 0,
                              na.rm = TRUE),
    .groups = "drop"
  )

write_csv(survival_audit, file.path(TABLES_DIR, "split_survival_outcome_audit.csv"))
cat("  Saved split_survival_outcome_audit.csv\n")
print(as.data.frame(survival_audit))

cat("=== 01 complete ===\n\n")
