# config/config.R — paths, seed, and feature sets for the diagnostic model pipeline.
# Assumes the working directory is this implementation's root when sourced.

PROJECT_ROOT <- getwd()
REPO_ROOT    <- dirname(dirname(dirname(PROJECT_ROOT)))

INPUT_FILE <- file.path(REPO_ROOT, "data", "restricted", "seer_processed",
                        "seer_diagnostic_input.csv")

TABLES_DIR  <- file.path(PROJECT_ROOT, "outputs", "tables")
FIGURES_DIR <- file.path(PROJECT_ROOT, "outputs", "figures")
MODELS_DIR  <- file.path(PROJECT_ROOT, "outputs", "models")

SEED <- 20230529

OUTCOME <- "mets_liver"

# 17 candidate predictors. tumor_size_mm excluded (paper used binned groups);
# ajcc_n excluded (n_stage_positive used instead, to avoid duplicated N-stage info).
CANDIDATE_PREDICTORS <- c(
  "age_at_diagnosis",
  "race",
  "sex_male",
  "married",
  "histology_paper",
  "grade",
  "primary_site_group",
  "ajcc_t",
  "n_stage_positive",
  "surgery_primary",
  "surgery_lymph_node",
  "radiotherapy",
  "chemotherapy",
  "tumor_size_group",
  "mets_bone",
  "mets_brain",
  "mets_lung"
)

BINARY_VARS <- c(
  "sex_male", "married", "n_stage_positive",
  "surgery_primary", "surgery_lymph_node",
  "radiotherapy", "chemotherapy",
  "mets_bone", "mets_brain", "mets_lung"
)

CATEGORICAL_VARS <- c(
  "race", "histology_paper", "grade",
  "primary_site_group", "ajcc_t", "ajcc_n", "tumor_size_group"
)

CONTINUOUS_VARS <- c("age_at_diagnosis", "tumor_size_mm")

# Sensitivity model: paper text cites chemotherapy as an independent predictor
FEATURES_PAPER_TEXT <- c(
  "grade", "n_stage_positive", "surgery_primary",
  "chemotherapy", "tumor_size_group", "mets_bone"
)

# Primary model: variables shown in the paper's diagnostic nomogram figure
FEATURES_NOMOGRAM_FIGURE <- c(
  "grade", "n_stage_positive", "surgery_primary",
  "tumor_size_group", "mets_bone"
)

# Model used for the primary nomogram/evaluation: "paper_text" | "nomogram_figure"
FINAL_MODEL_CHOICE <- "nomogram_figure"

# TNM baseline — no M-stage, to avoid information leakage with the liver-mets outcome
FEATURES_TNM_BASELINE <- c("ajcc_t", "n_stage_positive")

# Appends one row to model_sample_sizes_diagnostic.csv after complete-case filtering.
log_sample_size <- function(script, model_name, n_total, n_used) {
  row <- data.frame(
    script     = script,
    model_name = model_name,
    n_total    = n_total,
    n_used     = n_used,
    n_dropped  = n_total - n_used,
    stringsAsFactors = FALSE
  )
  csv_path <- file.path(TABLES_DIR, "model_sample_sizes_diagnostic.csv")
  if (file.exists(csv_path)) {
    prev <- tryCatch(read.csv(csv_path, stringsAsFactors = FALSE),
                     error = function(e) data.frame())
    write.csv(rbind(prev, row), csv_path, row.names = FALSE)
  } else {
    write.csv(row, csv_path, row.names = FALSE)
  }
}
