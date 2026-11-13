# config/config.R — paths, seed, and feature sets for the prognostic model pipeline.
# Assumes the working directory is this implementation's root when sourced.

PROJECT_ROOT <- getwd()
REPO_ROOT    <- dirname(dirname(dirname(PROJECT_ROOT)))

INPUT_FILE <- file.path(REPO_ROOT, "data", "restricted", "seer_processed",
                        "seer_prognostic_input.csv")

TABLES_DIR  <- file.path(PROJECT_ROOT, "outputs", "tables")
FIGURES_DIR <- file.path(PROJECT_ROOT, "outputs", "figures")
MODELS_DIR  <- file.path(PROJECT_ROOT, "outputs", "models")

SEED <- 20230529

# ---- Cohort split ------------------------------------------------------------
# "simple_random"    : plain 70/30 random split (paper-style default)
# "event_stratified" : stratified by os_event (deviation from paper — documented)
SPLIT_METHOD <- "simple_random"

# ---- Calibration ---------------------------------------------------------------
CALIBRATION_BOOTSTRAP_B <- 1000   # B=1000 matches paper; lower for speed testing

# ---- RF/RFE cross-validation --------------------------------------------------
RF_RFE_CV_FOLDS       <- 5        # K-fold CV for RMSE curve (Figure 4 analogue)
RF_RFE_REPEATS        <- 1        # number of CV repeats
RF_RFE_HORIZON_MONTHS <- 36       # time horizon for RMSE: 1 - S(t) vs y_t

# Candidate variable set for RF/RFE:
#   "all_estimable"  — all 17 paper-style predictors with nonzero variance (closest to paper Fig 4)
#   "screened_union" — union(univariate Cox p<0.05, Cox/AIC-selected)  [more conservative]
RF_RFE_CANDIDATE_MODE <- "all_estimable"

# ---- Survival outcome ----------------------------------------------------------
TIME_VAR  <- "os_months"
EVENT_VAR <- "os_event"

# ---- Time horizons --------------------------------------------------------------
TIME_HORIZONS_MONTHS <- c(12, 24, 36)
TIME_HORIZONS_LABELS <- c("1-year", "2-year", "3-year")

# ---- Contrasts: use treatment contrasts globally (reference-level HRs) --------
# Ensures Cox models show grade II vs I, III vs I, etc. (not polynomial)
options(contrasts = c("contr.treatment", "contr.treatment"))

# ---- Candidate predictors: 17 paper-style clinical features --------------------
CANDIDATE_PREDICTORS <- c(
  "age_at_diagnosis",
  "race",
  "sex_male",
  "married",
  "histology_prognostic_group",
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
  "race", "histology_prognostic_group", "grade",
  "primary_site_group", "ajcc_t", "ajcc_n", "tumor_size_group"
)

CONTINUOUS_VARS <- c("age_at_diagnosis", "tumor_size_mm", "os_months")

# ---- Model feature sets ---------------------------------------------------------

# PRIMARY model: paper Figure 5 nomogram variables
FEATURES_NOMOGRAM_FIGURE <- c(
  "mets_brain",
  "age_at_diagnosis",
  "grade",
  "histology_prognostic_group",
  "surgery_primary"
)

# AUDIT/SENSITIVITY model: variables screened in paper text / Table 4
FEATURES_TEXT_SCREENED <- c(
  "grade",
  "n_stage_positive",
  "surgery_primary",
  "chemotherapy",
  "tumor_size_group",
  "mets_bone"
)

FINAL_MODEL_CHOICE <- "nomogram_figure"

# T/N comparator (no M stage — all patients have liver mets, M is constant)
FEATURES_TN_BASELINE <- c("ajcc_t", "n_stage_positive")

# ---- Helper functions ------------------------------------------------------------

pretty_var_label <- function(x) {
  lookup <- c(
    age_at_diagnosis           = "Age",
    sex_male                   = "Sex",
    race                       = "Race",
    married                    = "Marital status",
    histology_prognostic_group = "Histological subtype",
    grade                      = "Grade",
    primary_site_group         = "Primary site",
    ajcc_t                     = "AJCC T stage",
    ajcc_n                     = "AJCC N stage",
    n_stage_positive           = "AJCC N stage",
    surgery_primary            = "Surgery",
    surgery_lymph_node         = "LN surgery",
    radiotherapy               = "Radiotherapy",
    chemotherapy               = "Chemotherapy",
    tumor_size_group           = "Tumor size",
    tumor_size_mm              = "Tumor size (mm)",
    mets_bone                  = "Bone metastasis",
    mets_brain                 = "Brain metastasis",
    mets_lung                  = "Lung metastasis"
  )
  ifelse(x %in% names(lookup), lookup[x], x)
}

pretty_var_list <- function(vars, sep = " + ", max_chars = 120) {
  txt <- paste(pretty_var_label(vars), collapse = sep)
  if (nchar(txt) > max_chars) txt <- paste0(substr(txt, 1, max_chars - 3), "...")
  txt
}

model_feature_label <- function(model_name, features) {
  paste0(model_name, " (", length(features), " vars: ",
         pretty_var_list(features), ")")
}

# Appends one row to model_sample_sizes_prognostic.csv
log_sample_size <- function(script, model_name, n_total, n_used) {
  row <- data.frame(
    script     = script,
    model_name = model_name,
    n_total    = n_total,
    n_used     = n_used,
    n_dropped  = n_total - n_used,
    stringsAsFactors = FALSE
  )
  csv_path <- file.path(TABLES_DIR, "model_sample_sizes_prognostic.csv")
  if (file.exists(csv_path)) {
    prev <- tryCatch(read.csv(csv_path, stringsAsFactors = FALSE),
                     error = function(e) data.frame())
    write.csv(rbind(prev, row), csv_path, row.names = FALSE)
  } else {
    write.csv(row, csv_path, row.names = FALSE)
  }
}

# Appends one row to methodological_notes_prognostic.csv
log_method_note <- function(component, method, note) {
  row <- data.frame(component = component, method = method, note = note,
                    stringsAsFactors = FALSE)
  path <- file.path(TABLES_DIR, "methodological_notes_prognostic.csv")
  if (file.exists(path)) {
    prev <- tryCatch(read.csv(path, stringsAsFactors = FALSE),
                     error = function(e) data.frame())
    write.csv(rbind(prev, row), path, row.names = FALSE)
  } else {
    write.csv(row, path, row.names = FALSE)
  }
}
