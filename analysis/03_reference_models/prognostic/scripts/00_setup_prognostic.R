# scripts/00_setup_prognostic.R
# Load, validate, clean, and split the prognostic dataset.
#
# - os_months == 0 is replaced with 0.5 for Cox model compatibility; the
#   original value is preserved in os_months_original.
# - Cohort split controlled by SPLIT_METHOD in config (default: simple_random).
# - Seed search ensures all primary Figure 5 categorical predictors have at
#   least one non-reference case in the training cohort.
# - grade, tumor_size_group, ajcc_t stored as unordered factors (treatment
#   contrasts, not polynomial) for interpretable Cox hazard ratios.

cat("=== 00_setup_prognostic.R ===\n")

source("config/config.R")

for (pkg in c("dplyr", "readr", "forcats")) {
  if (!requireNamespace(pkg, quietly = TRUE))
    stop("Install '", pkg, "' with: install.packages('", pkg, "')")
}
suppressPackageStartupMessages({
  library(dplyr); library(readr); library(forcats)
})

for (d in c(TABLES_DIR, FIGURES_DIR, MODELS_DIR)) {
  if (!dir.exists(d)) dir.create(d, recursive = TRUE)
}

# ---- Load data ----------------------------------------------------------------
cat("  Reading:", INPUT_FILE, "\n")
df_raw <- read_csv(INPUT_FILE, show_col_types = FALSE)
cat("  Dimensions:", nrow(df_raw), "x", ncol(df_raw), "\n")

required_cols <- c(
  "patient_id", "age_at_diagnosis", "sex_male", "race", "married",
  "histology_prognostic_group", "grade", "primary_site_group",
  "ajcc_t", "ajcc_n", "n_stage_positive",
  "surgery_primary", "surgery_lymph_node", "chemotherapy", "radiotherapy",
  "tumor_size_mm", "tumor_size_group",
  "mets_bone", "mets_brain", "mets_lung", "mets_liver",
  "os_months", "os_event"
)
missing_cols <- setdiff(required_cols, names(df_raw))
if (length(missing_cols) > 0) {
  stop("Missing required columns: ", paste(missing_cols, collapse = ", "))
}

cat("  Validating survival outcome ...\n")
if (!all(df_raw$os_event %in% c(0, 1))) stop("os_event must be 0 or 1.")
if (any(df_raw$os_months < 0))          stop("os_months must be >= 0.")

n_no_liver <- sum(df_raw$mets_liver != 1)
if (n_no_liver > 0) {
  stop("Dataset contains ", n_no_liver,
       " rows where mets_liver != 1. Pre-filter to liver metastasis patients.")
}
cat("  All", nrow(df_raw), "rows have mets_liver == 1. OK.\n")

# ---- Handle zero survival times: preserve original -----------------------------
n_zero_time <- sum(df_raw$os_months == 0)
cat("  os_months == 0:", n_zero_time, "rows\n")

zero_ids <- if (n_zero_time > 0) df_raw$patient_id[df_raw$os_months == 0] else character(0)
zero_time_tbl <- if (n_zero_time > 0) {
  data.frame(
    patient_id    = zero_ids,
    os_months_raw = 0,
    os_months_adj = 0.5,
    note = "os_months == 0 replaced with 0.5 for Cox model compatibility"
  )
} else {
  data.frame(patient_id = character(0), os_months_raw = numeric(0),
             os_months_adj = numeric(0), note = character(0))
}
write_csv(zero_time_tbl, file.path(TABLES_DIR, "zero_survival_time_handling.csv"))

df <- df_raw |>
  mutate(
    os_months_original = os_months,
    os_months          = if_else(os_months == 0, 0.5, as.numeric(os_months))
  )

# ---- Missingness report ---------------------------------------------------------
miss_df <- data.frame(
  variable    = names(df),
  n_missing   = sapply(df, function(x) sum(is.na(x))),
  pct_missing = round(sapply(df, function(x) mean(is.na(x))) * 100, 2),
  stringsAsFactors = FALSE
)
write_csv(miss_df, file.path(TABLES_DIR, "missingness_prognostic.csv"))
if (sum(miss_df$n_missing > 0) == 0) {
  cat("  No missing values in any variable.\n")
} else {
  cat("  Variables with missing values:\n")
  print(miss_df[miss_df$n_missing > 0, ])
}

# ---- Convert binary variables to labeled factors --------------------------------
binary_labels <- list(
  sex_male           = c("0" = "Female",      "1" = "Male"),
  married            = c("0" = "Not married", "1" = "Married"),
  n_stage_positive   = c("0" = "N0",          "1" = "N1"),
  surgery_primary    = c("0" = "No",          "1" = "Yes"),
  surgery_lymph_node = c("0" = "No",          "1" = "Yes"),
  radiotherapy       = c("0" = "No",          "1" = "Yes"),
  chemotherapy       = c("0" = "No",          "1" = "Yes"),
  mets_bone          = c("0" = "No",          "1" = "Yes"),
  mets_brain         = c("0" = "No",          "1" = "Yes"),
  mets_lung          = c("0" = "No",          "1" = "Yes")
)
for (v in names(binary_labels)) {
  if (v %in% names(df)) {
    lbl      <- binary_labels[[v]]
    df[[v]]  <- factor(lbl[as.character(df[[v]])], levels = lbl)
  }
}
cat("  Binary vars converted to labeled factors.\n")

# ---- Grade, tumor_size_group, ajcc_t as unordered factors -----------------------
if ("grade" %in% names(df)) {
  df <- df |>
    mutate(grade = factor(
      case_when(
        as.character(grade) == "1" ~ "I",
        as.character(grade) == "2" ~ "II",
        as.character(grade) == "3" ~ "III",
        as.character(grade) == "4" ~ "IV",
        TRUE ~ as.character(grade)
      ),
      levels  = c("I", "II", "III", "IV"),
      ordered = FALSE
    ))
  cat("  grade levels:", paste(levels(df$grade), collapse = " | "), "\n")
}

if ("tumor_size_group" %in% names(df)) {
  df <- df |>
    mutate(tumor_size_group = factor(
      tumor_size_group,
      levels  = c("0-26 mm", "26-36 mm", ">36 mm"),
      ordered = FALSE
    ))
  cat("  tumor_size_group levels:",
      paste(levels(df$tumor_size_group), collapse = " | "), "\n")
}

if ("ajcc_t" %in% names(df)) {
  t_vals   <- sort(unique(as.character(df$ajcc_t)))
  t_order  <- intersect(c("T0", "T1", "T2", "T3", "T4"), t_vals)
  t_extra  <- setdiff(t_vals, t_order)
  t_levels <- c(t_order, t_extra)
  df <- df |> mutate(ajcc_t = factor(ajcc_t, levels = t_levels, ordered = FALSE))
  cat("  ajcc_t levels:", paste(t_levels, collapse = " | "), "\n")
}

if ("ajcc_n" %in% names(df)) {
  n_order <- intersect(c("N0", "N1"), sort(unique(as.character(df$ajcc_n))))
  df <- df |> mutate(ajcc_n = factor(ajcc_n, levels = n_order, ordered = FALSE))
  cat("  ajcc_n levels:", paste(n_order, collapse = " | "), "\n")
}

if ("histology_prognostic_group" %in% names(df)) {
  h_levels <- c("Carcinoid tumor", "Others")
  present_h <- intersect(h_levels, unique(as.character(df$histology_prognostic_group)))
  extra_h   <- setdiff(unique(as.character(df$histology_prognostic_group)), h_levels)
  df <- df |>
    mutate(histology_prognostic_group = factor(
      histology_prognostic_group, levels = c(present_h, extra_h), ordered = FALSE))
  cat("  histology_prognostic_group levels:",
      paste(levels(df$histology_prognostic_group), collapse = " | "), "\n")
}

for (v in c("race", "primary_site_group")) {
  if (v %in% names(df)) {
    df[[v]] <- factor(df[[v]])
    cat(" ", v, "levels:", paste(levels(df[[v]]), collapse = " | "), "\n")
  }
}

# ---- Save factor levels summary --------------------------------------------------
fac_vars <- names(df)[sapply(df, is.factor)]
fac_tbl <- do.call(rbind, lapply(fac_vars, function(v) {
  data.frame(variable = v,
             levels   = paste(levels(df[[v]]), collapse = " | "),
             ordered  = is.ordered(df[[v]]),
             stringsAsFactors = FALSE)
}))
write_csv(fac_tbl, file.path(TABLES_DIR, "factor_levels_prognostic.csv"))
cat("  Saved factor_levels_prognostic.csv\n")

# ---- Primary Figure 5 categorical predictors check -------------------------------
# Variables that must have >=2 levels in the training cohort for rms fitting.
primary_cat_vars <- setdiff(FEATURES_NOMOGRAM_FIGURE, CONTINUOUS_VARS)
primary_cat_vars <- intersect(primary_cat_vars, names(df))
primary_cat_vars <- primary_cat_vars[sapply(primary_cat_vars,
                                            function(v) is.factor(df[[v]]))]

check_split_ok <- function(train_idx) {
  df_train_trial <- df[train_idx, ]
  all(sapply(primary_cat_vars, function(v) {
    length(unique(droplevels(df_train_trial[[v]]))) >= 2
  }))
}

cat("  SPLIT_METHOD =", SPLIT_METHOD, "\n")

do_split <- function(seed_val) {
  set.seed(seed_val)
  if (SPLIT_METHOD == "event_stratified") {
    event_idx   <- which(df[[EVENT_VAR]] == 1)
    censor_idx  <- which(df[[EVENT_VAR]] == 0)
    c(sample(event_idx,  round(length(event_idx)  * 0.7), replace = FALSE),
      sample(censor_idx, round(length(censor_idx) * 0.7), replace = FALSE))
  } else {
    sample(seq_len(nrow(df)), round(nrow(df) * 0.7), replace = FALSE)
  }
}

# Search for a seed where primary categorical predictors are all estimable in training
found_seed    <- SEED
found_attempt <- 1L
train_idx     <- do_split(SEED)

if (!check_split_ok(train_idx)) {
  cat("  [note] Initial seed gives unestimable primary vars. Searching seeds ...\n")
  found <- FALSE
  for (attempt in seq_len(1000L)) {
    trial_seed <- SEED + attempt
    trial_idx  <- do_split(trial_seed)
    if (check_split_ok(trial_idx)) {
      found_seed    <- trial_seed
      found_attempt <- attempt
      train_idx     <- trial_idx
      found         <- TRUE
      cat("  Found valid split at attempt", attempt,
          "(seed offset:", attempt, ")\n")
      break
    }
  }
  if (!found) {
    absent_vars <- primary_cat_vars[sapply(primary_cat_vars, function(v) {
      length(unique(droplevels(df[[v]]))) <= 1
    })]
    cat("  [warn] Could not find a split where all primary vars are estimable in 1000 seeds.\n")
    cat("  Variables absent from full dataset:", paste(absent_vars, collapse = ", "), "\n")
    cat("  These will be excluded from rms estimation with a method note.\n")
    log_method_note(
      "Cohort split",
      "Seed search failed",
      paste0("After 1000 seed attempts, the following primary Figure 5 variables ",
             "remain unestimable in training because they are absent/constant in ",
             "the full dataset: ", paste(absent_vars, collapse = ", "))
    )
  }
}

if (SPLIT_METHOD == "event_stratified") {
  log_method_note(
    "Cohort split",
    "event_stratified (deviation from paper)",
    "Split stratified by os_event. Paper states only 7:3 ratio without specifying stratification."
  )
}

df <- df |>
  mutate(cohort = if_else(row_number() %in% train_idx,
                          "Training", "Internal validation"))
df$cohort <- factor(df$cohort, levels = c("Training", "Internal validation"))

cat("  Training N =", sum(df$cohort == "Training"),
    "| Validation N =", sum(df$cohort == "Internal validation"), "\n")

# Verify estimability of primary vars in training
df_train <- df[df$cohort == "Training", ]
for (v in primary_cat_vars) {
  lvls_train <- unique(droplevels(df_train[[v]]))
  lvls_full  <- levels(df[[v]])
  absent     <- setdiff(lvls_full, as.character(lvls_train))
  if (length(absent) > 0 && length(lvls_full) > 1) {
    cat("  [note] Primary var '", v, "': level(s) [",
        paste(absent, collapse = ", "), "] absent from training cohort.\n", sep = "")
    log_method_note(
      paste0("Primary var '", v, "' — split"),
      "Level absent from training",
      paste0("Full dataset has levels [", paste(lvls_full, collapse = ", "),
             "]. Training cohort missing: [", paste(absent, collapse = ", "),
             "]. Cannot estimate effect in rms; variable excluded from nomogram ",
             "with audit. Cox predictions still computed via survival::coxph.")
    )
  }
}

split_tbl <- as.data.frame(table(cohort = df$cohort, event = df[[EVENT_VAR]]))
split_tbl$cohort_total <- ave(split_tbl$Freq, split_tbl$cohort, FUN = sum)
cat("  Cohort split:\n"); print(split_tbl)
write_csv(split_tbl, file.path(TABLES_DIR, "split_counts_prognostic.csv"))

saveRDS(df, file.path(MODELS_DIR, "prognostic_data_clean.rds"))
cat("  Saved prognostic_data_clean.rds\n")

cat("=== 00 complete ===\n\n")
