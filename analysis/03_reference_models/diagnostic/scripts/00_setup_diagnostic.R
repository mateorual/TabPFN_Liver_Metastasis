# scripts/00_setup_diagnostic.R
# Validate data, apply clinical factor ordering/labels, save clean dataset.

cat("=== 00_setup_diagnostic.R ===\n")

source("config/config.R")

required_pkgs <- c("readr", "dplyr", "tidyr", "forcats")
for (pkg in required_pkgs) {
  if (!requireNamespace(pkg, quietly = TRUE))
    stop("Package '", pkg, "' not installed. Run: install.packages('", pkg, "')")
}
suppressPackageStartupMessages({
  library(readr); library(dplyr); library(tidyr); library(forcats)
})

for (d in c(TABLES_DIR, FIGURES_DIR, MODELS_DIR)) {
  dir.create(d, showWarnings = FALSE, recursive = TRUE)
}

# Fresh sample-size log for this pipeline run
write.csv(
  data.frame(script = character(0), model_name = character(0),
             n_total = integer(0), n_used = integer(0), n_dropped = integer(0)),
  file.path(TABLES_DIR, "model_sample_sizes_diagnostic.csv"),
  row.names = FALSE
)

# ---- Load raw data ----------------------------------------------------------
cat("  Reading:", INPUT_FILE, "\n")
if (!file.exists(INPUT_FILE)) stop("Input file not found: ", INPUT_FILE)
df_raw <- readr::read_csv(INPUT_FILE, show_col_types = FALSE)
cat("  Dimensions:", nrow(df_raw), "x", ncol(df_raw), "\n")

if (!OUTCOME %in% names(df_raw))
  stop("Outcome '", OUTCOME, "' not found in dataset.")
stopifnot("Outcome must be binary 0/1" =
            all(df_raw[[OUTCOME]] %in% c(0L, 1L, NA_integer_)))
cat("  Outcome distribution:\n")
print(table(df_raw[[OUTCOME]], useNA = "always"))

present_preds <- intersect(CANDIDATE_PREDICTORS, names(df_raw))
missing_preds <- setdiff(CANDIDATE_PREDICTORS, names(df_raw))
if (length(missing_preds) > 0)
  warning("Predictors not in data: ", paste(missing_preds, collapse = ", "))
cat("  Candidate predictors present:", length(present_preds), "/",
    length(CANDIDATE_PREDICTORS), "\n")

# ---- Missingness report -----------------------------------------------------
report_vars <- unique(c(OUTCOME, present_preds,
                        intersect(c("ajcc_n", "tumor_size_mm"), names(df_raw))))
miss_df <- df_raw |>
  select(all_of(intersect(report_vars, names(df_raw)))) |>
  summarise(across(everything(), ~ sum(is.na(.)))) |>
  tidyr::pivot_longer(everything(), names_to = "variable", values_to = "n_missing") |>
  mutate(n_total = nrow(df_raw), pct_missing = round(100 * n_missing / n_total, 2))

readr::write_csv(miss_df, file.path(TABLES_DIR, "missingness_diagnostic.csv"))
if (sum(miss_df$n_missing) == 0) {
  cat("  No missing values in any variable.\n")
} else {
  print(miss_df |> filter(n_missing > 0))
}

# ---- Build dataset ----------------------------------------------------------
keep_vars <- unique(c("patient_id", OUTCOME, present_preds,
                      intersect(c("ajcc_n", "tumor_size_mm"), names(df_raw))))
df <- df_raw |>
  select(all_of(intersect(keep_vars, names(df_raw)))) |>
  mutate(!!OUTCOME := as.integer(.data[[OUTCOME]]))

# ---- Binary predictors -> labeled factors (reference = "No"/lower-risk level) ---
binary_labels <- list(
  sex_male           = c("Female",      "Male"),
  married            = c("Not married", "Married"),
  n_stage_positive   = c("N0",          "N1"),
  surgery_primary    = c("No",          "Yes"),
  surgery_lymph_node = c("No",          "Yes"),
  radiotherapy       = c("No",          "Yes"),
  chemotherapy       = c("No",          "Yes"),
  mets_bone          = c("No",          "Yes"),
  mets_brain         = c("No",          "Yes"),
  mets_lung          = c("No",          "Yes")
)

for (bv in names(binary_labels)) {
  if (bv %in% names(df)) {
    lbl <- binary_labels[[bv]]
    df <- df |>
      mutate(!!bv := factor(as.integer(.data[[bv]]),
                             levels = c(0L, 1L), labels = lbl))
  }
}

# ---- Grade: explicit clinical ordering (I < II < III < IV) -----------------
if ("grade" %in% names(df)) {
  raw_grades <- unique(as.character(df$grade[!is.na(df$grade)]))

  grade_orders <- list(
    c("1", "2", "3", "4"),
    c("I", "II", "III", "IV"),
    c("Grade I",  "Grade II",  "Grade III",  "Grade IV"),
    c("Grade 1",  "Grade 2",   "Grade 3",    "Grade 4"),
    c("G1", "G2", "G3", "G4"),
    c("Well differentiated",  "Moderately differentiated",
      "Poorly differentiated", "Undifferentiated"),
    c("Well Differentiated",  "Moderately Differentiated",
      "Poorly Differentiated", "Undifferentiated")
  )

  matched_order <- NULL
  for (ord in grade_orders) {
    if (all(raw_grades %in% ord)) {
      matched_order <- ord[ord %in% raw_grades]
      break
    }
  }
  if (is.null(matched_order)) {
    warning("Grade values [", paste(raw_grades, collapse = ", "),
            "] do not match any known ordering. Using sorted levels — verify correctness.")
    matched_order <- sort(raw_grades)
  }

  df <- df |> mutate(grade = factor(as.character(grade), levels = matched_order))
  cat("  Grade levels (low-to-high):", paste(levels(df$grade), collapse = " | "), "\n")
}

# ---- Tumor size group: ordered 0-26 mm < 26-36 mm < >36 mm -----------------
if ("tumor_size_group" %in% names(df)) {
  raw_ts <- unique(as.character(df$tumor_size_group[!is.na(df$tumor_size_group)]))
  tsize_orders <- list(
    c("0-26 mm",  "26-36 mm",  ">36 mm"),
    c("0-26mm",   "26-36mm",   ">36mm"),
    c("<=26mm",   "26-36mm",   ">36mm"),
    c("<=26 mm",  "26-36 mm",  ">36 mm")
  )
  matched_ts <- NULL
  for (ord in tsize_orders) {
    if (all(raw_ts %in% ord)) { matched_ts <- ord[ord %in% raw_ts]; break }
  }
  if (is.null(matched_ts)) {
    warning("tumor_size_group values [", paste(raw_ts, collapse = ", "),
            "] do not match known patterns. Using sorted levels.")
    matched_ts <- sort(raw_ts)
  }
  df <- df |>
    mutate(tumor_size_group = factor(as.character(tumor_size_group), levels = matched_ts))
}

# ---- AJCC T stage: ordered T0 < T1 < T2 < T3 < T4 --------------------------
if ("ajcc_t" %in% names(df)) {
  raw_t      <- unique(as.character(df$ajcc_t[!is.na(df$ajcc_t)]))
  t_template <- c("T0", "T1", "T1a", "T1b", "T2", "T3", "T4", "Tis", "TX")
  ordered_t  <- c(intersect(t_template, raw_t), setdiff(raw_t, t_template))
  df <- df |> mutate(ajcc_t = factor(as.character(ajcc_t), levels = ordered_t))
}

# ---- AJCC N stage (kept for Table 1): ordered N0 < N1 < NX -----------------
if ("ajcc_n" %in% names(df)) {
  raw_n      <- unique(as.character(df$ajcc_n[!is.na(df$ajcc_n)]))
  n_template <- c("N0", "N1", "N2", "N3", "NX")
  ordered_n  <- c(intersect(n_template, raw_n), setdiff(raw_n, n_template))
  df <- df |> mutate(ajcc_n = factor(as.character(ajcc_n), levels = ordered_n))
}

# ---- Race (White as reference) ---------------------------------------------
if ("race" %in% names(df)) {
  ref_race <- if ("White" %in% as.character(df$race)) "White" else
    names(sort(table(df$race), decreasing = TRUE))[1]
  df <- df |> mutate(race = relevel(factor(race), ref = ref_race))
}

# ---- Histology (most common as reference) ----------------------------------
if ("histology_paper" %in% names(df)) {
  mc <- names(sort(table(df$histology_paper), decreasing = TRUE))[1]
  df <- df |> mutate(histology_paper = relevel(factor(histology_paper), ref = mc))
}

# ---- Primary site (clinical order if recognizable) --------------------------
if ("primary_site_group" %in% names(df)) {
  raw_sites     <- unique(as.character(df$primary_site_group[!is.na(df$primary_site_group)]))
  site_template <- c("Head of pancreas", "Body of pancreas", "Tail of pancreas", "Others",
                     "Head", "Body", "Tail", "Other")
  ordered_sites <- c(intersect(site_template, raw_sites), setdiff(raw_sites, site_template))
  df <- df |>
    mutate(primary_site_group = factor(as.character(primary_site_group),
                                       levels = ordered_sites))
}

# ---- Save factor-level table -----------------------------------------------
fvars <- unique(c(CATEGORICAL_VARS, BINARY_VARS))
flevels_rows <- lapply(intersect(fvars, names(df)), function(v) {
  if (is.factor(df[[v]]))
    data.frame(variable    = v,
               level_order = paste(levels(df[[v]]), collapse = " | "),
               stringsAsFactors = FALSE)
})
flevels_df <- do.call(rbind, Filter(Negate(is.null), flevels_rows))
readr::write_csv(flevels_df, file.path(TABLES_DIR, "factor_levels_diagnostic.csv"))

saveRDS(df, file.path(MODELS_DIR, "diagnostic_data_clean.rds"))
cat("  Saved clean data ->", file.path(MODELS_DIR, "diagnostic_data_clean.rds"), "\n")

cat("=== 00 complete ===\n\n")
