# scripts/12_run_all_prognostic.R
# Run the complete prognostic model pipeline in order.
#   Rscript scripts/12_run_all_prognostic.R
#   or from R: source("scripts/12_run_all_prognostic.R")

cat(strrep("=", 65), "\n")
cat("  Prognostic Model Pipeline — Li et al. 2023 Reproduction\n")
cat(strrep("=", 65), "\n\n")

pipeline <- c(
  "scripts/00_setup_prognostic.R",
  "scripts/01_table1_prognostic.R",
  "scripts/02_univariate_cox.R",
  "scripts/03_cox_aic_rfe_feature_selection.R",
  "scripts/04_survival_rf_rfe_feature_selection.R",
  "scripts/05_multivariable_cox_models.R",
  "scripts/06_prognostic_nomogram.R",
  "scripts/07_time_dependent_roc.R",
  "scripts/08_survival_calibration.R",
  "scripts/09_survival_dca.R",
  "scripts/10_kaplan_meier_risk_groups.R",
  "scripts/11_combined_paper_style_figures.R"
)

start_total <- proc.time()

for (script in pipeline) {
  cat(strrep("-", 65), "\n")
  cat("  Stage:", script, "\n")
  cat(strrep("-", 65), "\n")
  t_start <- proc.time()

  tryCatch(
    withCallingHandlers(
      source(script, echo = FALSE, local = FALSE),
      warning = function(w) {
        cat("  [WARNING]", conditionMessage(w), "\n")
        invokeRestart("muffleWarning")
      }
    ),
    error = function(e) {
      cat("\n[PIPELINE ERROR] Script failed:", script, "\n")
      cat("  Error message:", conditionMessage(e), "\n")
      stop("Pipeline aborted at: ", script)
    }
  )

  elapsed <- proc.time() - t_start
  cat("  Elapsed:", round(elapsed["elapsed"], 1), "s\n\n")
}

elapsed_total <- proc.time() - start_total
cat(strrep("=", 65), "\n")
cat("  Pipeline completed in", round(elapsed_total["elapsed"], 1), "seconds.\n")
cat(strrep("=", 65), "\n\n")

# ---- Output summary ----------------------------------------------------------
source("config/config.R", local = TRUE)

MAIN_FIGURES <- c(
  "figures/fig4_rf_rfe_rmse.png",
  "figures/fig5_prognostic_nomogram.png",
  "figures/fig6_roc_training_internal_6panel.png",
  "figures/fig7_dca_training_internal_6panel.png",
  "figures/fig8_km_risk_groups.png"
)

check <- function(rel_path) {
  full <- file.path(PROJECT_ROOT, "outputs", rel_path)
  if (file.exists(full)) "[OK]     " else "[MISSING]"
}

cat("Figures:\n")
for (f in MAIN_FIGURES) cat(" ", check(f), f, "\n")

cat("\nTables:\n")
for (f in sort(list.files(file.path(PROJECT_ROOT, "outputs", "tables")))) {
  cat("  ", f, "\n")
}

cat("\nModels:\n")
for (f in sort(list.files(file.path(PROJECT_ROOT, "outputs", "models")))) {
  cat("  ", f, "\n")
}

ps_rds <- file.path(PROJECT_ROOT, "outputs", "models", "pipeline_selected_features.rds")
if (file.exists(ps_rds)) {
  ps_vars <- readRDS(ps_rds)
  cat("\nPipeline-selected final variables (", length(ps_vars), "):\n")
  for (v in ps_vars) cat("  -", v, "\n")
}

split_path <- file.path(PROJECT_ROOT, "outputs", "tables", "split_counts_prognostic.csv")
if (file.exists(split_path)) {
  cat("\nCohort split:\n")
  print(read.csv(split_path, stringsAsFactors = FALSE))
}

cat("\nDone.\n")
