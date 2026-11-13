# scripts/09_run_all_diagnostic.R
# Run the complete diagnostic model pipeline in order.
#   Rscript scripts/09_run_all_diagnostic.R
#   or from R: source("scripts/09_run_all_diagnostic.R")

cat(strrep("=", 65), "\n")
cat("  Diagnostic Model Pipeline — Li et al. 2023 Reproduction\n")
cat(strrep("=", 65), "\n\n")

pipeline <- c(
  "scripts/00_setup_diagnostic.R",
  "scripts/01_table1_diagnostic.R",
  "scripts/02_lasso_feature_selection.R",
  "scripts/03_random_forest_importance.R",
  "scripts/04_logistic_regression_models.R",
  "scripts/05_diagnostic_nomogram.R",
  "scripts/06_roc_analysis.R",
  "scripts/07_calibration_analysis.R",
  "scripts/08_dca_analysis.R",
  "scripts/10_combined_diagnostic_figures.R"
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
  "figures/fig1_ab_lasso_combined.png",
  "figures/fig1_cd_rf_importance_combined.png",
  "figures/diagnostic_nomogram.png",
  "figures/fig3_abc_pipeline_selected_evaluation.png"
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

cat("\nDone.\n")
