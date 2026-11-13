# scripts/11_combined_paper_style_figures.R
# Cross-script audit tables: full feature-selection stage summary, plot
# legend labels, and reporting notes for the manuscript/thesis.
# Figures 4-8 are saved directly by their producing scripts (04, 06, 07, 09, 10).

cat("=== 11_combined_paper_style_figures.R ===\n")

source("config/config.R")

for (pkg in c("dplyr", "readr")) {
  if (!requireNamespace(pkg, quietly = TRUE))
    stop("Install '", pkg, "' with: install.packages('", pkg, "')")
}
suppressPackageStartupMessages({ library(dplyr); library(readr) })

rf_rfe_vars <- tryCatch(readRDS(file.path(MODELS_DIR, "rf_rfe_selected_vars.rds")),
                        error = function(e) character(0))
aic_vars    <- tryCatch(readRDS(file.path(MODELS_DIR, "cox_aic_selected_vars.rds")),
                        error = function(e) character(0))
ps_vars     <- tryCatch(readRDS(file.path(MODELS_DIR, "pipeline_selected_features.rds")),
                        error = function(e) character(0))

make_row <- function(stage, vars) {
  data.frame(stage = stage, n_variables = length(vars),
             variables = paste(vars, collapse = ", "),
             stringsAsFactors = FALSE)
}

audit_tbl <- bind_rows(
  make_row("Cox/AIC selected",         aic_vars),
  make_row("RF/RFE selected",          rf_rfe_vars),
  make_row("Pipeline-selected final",  ps_vars),
  make_row("Nomogram Figure (primary)", FEATURES_NOMOGRAM_FIGURE),
  make_row("Paper text screened (audit)", FEATURES_TEXT_SCREENED),
  make_row("AJCC T/N baseline",        FEATURES_TN_BASELINE)
)
write_csv(audit_tbl,
          file.path(TABLES_DIR, "pipeline_feature_selection_audit_prognostic.csv"))
cat("  Saved pipeline_feature_selection_audit_prognostic.csv\n")
print(audit_tbl[, c("stage", "n_variables")])

legend_rows <- data.frame(
  figure        = c("ROC / DCA", "ROC / DCA"),
  model_key     = c("nomogram_figure", "tn_baseline"),
  display_label = c(
    model_feature_label("Nomogram", FEATURES_NOMOGRAM_FIGURE),
    model_feature_label("AJCC T/N baseline", FEATURES_TN_BASELINE)
  ),
  stringsAsFactors = FALSE
)
write_csv(legend_rows, file.path(TABLES_DIR, "plot_legend_audit_prognostic.csv"))
cat("  Saved plot_legend_audit_prognostic.csv\n")

reporting_notes <- paste(
  "Kaplan-Meier curves were generated for low- and high-risk groups defined",
  "using the median linear predictor from the training cohort. The",
  "training-derived threshold was applied unchanged to the internal validation",
  "cohort. Survival curves are shown with 95% confidence intervals and",
  "number-at-risk tables. Differences between groups were evaluated using the",
  "log-rank test.",
  "",
  "For the random-forest feature-selection curve, the original article reported",
  "RMSE from cross-validation but did not specify how RMSE was computed for",
  "censored survival data. We therefore implemented a cross-validated random",
  "survival forest approximation. Variables were ranked by random survival",
  "forest importance, models were refitted iteratively using the top k",
  "variables, and cross-validated RMSE was computed using predicted expected",
  "survival time among uncensored validation patients. This figure should be",
  "interpreted as a methodological analogue rather than an exact reconstruction",
  "of the original authors' code.",
  sep = "\n"
)
writeLines(reporting_notes,
           file.path(TABLES_DIR, "prognostic_reporting_notes.txt"))
cat("  Saved prognostic_reporting_notes.txt\n")

cat("=== 11 complete ===\n\n")
