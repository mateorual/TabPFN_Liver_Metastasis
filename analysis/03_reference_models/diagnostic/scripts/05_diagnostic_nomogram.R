# scripts/05_diagnostic_nomogram.R
# Diagnostic nomogram (rms::lrm + rms::nomogram) -> diagnostic_nomogram figure.

cat("=== 05_diagnostic_nomogram.R ===\n")

source("config/config.R")

for (pkg in c("rms", "dplyr", "readr")) {
  if (!requireNamespace(pkg, quietly = TRUE))
    stop("Install '", pkg, "' with: install.packages('", pkg, "')")
}
suppressPackageStartupMessages({ library(rms); library(dplyr); library(readr) })

df <- readRDS(file.path(MODELS_DIR, "diagnostic_data_clean.rds"))

features <- switch(FINAL_MODEL_CHOICE,
  paper_text      = FEATURES_PAPER_TEXT,
  nomogram_figure = FEATURES_NOMOGRAM_FIGURE,
  FEATURES_NOMOGRAM_FIGURE
)

present_feats <- intersect(features, names(df))
df_model <- df |> select(all_of(c(present_feats, OUTCOME))) |> na.omit()

n_drop <- nrow(df) - nrow(df_model)
if (n_drop > 0) cat("  Dropped", n_drop, "rows (complete-case).\n")
log_sample_size("05_nomogram", paste0("lrm_", FINAL_MODEL_CHOICE),
                nrow(df), nrow(df_model))

levels_rows <- lapply(present_feats, function(v) {
  if (is.factor(df_model[[v]])) {
    data.frame(variable = v,
               levels   = paste(levels(df_model[[v]]), collapse = " | "),
               stringsAsFactors = FALSE)
  } else {
    data.frame(variable = v, levels = "continuous", stringsAsFactors = FALSE)
  }
})
levels_df <- do.call(rbind, levels_rows)
readr::write_csv(levels_df, file.path(TABLES_DIR, "nomogram_variable_levels.csv"))

# Grade must be in low-to-high clinical order, or the nomogram misreads risk direction.
if ("grade" %in% present_feats && is.factor(df_model$grade)) {
  grade_lvls <- levels(df_model$grade)
  idx_3 <- grep("3|III|Poorly|poorly", grade_lvls)
  idx_4 <- grep("4|IV|Undiff|undiff", grade_lvls)
  if (length(idx_3) > 0 && length(idx_4) > 0 && idx_4[1] < idx_3[1]) {
    stop("Grade levels appear to be in wrong clinical order: ",
         paste(grade_lvls, collapse = " | "),
         "\nFix grade ordering in 00_setup_diagnostic.R before running the nomogram.")
  }
}

dd <- rms::datadist(df_model)
options(datadist = "dd")

rms_formula <- as.formula(paste0(OUTCOME, " ~ ", paste(present_feats, collapse = " + ")))
cat("  Formula:", deparse(rms_formula), "\n")

lrm_fit <- tryCatch(
  rms::lrm(rms_formula, data = df_model, x = TRUE, y = TRUE),
  error = function(e) {
    cat("  [error] rms::lrm failed:", conditionMessage(e), "\n")
    NULL
  }
)
if (is.null(lrm_fit)) stop("lrm model could not be fitted — nomogram skipped.")
cat("  lrm fitted. C-statistic:", round(lrm_fit$stats["C"], 3), "\n")

nom <- tryCatch(
  rms::nomogram(
    lrm_fit, fun = plogis,
    fun.at   = c(0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    funlabel = "Predicted Probability\nof Liver Metastasis",
    lp.at    = seq(-3, 3, by = 0.5)
  ),
  error = function(e) {
    cat("  [warn] nomogram() with lp.at failed:", conditionMessage(e), "\nRetrying with defaults.\n")
    rms::nomogram(lrm_fit, fun = plogis,
                  funlabel = "Predicted Probability of Liver Metastasis")
  }
)

png(file.path(FIGURES_DIR, "diagnostic_nomogram.png"),
    width = 10, height = 7, units = "in", res = 150)
plot(nom, cex.axis = 0.75, cex.var = 0.85,
     main = "Diagnostic Nomogram for Liver Metastasis Risk")
dev.off()

pdf(file.path(FIGURES_DIR, "diagnostic_nomogram.pdf"), width = 10, height = 7)
plot(nom, cex.axis = 0.75, cex.var = 0.85,
     main = "Diagnostic Nomogram for Liver Metastasis Risk")
dev.off()
cat("  Saved diagnostic_nomogram figures.\n")

saveRDS(lrm_fit, file.path(MODELS_DIR, "diagnostic_nomogram_lrm_model.rds"))

cat("=== 05 complete ===\n\n")
