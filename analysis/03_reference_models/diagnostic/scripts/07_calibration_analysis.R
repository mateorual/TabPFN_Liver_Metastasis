# scripts/07_calibration_analysis.R
# Calibration tables: bootstrap calibration curve (rms::calibrate) and
# Brier score / intercept / slope, for the primary model and pipeline_selected.

cat("=== 07_calibration_analysis.R ===\n")

source("config/config.R")

for (pkg in c("rms", "dplyr", "readr")) {
  if (!requireNamespace(pkg, quietly = TRUE))
    stop("Install '", pkg, "' with: install.packages('", pkg, "')")
}
suppressPackageStartupMessages({ library(rms); library(dplyr); library(readr) })

set.seed(SEED)

lrm_path <- file.path(MODELS_DIR, "diagnostic_nomogram_lrm_model.rds")
if (!file.exists(lrm_path)) stop("lrm model not found. Run 05_diagnostic_nomogram.R first.")
lrm_fit <- readRDS(lrm_path)
df <- readRDS(file.path(MODELS_DIR, "diagnostic_data_clean.rds"))

features <- switch(FINAL_MODEL_CHOICE,
  paper_text      = FEATURES_PAPER_TEXT,
  nomogram_figure = FEATURES_NOMOGRAM_FIGURE,
  FEATURES_NOMOGRAM_FIGURE
)
present_feats <- intersect(features, names(df))
df_model <- df |> select(all_of(c(present_feats, OUTCOME))) |> na.omit()

dd <- rms::datadist(df_model)
options(datadist = "dd")

cat("  Running rms::calibrate() with 1000 bootstrap replicates ...\n")
cal <- tryCatch(
  rms::calibrate(lrm_fit, method = "boot", B = 1000),
  error = function(e) {
    cat("  [warn] rms::calibrate failed:", conditionMessage(e), "\nFalling back to apparent calibration.\n")
    tryCatch(rms::calibrate(lrm_fit, method = "add", B = 0), error = function(e2) NULL)
  }
)

# Extracts the calibration curve (predicted vs. observed) from an rms::calibrate()
# object and saves it as a wide-format table (no plot).
save_calibration_table <- function(cal_obj, filename_stem) {
  if (is.null(cal_obj)) {
    cat("  [skip]", filename_stem, "- calibration object is NULL.\n")
    return(invisible(NULL))
  }
  cal_mat        <- unclass(cal_obj)
  available_cols <- colnames(cal_mat)

  pred_col <- intersect(c("predy", "mean.predicted"), available_cols)[1]
  obs_col  <- intersect(c("calibrated.orig", "calibrated.obs"), available_cols)[1]
  if (is.na(pred_col) || is.na(obs_col)) {
    cat("  [warn] Cannot identify calibration columns for", filename_stem, "\n")
    return(invisible(NULL))
  }

  keep <- !is.na(cal_mat[, pred_col]) & !is.na(cal_mat[, obs_col])
  cal_df <- data.frame(
    group          = seq_len(sum(keep)),
    mean_predicted = as.numeric(cal_mat[keep, pred_col]),
    apparent       = as.numeric(cal_mat[keep, obs_col]),
    ideal          = as.numeric(cal_mat[keep, pred_col])
  )
  cal_df$bias_corrected <- if ("calibrated.corrected" %in% available_cols)
    as.numeric(cal_mat[keep, "calibrated.corrected"]) else NA_real_

  write_csv(cal_df, file.path(TABLES_DIR, paste0(filename_stem, "_curve.csv")))
  write_csv(
    cal_df |> rename(decile_group = group) |> mutate(n_groups = nrow(cal_df)),
    file.path(TABLES_DIR, paste0(filename_stem, "_deciles.csv"))
  )
  cat("  Saved", filename_stem, "_curve.csv and _deciles.csv\n")
  invisible(cal_df)
}

save_calibration_table(cal, "calibration_diagnostic_model")

# ---- Calibration metrics: Brier score, intercept, slope --------------------
final_glm <- tryCatch(
  readRDS(file.path(MODELS_DIR, paste0("logistic_model_", FINAL_MODEL_CHOICE, ".rds"))),
  error = function(e) readRDS(file.path(MODELS_DIR, "logistic_model_final.rds"))
)

metrics_df <- tryCatch({
  probs <- predict(final_glm, newdata = df, type = "response", na.action = na.exclude)
  valid <- !is.na(probs)
  y_obs <- df[[OUTCOME]][valid]
  p_hat <- probs[valid]

  brier <- mean((y_obs - p_hat)^2)

  logit_p   <- qlogis(pmin(pmax(p_hat, 1e-6), 1 - 1e-6))
  cal_model <- glm(y_obs ~ logit_p, family = binomial())
  cal_int   <- coef(cal_model)[1]
  cal_slope <- coef(cal_model)[2]

  data.frame(
    metric = c("brier_score", "calibration_intercept", "calibration_slope"),
    value  = round(c(brier, cal_int, cal_slope), 4),
    interpretation = c(
      "Lower is better (0 = perfect); ~prevalence^2 for null",
      "0 = perfect intercept; >0 = underestimation",
      "1 = perfect slope; <1 = overconfident"
    )
  )
}, error = function(e) {
  cat("  [warn] Could not compute calibration metrics:", conditionMessage(e), "\n")
  data.frame(metric = character(0), value = numeric(0), interpretation = character(0))
})

write_csv(metrics_df, file.path(TABLES_DIR, "calibration_metrics_diagnostic.csv"))
print(metrics_df)

# ---- Supplementary: pipeline-selected calibration table --------------------
ps_feat_path <- file.path(MODELS_DIR, "pipeline_selected_features.rds")
ps_glm_path  <- file.path(MODELS_DIR, "logistic_model_pipeline_selected.rds")

if (file.exists(ps_feat_path) && file.exists(ps_glm_path)) {
  tryCatch({
    ps_feats   <- readRDS(ps_feat_path)
    present_ps <- intersect(ps_feats, names(df))
    df_ps      <- df |> select(all_of(c(present_ps, OUTCOME))) |> na.omit()

    dd_ps <- rms::datadist(df_ps)
    options(datadist = "dd_ps")

    lrm_ps <- rms::lrm(
      as.formula(paste0(OUTCOME, " ~ ", paste(present_ps, collapse = " + "))),
      data = df_ps, x = TRUE, y = TRUE
    )

    cat("  Running rms::calibrate() for pipeline_selected (B=1000) ...\n")
    cal_ps <- tryCatch(
      rms::calibrate(lrm_ps, method = "boot", B = 1000),
      error = function(e) {
        cat("  [warn] rms::calibrate failed for pipeline_selected:", conditionMessage(e), "\n")
        NULL
      }
    )
    save_calibration_table(cal_ps, "calibration_pipeline_selected")

    options(datadist = "dd")
  }, error = function(e) {
    cat("  [warn] Pipeline-selected calibration failed:", conditionMessage(e), "\n")
  })
} else {
  cat("  [skip] Pipeline-selected model or features not found.\n")
}

cat("=== 07 complete ===\n\n")
