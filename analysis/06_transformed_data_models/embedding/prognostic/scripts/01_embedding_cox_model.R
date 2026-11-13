# scripts/01_embedding_cox_model.R
# Ridge-penalized Cox regression (glmnet family="cox") on the 1024-dim
# TabPFN embedding matrix (event_emb_* + time_emb_* concatenated), fit on
# the TRAINING cohort only. Ridge because embeddings are dense/correlated
# with p=1024 > n_train (~323) — strong regularization is required just for
# the model to be identifiable, let alone avoid overfitting.
#
# Produces, for both Training and Internal validation cohorts:
#   - risk score (linear predictor) from the embedding Cox model
#   - predicted survival probability at 12/24/36 months (via
#     survival::survfit() dispatch on the fitted cv.glmnet Cox object)
# and the same two outputs from a plain AJCC T/N baseline coxph model
# (fit on training only, matching the clinical pipeline's baseline treatment).

cat("=== 01_embedding_cox_model.R ===\n")

source("config/config.R")

for (pkg in c("glmnet", "survival", "dplyr", "readr")) {
  if (!requireNamespace(pkg, quietly = TRUE))
    stop("Install '", pkg, "' with: install.packages('", pkg, "')")
}
suppressPackageStartupMessages({
  library(glmnet); library(survival); library(dplyr); library(readr)
})

df <- readRDS(file.path(MODELS_DIR, "embedding_prognostic_data_clean.rds"))
event_emb_cols <- readRDS(file.path(MODELS_DIR, "event_emb_columns.rds"))
time_emb_cols  <- readRDS(file.path(MODELS_DIR, "time_emb_columns.rds"))
emb_cols <- c(event_emb_cols, time_emb_cols)

df_t <- df[df$cohort == "Training", ]
df_v <- df[df$cohort == "Internal validation", ]
cat("  Training N =", nrow(df_t), "| Events =", sum(df_t[[EVENT_VAR]]),
    "| Validation N =", nrow(df_v), "\n")

X_t <- as.matrix(df_t[, emb_cols])
X_v <- as.matrix(df_v[, emb_cols])
y_t <- Surv(df_t[[TIME_VAR]], df_t[[EVENT_VAR]])
cat("  Embedding matrix (training):", nrow(X_t), "x", ncol(X_t), "\n")

# ---- Ridge Cox regression -----------------------------------------------------
set.seed(SEED)
cv_fit <- cv.glmnet(
  X_t, y_t, family = "cox", alpha = GLMNET_ALPHA, nfolds = GLMNET_NFOLDS,
  cox.ties = "efron"
)
cat("  Ridge Cox: lambda.min =", round(cv_fit$lambda.min, 5), "\n")

lp_t <- as.numeric(predict(cv_fit, newx = X_t, s = "lambda.min", type = "link"))
lp_v <- as.numeric(predict(cv_fit, newx = X_v, s = "lambda.min", type = "link"))

surv_fit_t <- survival::survfit(cv_fit, s = "lambda.min", x = X_t, y = y_t, newx = X_t)
surv_fit_v <- survival::survfit(cv_fit, s = "lambda.min", x = X_t, y = y_t, newx = X_v)
surv_t <- summary(surv_fit_t, times = TIME_HORIZONS_MONTHS)$surv  # [horizons x N_train]
surv_v <- summary(surv_fit_v, times = TIME_HORIZONS_MONTHS)$surv  # [horizons x N_val]

# ---- AJCC T/N baseline (coxph, fit on training only) --------------------------
tn_formula <- as.formula(paste0("Surv(", TIME_VAR, ", ", EVENT_VAR, ") ~ ",
                                paste(FEATURES_TN_BASELINE, collapse = " + ")))
tn_fit <- coxph(tn_formula, data = df_t, ties = "efron")
cat("  AJCC T/N baseline: N =", tn_fit$n, "| C-index =",
    round(as.numeric(summary(tn_fit)$concordance["C"]), 3), "\n")

tn_lp_t <- predict(tn_fit, newdata = df_t, type = "lp")
tn_lp_v <- predict(tn_fit, newdata = df_v, type = "lp")

tn_surv_fit_t <- survfit(tn_fit, newdata = df_t)
tn_surv_fit_v <- survfit(tn_fit, newdata = df_v)
tn_surv_t <- summary(tn_surv_fit_t, times = TIME_HORIZONS_MONTHS)$surv
tn_surv_v <- summary(tn_surv_fit_v, times = TIME_HORIZONS_MONTHS)$surv

# ---- Assemble risk-score table -------------------------------------------------
risk_tbl <- bind_rows(
  data.frame(patient_id = df_t$patient_id, cohort = "Training",
             time_months = df_t[[TIME_VAR]], event = df_t[[EVENT_VAR]],
             embedding_lp = lp_t, tn_baseline_lp = tn_lp_t, stringsAsFactors = FALSE),
  data.frame(patient_id = df_v$patient_id, cohort = "Internal validation",
             time_months = df_v[[TIME_VAR]], event = df_v[[EVENT_VAR]],
             embedding_lp = lp_v, tn_baseline_lp = tn_lp_v, stringsAsFactors = FALSE)
)
write_csv(risk_tbl, file.path(TABLES_DIR, "embedding_prognostic_risk_scores.csv"))

# ---- Assemble predicted-survival table (long format) ---------------------------
make_surv_long <- function(patient_ids, cohort_name, surv_mat_emb, surv_mat_tn) {
  bind_rows(lapply(seq_along(TIME_HORIZONS_MONTHS), function(hi) {
    data.frame(
      patient_id      = patient_ids,
      cohort          = cohort_name,
      time_label      = TIME_HORIZONS_LABELS[hi],
      time_months     = TIME_HORIZONS_MONTHS[hi],
      embedding_surv  = surv_mat_emb[hi, ],
      tn_baseline_surv = surv_mat_tn[hi, ],
      stringsAsFactors = FALSE
    )
  }))
}
surv_long <- bind_rows(
  make_surv_long(df_t$patient_id, "Training", surv_t, tn_surv_t),
  make_surv_long(df_v$patient_id, "Internal validation", surv_v, tn_surv_v)
)
write_csv(surv_long, file.path(TABLES_DIR, "embedding_prognostic_survival_predictions.csv"))
cat("  Saved risk-score and survival-prediction tables.\n")

saveRDS(cv_fit, file.path(MODELS_DIR, "embedding_ridge_cox_cv.rds"))
saveRDS(tn_fit, file.path(MODELS_DIR, "tn_baseline_coxph.rds"))

log_sample_size("01_embedding_cox_model", "embedding_ridge_cox", nrow(df), nrow(df_t))
log_sample_size("01_embedding_cox_model", "tn_baseline", nrow(df), nrow(df_t))

cat("=== 01 complete ===\n\n")
