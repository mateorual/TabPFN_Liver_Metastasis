# scripts/01_embedding_model.R
# Ridge-penalized logistic regression on the 512-dim TabPFN embedding matrix,
# predicting mets_liver. Ridge (not LASSO) because embeddings are dense,
# correlated features with no natural sparsity structure.
#
# Evaluation uses OUT-OF-FOLD predictions from cv.glmnet's internal 10-fold CV
# (keep=TRUE), not in-sample fitted values: with p=512 features on N=2672
# rows, in-sample predictions would be materially optimistic. This mirrors
# the leakage-safety spirit of the embeddings themselves (which are already
# OOF with respect to the TabPFN embedding step).
#
# AJCC T/N baseline is a plain 2-variable glm fit in-sample (low-dimensional,
# no OOF concern), matching how the clinical pipeline treats its baseline.

cat("=== 01_embedding_model.R ===\n")

source("config/config.R")

for (pkg in c("glmnet", "dplyr", "readr")) {
  if (!requireNamespace(pkg, quietly = TRUE))
    stop("Install '", pkg, "' with: install.packages('", pkg, "')")
}
suppressPackageStartupMessages({ library(glmnet); library(dplyr); library(readr) })

df      <- readRDS(file.path(MODELS_DIR, "embedding_diagnostic_data_clean.rds"))
emb_cols <- readRDS(file.path(MODELS_DIR, "embedding_columns.rds"))

X <- as.matrix(df[, emb_cols])
y <- df[[OUTCOME]]
cat("  Embedding matrix:", nrow(X), "x", ncol(X), "\n")

# ---- Ridge logistic regression with out-of-fold predictions -----------------
set.seed(SEED)
cv_fit <- cv.glmnet(
  X, y, family = "binomial", alpha = GLMNET_ALPHA, nfolds = GLMNET_NFOLDS,
  type.measure = "deviance", keep = TRUE
)

lambda_idx <- which(cv_fit$lambda == cv_fit$lambda.min)
oof_link   <- cv_fit$fit.preval[, lambda_idx]
oof_prob   <- plogis(oof_link)

cat("  Ridge logistic: lambda.min =", round(cv_fit$lambda.min, 5), "\n")
cat("  OOF predictions range: [", round(min(oof_prob), 4), ",", round(max(oof_prob), 4), "]\n")

pred_df <- data.frame(
  patient_id      = df$patient_id,
  mets_liver      = y,
  embedding_model = oof_prob,
  stringsAsFactors = FALSE
)

# ---- AJCC T/N baseline (in-sample, low-dimensional) --------------------------
tn_glm <- glm(
  as.formula(paste0(OUTCOME, " ~ ", paste(FEATURES_TNM_BASELINE, collapse = " + "))),
  data = df, family = binomial()
)
pred_df$tnm_baseline <- predict(tn_glm, type = "response")

write_csv(pred_df, file.path(TABLES_DIR, "embedding_diagnostic_predictions.csv"))
cat("  Saved embedding_diagnostic_predictions.csv\n")

saveRDS(cv_fit, file.path(MODELS_DIR, "embedding_ridge_logistic_cv.rds"))
saveRDS(tn_glm, file.path(MODELS_DIR, "tnm_baseline_glm.rds"))

log_sample_size("01_embedding_model", "embedding_ridge_logistic", nrow(df), nrow(df))
log_sample_size("01_embedding_model", "tnm_baseline", nrow(df), nrow(df))

cat("=== 01 complete ===\n\n")
