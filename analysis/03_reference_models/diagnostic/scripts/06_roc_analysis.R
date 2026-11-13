# scripts/06_roc_analysis.R
# ROC/AUC table for all diagnostic models (paper_text, nomogram_figure,
# pipeline_selected, tnm_baseline), evaluated on a common complete-case set.

cat("=== 06_roc_analysis.R ===\n")

source("config/config.R")
source("scripts/helpers_diagnostic.R")

for (pkg in c("pROC", "dplyr", "readr")) {
  if (!requireNamespace(pkg, quietly = TRUE))
    stop("Install '", pkg, "' with: install.packages('", pkg, "')")
}
suppressPackageStartupMessages({
  library(pROC); library(dplyr); library(readr)
})

set.seed(SEED)

df <- readRDS(file.path(MODELS_DIR, "diagnostic_data_clean.rds"))

load_glm <- function(name) {
  p <- file.path(MODELS_DIR, paste0("logistic_model_", name, ".rds"))
  if (!file.exists(p)) return(NULL)
  readRDS(p)
}

ps_feats <- tryCatch(
  readRDS(file.path(MODELS_DIR, "pipeline_selected_features.rds")),
  error = function(e) character(0)
)

models_list <- Filter(Negate(is.null), list(
  paper_text        = load_glm("paper_text"),
  nomogram_figure   = load_glm("nomogram_figure"),
  pipeline_selected = load_glm("pipeline_selected")
))

# AJCC T/N baseline: the comparator fitted and saved by script 04.
tnm_m <- load_glm("tnm_baseline")
if (is.null(tnm_m)) stop("logistic_model_tnm_baseline.rds not found. Run script 04 first.")
models_list[["tnm_baseline"]] <- tnm_m
cat("  Models loaded:", paste(names(models_list), collapse = ", "), "\n")

labels <- c(
  paper_text        = model_display_label("Paper Text", FEATURES_PAPER_TEXT, max_chars = 120),
  nomogram_figure   = model_display_label("Nomogram Figure", FEATURES_NOMOGRAM_FIGURE, max_chars = 120),
  pipeline_selected = model_display_label("Pipeline-selected",
                        if (length(ps_feats) > 0) ps_feats else character(0), max_chars = 120),
  tnm_baseline      = model_display_label("AJCC T/N baseline", FEATURES_TNM_BASELINE, max_chars = 120)
)

get_probs <- function(model, data) {
  tryCatch(
    as.numeric(predict(model, newdata = data, type = "response", na.action = na.exclude)),
    error = function(e) NULL
  )
}

# ---- Common evaluation set: complete cases across all model variables -----
all_model_vars <- unique(c(
  FEATURES_PAPER_TEXT, FEATURES_NOMOGRAM_FIGURE,
  if (length(ps_feats) > 0) ps_feats else character(0),
  FEATURES_TNM_BASELINE, OUTCOME
))
all_model_vars <- intersect(all_model_vars, names(df))
df_common      <- df[complete.cases(df[, all_model_vars]), ]
n_common       <- nrow(df_common)
cat("  Common evaluation set: N =", n_common, "\n")

auc_rows <- list()
for (nm in names(models_list)) {
  probs <- get_probs(models_list[[nm]], df_common)
  if (is.null(probs)) next

  valid  <- !is.na(probs)
  y_obs  <- df_common[[OUTCOME]][valid]
  p_hat  <- probs[valid]
  n_used <- sum(valid)

  roc_obj <- tryCatch(pROC::roc(y_obs, p_hat, direction = "<", quiet = TRUE),
                       error = function(e) NULL)
  if (is.null(roc_obj)) next

  ci_obj <- tryCatch(
    pROC::ci.auc(roc_obj, method = "bootstrap", boot.n = 1000),
    error = function(e) pROC::ci.auc(roc_obj, method = "delong")
  )

  auc_rows[[nm]] <- data.frame(
    model       = nm,
    label       = labels[nm],
    auc         = round(as.numeric(pROC::auc(roc_obj)), 4),
    ci_lower_95 = round(as.numeric(ci_obj[1]), 4),
    ci_upper_95 = round(as.numeric(ci_obj[3]), 4),
    n           = n_used,
    eval_set    = "common_all_models",
    stringsAsFactors = FALSE
  )
  cat("  ", nm, ": AUC =", round(as.numeric(pROC::auc(roc_obj)), 3),
      "(", round(as.numeric(ci_obj[1]), 3), "-", round(as.numeric(ci_obj[3]), 3), ")\n")
  log_sample_size("06_roc", nm, nrow(df), n_used)
}

auc_df <- bind_rows(auc_rows)
write_csv(auc_df, file.path(TABLES_DIR, "roc_auc_diagnostic.csv"))

write_csv(
  data.frame(figure = "roc_auc_diagnostic", eval_set = "common_all_models",
             n = n_common, vars_included = paste(setdiff(all_model_vars, OUTCOME), collapse = "; "),
             stringsAsFactors = FALSE),
  file.path(TABLES_DIR, "roc_evaluation_sample_sizes.csv")
)

cat("=== 06 complete ===\n\n")
