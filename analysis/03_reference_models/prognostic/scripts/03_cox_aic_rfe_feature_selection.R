# scripts/03_cox_aic_rfe_feature_selection.R
# Cox proportional hazards model with AIC-based stepwise feature selection.
# Training cohort only. Uses MASS::stepAIC() — backward is primary (closest to
# the paper's RFE+AIC step), bidirectional ("both") is saved as a sensitivity
# check. The paper does not specify the exact AIC-RFE implementation.

cat("=== 03_cox_aic_rfe_feature_selection.R ===\n")

source("config/config.R")

for (pkg in c("survival", "broom", "dplyr", "readr")) {
  if (!requireNamespace(pkg, quietly = TRUE))
    stop("Install '", pkg, "' with: install.packages('", pkg, "')")
}
suppressPackageStartupMessages({
  library(survival); library(broom); library(dplyr); library(readr)
})

df   <- readRDS(file.path(MODELS_DIR, "prognostic_data_clean.rds"))
df_t <- df[df$cohort == "Training", ]
cat("  Training N =", nrow(df_t), "| Events =", sum(df_t[[EVENT_VAR]]), "\n")

preds_avail <- intersect(CANDIDATE_PREDICTORS, names(df_t))
zero_var    <- sapply(preds_avail, function(v) {
  if (is.factor(df_t[[v]])) length(unique(droplevels(df_t[[v]]))) <= 1
  else isTRUE(var(as.numeric(df_t[[v]]), na.rm = TRUE) == 0)
})
if (any(zero_var)) {
  cat("  [note] Excluding zero-variance predictors from AIC selection:",
      paste(preds_avail[zero_var], collapse = ", "), "\n")
  preds_avail <- preds_avail[!zero_var]
}
cat("  Estimable predictors (", length(preds_avail), "):",
    paste(preds_avail, collapse = ", "), "\n")

df_cc <- df_t |>
  select(all_of(c(preds_avail, TIME_VAR, EVENT_VAR))) |>
  na.omit()
cat("  Complete-case N =", nrow(df_cc), "\n")

full_formula <- as.formula(
  paste0("Surv(", TIME_VAR, ", ", EVENT_VAR, ") ~ ",
         paste(preds_avail, collapse = " + "))
)

warn_log <- character(0)
cox_full <- tryCatch(
  withCallingHandlers(
    coxph(full_formula, data = df_cc, ties = "efron"),
    warning = function(w) {
      warn_log <<- c(warn_log, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  ),
  error = function(e) {
    cat("  [warn] Full Cox model failed:", conditionMessage(e), "\n"); NULL
  }
)
if (is.null(cox_full)) {
  stop("Full Cox model could not be fitted. Check the data.")
}
cat("  Full Cox: N =", cox_full$n, "| Events =", cox_full$nevent,
    "| C-index =",
    round(as.numeric(summary(cox_full)$concordance["C"]), 3), "\n")
saveRDS(cox_full, file.path(MODELS_DIR, "cox_aic_full_model.rds"))

use_stepAIC <- requireNamespace("MASS", quietly = TRUE)

save_aic_path <- function(fit, stem) {
  path_tbl <- tryCatch({
    tr <- fit$anova
    if (!is.null(tr)) as.data.frame(tr) else
      data.frame(note = "step trace not available")
  }, error = function(e) data.frame(note = "step trace not available"))
  write_csv(path_tbl, file.path(TABLES_DIR, paste0(stem, ".csv")))
  cat("  Saved", stem, ".csv\n")
}

cox_aic    <- NULL
aic_method <- NULL

if (use_stepAIC) {
  cat("  Running MASS::stepAIC (direction = 'backward') ...\n")
  cox_aic_back <- tryCatch(
    suppressWarnings(MASS::stepAIC(cox_full, direction = "backward", trace = FALSE)),
    error = function(e) { cat("  [warn] backward stepAIC:", conditionMessage(e), "\n"); NULL }
  )
  save_aic_path(if (!is.null(cox_aic_back)) cox_aic_back else cox_full,
                "cox_aic_selection_path_backward")
  cox_aic    <- cox_aic_back
  aic_method <- "MASS::stepAIC (direction = backward)"

  cat("  Running MASS::stepAIC (direction = 'both', sensitivity) ...\n")
  cox_aic_both <- tryCatch(
    suppressWarnings(MASS::stepAIC(cox_full, direction = "both", trace = FALSE)),
    error = function(e) { cat("  [warn] both stepAIC:", conditionMessage(e), "\n"); NULL }
  )
  save_aic_path(if (!is.null(cox_aic_both)) cox_aic_both else cox_full,
                "cox_aic_selection_path_both_sensitivity")
  if (!is.null(cox_aic_both)) {
    saveRDS(cox_aic_both, file.path(MODELS_DIR, "cox_aic_selected_model_both.rds"))
  }

  if (is.null(cox_aic)) cox_aic <- cox_aic_both   # fall back to 'both' if backward failed

} else {
  cat("  [warn] MASS not available. Falling back to univariate p < 0.1 filter.\n")
  aic_method <- "Fallback: univariate global p < 0.1"
}

log_method_note(
  "Cox AIC feature selection",
  "Approximation of paper RFE+AIC (underspecified)",
  paste0("Primary: MASS::stepAIC backward. Sensitivity: bidirectional. ",
         "Paper does not specify exact AIC-RFE implementation.")
)

if (is.null(cox_aic)) {
  uni_path <- file.path(TABLES_DIR, "univariate_cox_variable_level_summary.csv")
  if (file.exists(uni_path)) {
    uni_tbl <- read_csv(uni_path, show_col_types = FALSE)
    selected_fallback <- uni_tbl |>
      filter(!is.na(global_lrt_p), global_lrt_p < 0.1) |>
      pull(predictor)
    selected_fallback <- intersect(selected_fallback, preds_avail)
    cat("  Fallback selected (p < 0.1):", paste(selected_fallback, collapse = ", "), "\n")
  } else {
    cat("  [warn] univariate_cox_variable_level_summary.csv not found.",
        "Using all predictors.\n")
    selected_fallback <- preds_avail
  }

  if (length(selected_fallback) > 0) {
    f_fb <- as.formula(
      paste0("Surv(", TIME_VAR, ", ", EVENT_VAR, ") ~ ",
             paste(selected_fallback, collapse = " + "))
    )
    cox_aic <- tryCatch(coxph(f_fb, data = df_cc),
                        error = function(e) NULL)
  }
  aic_method <- paste0(aic_method, " (stepAIC failed)")
}

if (is.null(cox_aic)) {
  cat("  [warn] AIC selection failed entirely. Saving full model as AIC-selected.\n")
  cox_aic    <- cox_full
  aic_method <- "Full model (AIC selection failed)"
}

# ---- Extract selected variables --------------------------------------------------
aic_terms    <- names(coef(cox_aic))
aic_terms    <- aic_terms[!is.na(coef(cox_aic))]   # drop singular terms
aic_vars_raw <- gsub("([A-Za-z_0-9]+).*", "\\1", aic_terms)
aic_vars_raw <- unique(aic_vars_raw)
selected_vars <- intersect(preds_avail, aic_vars_raw)
for (v in preds_avail) {
  if (any(grepl(paste0("^", v), aic_terms))) selected_vars <- union(selected_vars, v)
}
selected_vars <- intersect(selected_vars, preds_avail)

cat("  AIC-selected variables (", length(selected_vars), "):",
    paste(selected_vars, collapse = ", "), "\n")
cat("  AIC-selected model: N =", cox_aic$n,
    "| C-index =",
    round(as.numeric(summary(cox_aic)$concordance["C"]), 3), "\n")

aic_path_tbl <- tryCatch({
  step_trace <- cox_aic$anova
  if (!is.null(step_trace)) {
    as.data.frame(step_trace) |>
      mutate(model = "cox_aic_selected") |>
      select(model, everything())
  } else {
    data.frame(note = "stepAIC trace not available", stringsAsFactors = FALSE)
  }
}, error = function(e) {
  data.frame(note = "AIC path not available", stringsAsFactors = FALSE)
})
write_csv(aic_path_tbl, file.path(TABLES_DIR, "cox_aic_selection_path.csv"))
cat("  Saved cox_aic_selection_path.csv\n")

tidy_aic <- tryCatch(
  broom::tidy(cox_aic, exponentiate = TRUE, conf.int = TRUE) |>
    rename(HR = estimate, ci_lower_95 = conf.low, ci_upper_95 = conf.high,
           p_value = p.value, std_error = std.error) |>
    mutate(model = "cox_aic_selected", n_obs = cox_aic$n,
           n_events = cox_aic$nevent),
  error = function(e) NULL
)
if (!is.null(tidy_aic)) {
  write_csv(tidy_aic, file.path(TABLES_DIR, "cox_aic_selected_model_results.csv"))
  cat("  Saved cox_aic_selected_model_results.csv\n")
}

write_csv(data.frame(variable = selected_vars, stringsAsFactors = FALSE),
          file.path(TABLES_DIR, "cox_aic_selected_features.csv"))
saveRDS(cox_aic,        file.path(MODELS_DIR, "cox_aic_selected_model.rds"))
saveRDS(selected_vars,  file.path(MODELS_DIR, "cox_aic_selected_vars.rds"))
cat("  Saved cox_aic_selected_model.rds / cox_aic_selected_vars.rds\n")

log_method_note(
  "Cox AIC feature selection",
  aic_method,
  paste0("Selected ", length(selected_vars), " vars: ",
         paste(selected_vars, collapse = ", "))
)

log_sample_size("03_cox_aic", "cox_aic_selected", nrow(df), cox_aic$n)

cat("=== 03 complete ===\n\n")
