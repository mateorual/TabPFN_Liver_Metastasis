# scripts/02_univariate_cox.R
# Univariate Cox proportional hazards models on the TRAINING cohort only.
# Saves term-level results, variable-level summary with global LRT p-values,
# and PH assumption diagnostics via cox.zph().

cat("=== 02_univariate_cox.R ===\n")

source("config/config.R")

for (pkg in c("survival", "broom", "dplyr", "readr", "purrr")) {
  if (!requireNamespace(pkg, quietly = TRUE))
    stop("Install '", pkg, "' with: install.packages('", pkg, "')")
}
suppressPackageStartupMessages({
  library(survival); library(broom); library(dplyr); library(readr); library(purrr)
})

df   <- readRDS(file.path(MODELS_DIR, "prognostic_data_clean.rds"))
df_t <- df[df$cohort == "Training", ]
cat("  Training N =", nrow(df_t), "| Events =", sum(df_t[[EVENT_VAR]]), "\n")

fit_uni_cox <- function(predictor, data) {
  f_str <- paste0("Surv(", TIME_VAR, ", ", EVENT_VAR, ") ~ ", predictor)
  f     <- as.formula(f_str)
  warn_msgs <- character(0)
  fit <- tryCatch(
    withCallingHandlers(
      coxph(f, data = data, ties = "efron"),
      warning = function(w) { warn_msgs <<- c(warn_msgs, conditionMessage(w));
                               invokeRestart("muffleWarning") }
    ),
    error = function(e) { cat("  [warn] coxph failed for", predictor, ":",
                               conditionMessage(e), "\n"); NULL }
  )
  if (is.null(fit)) {
    return(list(tidy = NULL, var_summary = NULL, ph = NULL, model = NULL))
  }
  n_obs    <- fit$n
  n_events <- fit$nevent

  tidy_df  <- tryCatch(
    broom::tidy(fit, exponentiate = TRUE, conf.int = TRUE) |>
      filter(term != "(Intercept)") |>
      mutate(
        predictor   = predictor,
        n_obs       = n_obs,
        n_events    = n_events,
        warning_msg = if (length(warn_msgs) > 0) paste(warn_msgs, collapse = "; ") else NA_character_
      ) |>
      rename(HR = estimate, ci_lower_95 = conf.low, ci_upper_95 = conf.high,
             p_value = p.value, std_error = std.error),
    error = function(e) NULL
  )

  null_fit  <- coxph(as.formula(paste0("Surv(", TIME_VAR, ",", EVENT_VAR, ") ~ 1")),
                     data = data)
  global_p  <- tryCatch({
    lrt <- anova(null_fit, fit, test = "LRT")
    as.numeric(lrt[["Pr(>|Chi|)"]][2])
  }, error = function(e) NA_real_)

  var_summary <- data.frame(
    predictor     = predictor,
    n_obs         = n_obs,
    n_events      = n_events,
    n_terms       = if (!is.null(tidy_df)) nrow(tidy_df) else NA,
    global_lrt_p  = round(global_p, 5),
    concordance   = tryCatch(summary(fit)$concordance["C"], error = function(e) NA_real_),
    warning_msg   = if (length(warn_msgs) > 0) paste(warn_msgs, collapse = "; ") else NA_character_,
    stringsAsFactors = FALSE
  )

  ph_df <- tryCatch({
    ph <- cox.zph(fit)
    as.data.frame(ph$table) |>
      tibble::rownames_to_column("term") |>
      mutate(predictor = predictor)
  }, error = function(e) NULL)

  list(tidy = tidy_df, var_summary = var_summary, ph = ph_df, model = fit)
}

cat("  Running univariate Cox models ...\n")
preds <- intersect(CANDIDATE_PREDICTORS, names(df_t))

all_results <- map(preds, ~ fit_uni_cox(.x, df_t))
names(all_results) <- preds

tidy_all <- bind_rows(map(all_results, "tidy"))
write_csv(tidy_all, file.path(TABLES_DIR, "univariate_cox_results.csv"))
cat("  Saved univariate_cox_results.csv\n")

var_summary_all <- bind_rows(map(all_results, "var_summary"))
write_csv(var_summary_all,
          file.path(TABLES_DIR, "univariate_cox_variable_level_summary.csv"))
cat("  Saved univariate_cox_variable_level_summary.csv\n")
cat("  Variable-level summary (top 10 by global p):\n")
print(var_summary_all |> arrange(global_lrt_p) |>
        select(predictor, n_events, global_lrt_p, concordance) |> head(10))

ph_all <- bind_rows(map(all_results, "ph"))
if (nrow(ph_all) > 0) {
  write_csv(ph_all, file.path(TABLES_DIR, "univariate_cox_ph_assumption.csv"))
  cat("  Saved univariate_cox_ph_assumption.csv\n")
}

saveRDS(map(all_results, "model"),
        file.path(MODELS_DIR, "univariate_cox_models.rds"))
cat("  Saved univariate_cox_models.rds\n")

log_sample_size("02_uni_cox", "all_predictors", nrow(df), nrow(df_t))

cat("=== 02 complete ===\n\n")
