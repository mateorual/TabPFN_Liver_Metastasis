# scripts/05_multivariable_cox_models.R
# Multivariable Cox proportional hazards models — TRAINING COHORT ONLY.
#
# Models:
#   (1) nomogram_figure   — paper Figure 5 variables (PRIMARY)
#   (2) text_screened     — paper text/Table 4 screened variables (AUDIT)
#   (3) cox_aic_selected  — AIC step-selected variables (AUDIT)
#   (4) rf_rfe_selected   — RF/RFE-selected variables (AUDIT)
#   (5) pipeline_selected — union(cox_aic, rf_rfe) with global p < 0.05 filter
#   (6) tn_baseline       — AJCC T/N comparator

cat("=== 05_multivariable_cox_models.R ===\n")

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

build_cox <- function(features, data, label) {
  present <- intersect(features, names(data))
  zero_v  <- sapply(present, function(v) {
    if (is.factor(data[[v]])) length(unique(droplevels(data[[v]]))) <= 1
    else FALSE
  })
  if (any(zero_v)) {
    cat("  [note] Excluding zero-variance vars from", label, ":",
        paste(present[zero_v], collapse = ", "), "\n")
    present <- present[!zero_v]
  }
  if (length(present) == 0) {
    cat("  [warn] No features for model", label, "\n"); return(NULL)
  }
  f <- as.formula(paste0("Surv(", TIME_VAR, ", ", EVENT_VAR, ") ~ ",
                         paste(present, collapse = " + ")))
  warn_log <- character(0)
  fit <- tryCatch(
    withCallingHandlers(
      coxph(f, data = data, ties = "efron"),
      warning = function(w) { warn_log <<- c(warn_log, conditionMessage(w));
                               invokeRestart("muffleWarning") }
    ),
    error = function(e) { cat("  [warn] coxph failed for", label, ":",
                               conditionMessage(e), "\n"); NULL }
  )
  if (!is.null(fit) && length(warn_log) > 0) {
    cat("  [warn]", label, ":", paste(warn_log, collapse = "; "), "\n")
  }
  fit
}

save_cox_results <- function(fit, label, features, csv_name) {
  if (is.null(fit)) return(invisible(NULL))
  tidy_df <- tryCatch(
    broom::tidy(fit, exponentiate = TRUE, conf.int = TRUE) |>
      rename(HR = estimate, ci_lower_95 = conf.low, ci_upper_95 = conf.high,
             p_value = p.value, std_error = std.error) |>
      mutate(model = label, n_obs = fit$n, n_events = fit$nevent,
             aic = AIC(fit)),
    error = function(e) NULL
  )
  if (!is.null(tidy_df)) {
    write_csv(tidy_df, file.path(TABLES_DIR, csv_name))
    cat("  Saved", csv_name, "| N =", fit$n, "| Events =", fit$nevent,
        "| C-index =",
        round(as.numeric(tryCatch(summary(fit)$concordance["C"],
                                  error = function(e) NA_real_)), 3), "\n")
  }
  invisible(fit)
}

lrt_global_p_cox <- function(v, fit, data) {
  f_red <- tryCatch(update(formula(fit), paste(". ~ . -", v)), error = function(e) NULL)
  if (is.null(f_red)) return(NA_real_)
  tryCatch({
    m_red <- coxph(f_red, data = data)
    lrt   <- anova(m_red, fit, test = "LRT")
    as.numeric(lrt[["Pr(>|Chi|)"]][2])
  }, error = function(e) NA_real_)
}

# ---- (1) nomogram_figure — PRIMARY --------------------------------------------
cat("\n  [1] nomogram_figure (primary)\n")
model_nom <- build_cox(FEATURES_NOMOGRAM_FIGURE, df_t, "nomogram_figure")
save_cox_results(model_nom, "nomogram_figure", FEATURES_NOMOGRAM_FIGURE,
                 "multivariable_cox_results_nomogram_figure.csv")
if (!is.null(model_nom)) {
  saveRDS(model_nom, file.path(MODELS_DIR, "cox_model_nomogram_figure.rds"))
  log_sample_size("05_cox", "nomogram_figure", nrow(df), model_nom$n)
}

# ---- (2) text_screened — AUDIT/SENSITIVITY ------------------------------------
cat("\n  [2] text_screened (audit)\n")
model_ts <- build_cox(FEATURES_TEXT_SCREENED, df_t, "text_screened")
save_cox_results(model_ts, "text_screened", FEATURES_TEXT_SCREENED,
                 "multivariable_cox_results_text_screened.csv")
if (!is.null(model_ts)) {
  saveRDS(model_ts, file.path(MODELS_DIR, "cox_model_text_screened.rds"))
  log_sample_size("05_cox", "text_screened", nrow(df), model_ts$n)
}

# ---- (3) cox_aic_selected — AUDIT ---------------------------------------------
cat("\n  [3] cox_aic_selected (audit)\n")
aic_vars <- tryCatch(readRDS(file.path(MODELS_DIR, "cox_aic_selected_vars.rds")),
                     error = function(e) { cat("  [skip] cox_aic_selected_vars.rds not found.\n"); character(0) })
model_aic <- NULL
if (length(aic_vars) > 0) {
  model_aic <- build_cox(aic_vars, df_t, "cox_aic_selected")
  save_cox_results(model_aic, "cox_aic_selected", aic_vars,
                   "multivariable_cox_results_cox_aic_selected.csv")
  if (!is.null(model_aic)) {
    saveRDS(model_aic, file.path(MODELS_DIR, "cox_model_cox_aic_selected.rds"))
    log_sample_size("05_cox", "cox_aic_selected", nrow(df), model_aic$n)
  }
}

# ---- (4) rf_rfe_selected — AUDIT ----------------------------------------------
cat("\n  [4] rf_rfe_selected (audit)\n")
rfe_vars <- tryCatch(readRDS(file.path(MODELS_DIR, "rf_rfe_selected_vars.rds")),
                     error = function(e) { cat("  [skip] rf_rfe_selected_vars.rds not found.\n"); character(0) })
model_rfe <- NULL
if (length(rfe_vars) > 0) {
  model_rfe <- build_cox(rfe_vars, df_t, "rf_rfe_selected")
  save_cox_results(model_rfe, "rf_rfe_selected", rfe_vars,
                   "multivariable_cox_results_rf_rfe_selected.csv")
  if (!is.null(model_rfe)) {
    saveRDS(model_rfe, file.path(MODELS_DIR, "cox_model_rf_rfe_selected.rds"))
    log_sample_size("05_cox", "rf_rfe_selected", nrow(df), model_rfe$n)
  }
}

# ---- (5) pipeline_selected — union(cox_aic, rf_rfe) + p < 0.05 filter --------
cat("\n  [5] pipeline_selected\n")
ps_raw_vars <- union(aic_vars, rfe_vars)
if (length(ps_raw_vars) == 0) {
  cat("  [note] Empty union; using text_screened as pipeline_selected fallback.\n")
  ps_raw_vars <- FEATURES_TEXT_SCREENED
}
ps_raw_vars <- intersect(ps_raw_vars, names(df_t))
cat("  Pipeline raw (union,", length(ps_raw_vars), "):",
    paste(ps_raw_vars, collapse = ", "), "\n")

model_ps_raw <- build_cox(ps_raw_vars, df_t, "pipeline_raw")

if (!is.null(model_ps_raw)) {
  gp_ps <- map_dfr(intersect(ps_raw_vars, names(df_t)), function(v) {
    p <- lrt_global_p_cox(v, model_ps_raw, df_t)
    data.frame(variable = v, global_lrt_p = round(p, 5), stringsAsFactors = FALSE)
  })
  ps_final_vars <- gp_ps |>
    filter(!is.na(global_lrt_p), global_lrt_p < 0.05) |>
    pull(variable)
  if (length(ps_final_vars) < 2) {
    cat("  [note] < 2 vars with p < 0.05; keeping full pipeline raw set.\n")
    ps_final_vars <- ps_raw_vars
  }
} else {
  ps_final_vars <- ps_raw_vars
}
cat("  Pipeline final (", length(ps_final_vars), "):",
    paste(ps_final_vars, collapse = ", "), "\n")

model_ps <- build_cox(ps_final_vars, df_t, "pipeline_selected")
save_cox_results(model_ps, "pipeline_selected", ps_final_vars,
                 "multivariable_cox_results_pipeline_selected.csv")
if (!is.null(model_ps)) {
  saveRDS(model_ps,       file.path(MODELS_DIR, "cox_model_pipeline_selected.rds"))
  saveRDS(ps_final_vars,  file.path(MODELS_DIR, "pipeline_selected_features.rds"))
  write_csv(data.frame(variable = ps_final_vars),
            file.path(TABLES_DIR, "pipeline_selected_features_prognostic.csv"))
  log_sample_size("05_cox", "pipeline_selected", nrow(df), model_ps$n)
}

# ---- (6) tn_baseline -----------------------------------------------------------
cat("\n  [6] tn_baseline\n")
model_tn <- build_cox(FEATURES_TN_BASELINE, df_t, "tn_baseline")
save_cox_results(model_tn, "tn_baseline", FEATURES_TN_BASELINE,
                 "multivariable_cox_results_tn_baseline.csv")
if (!is.null(model_tn)) {
  saveRDS(model_tn, file.path(MODELS_DIR, "cox_model_tn_baseline.rds"))
  log_sample_size("05_cox", "tn_baseline", nrow(df), model_tn$n)
}

# ---- Model summary table -------------------------------------------------------
model_list_named <- list(
  nomogram_figure   = model_nom,
  text_screened     = model_ts,
  cox_aic_selected  = model_aic,
  rf_rfe_selected   = model_rfe,
  pipeline_selected = model_ps,
  tn_baseline       = model_tn
)

summary_rows <- map_dfr(names(model_list_named), function(nm) {
  fit <- model_list_named[[nm]]
  if (is.null(fit)) return(data.frame())
  conc   <- tryCatch(as.numeric(summary(fit)$concordance["C"]),
                     error = function(e) NA_real_)
  se_c   <- tryCatch(as.numeric(summary(fit)$concordance["se(C)"]),
                     error = function(e) NA_real_)
  if (length(conc) != 1L) conc  <- NA_real_
  if (length(se_c) != 1L) se_c  <- NA_real_
  g_p    <- tryCatch(as.numeric(fit$logtest["pvalue"]), error = function(e) NA_real_)
  if (length(g_p) != 1L) g_p <- NA_real_
  data.frame(
    model           = nm,
    n_obs           = fit$n,
    n_events        = fit$nevent,
    concordance     = round(conc, 4),
    se_concordance  = round(se_c, 4),
    c_index_lower   = round(conc - 1.96 * se_c, 4),
    c_index_upper   = round(conc + 1.96 * se_c, 4),
    aic             = round(AIC(fit), 2),
    global_p_lrt    = round(g_p, 5),
    stringsAsFactors = FALSE
  )
})

write_csv(summary_rows, file.path(TABLES_DIR, "multivariable_cox_model_summary.csv"))
cat("\n  Model summary:\n"); print(summary_rows)

ph_rows <- map_dfr(names(model_list_named), function(nm) {
  fit <- model_list_named[[nm]]
  if (is.null(fit)) return(data.frame())
  tryCatch({
    ph_obj <- cox.zph(fit)
    as.data.frame(ph_obj$table) |>
      tibble::rownames_to_column("term") |>
      mutate(model = nm)
  }, error = function(e) data.frame())
})
if (nrow(ph_rows) > 0) {
  write_csv(ph_rows, file.path(TABLES_DIR, "cox_ph_assumption_multivariable.csv"))
  cat("  Saved cox_ph_assumption_multivariable.csv\n")
}

# ---- Feature sets audit table ---------------------------------------------------
make_row <- function(nm, vars) {
  data.frame(model = nm, n_vars = length(vars),
             variables = paste(vars, collapse = ", "),
             role = dplyr::case_when(
               nm == "nomogram_figure"              ~ "Primary (paper Figure 5)",
               nm == "paper_text_screened_audit"    ~ "Audit (paper text screened; NOT primary — paper internally inconsistent)",
               nm == "text_screened"                ~ "Audit (paper text screened; NOT primary — paper internally inconsistent)",
               nm == "cox_aic_selected"             ~ "Audit (Cox/AIC selection)",
               nm == "rf_rfe_selected"              ~ "Audit (RF/RFE selection)",
               nm == "pipeline_selected"            ~ "Data-driven (union + p<0.05)",
               nm == "tn_baseline"                  ~ "Comparator (AJCC T/N baseline)",
               TRUE                                 ~ "Other"
             ),
             stringsAsFactors = FALSE)
}
feat_audit <- bind_rows(
  make_row("nomogram_figure",           FEATURES_NOMOGRAM_FIGURE),
  make_row("paper_text_screened_audit", FEATURES_TEXT_SCREENED),
  make_row("cox_aic_selected",          aic_vars),
  make_row("rf_rfe_selected",           rfe_vars),
  make_row("pipeline_selected",         ps_final_vars),
  make_row("tn_baseline",               FEATURES_TN_BASELINE)
)
write_csv(feat_audit, file.path(TABLES_DIR, "model_feature_sets_prognostic.csv"))
write_csv(feat_audit, file.path(TABLES_DIR, "pipeline_feature_selection_audit_prognostic.csv"))
cat("  Saved model_feature_sets_prognostic.csv\n")

# ---- Paper variable inconsistency audit table ----------------------------------
inconsistency_tbl <- bind_rows(
  data.frame(
    source    = "Figure 5 / abstract (primary model)",
    n_vars    = length(FEATURES_NOMOGRAM_FIGURE),
    variables = paste(FEATURES_NOMOGRAM_FIGURE, collapse = ", "),
    note      = "Final nomogram figure and abstract. Used as FEATURES_NOMOGRAM_FIGURE.",
    stringsAsFactors = FALSE
  ),
  data.frame(
    source    = "Results text / Table 4 screened",
    n_vars    = length(FEATURES_TEXT_SCREENED),
    variables = paste(FEATURES_TEXT_SCREENED, collapse = ", "),
    note      = "Variables mentioned in text as screened (grade, N stage, surgery, chemo, tumor size, bone mets). Inconsistent with Figure 5.",
    stringsAsFactors = FALSE
  ),
  data.frame(
    source    = "Data-driven pipeline-selected",
    n_vars    = length(ps_final_vars),
    variables = paste(ps_final_vars, collapse = ", "),
    note      = "union(Cox/AIC, RF/RFE) filtered by multivariable global p < 0.05.",
    stringsAsFactors = FALSE
  )
)
write_csv(inconsistency_tbl,
          file.path(TABLES_DIR, "paper_prognostic_variable_inconsistency_audit.csv"))
cat("  Saved paper_prognostic_variable_inconsistency_audit.csv\n")

cat("=== 05 complete ===\n\n")
