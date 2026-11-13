# scripts/04_logistic_regression_models.R
# Univariate and multivariable logistic regression for liver metastasis.
#
# Models: (A) paper_text, (B) nomogram_figure — both fixed feature sets from the
# paper. (C) pipeline_selected — derived from LASSO lambda.1se shortlist ->
# multivariable LR, keeping variables with global LRT p < 0.05.

cat("=== 04_logistic_regression_models.R ===\n")

source("config/config.R")

for (pkg in c("dplyr", "broom", "readr", "purrr")) {
  if (!requireNamespace(pkg, quietly = TRUE))
    stop("Install '", pkg, "' with: install.packages('", pkg, "')")
}

suppressPackageStartupMessages({
  library(dplyr); library(broom); library(readr); library(purrr)
})

df <- readRDS(file.path(MODELS_DIR, "diagnostic_data_clean.rds"))

fit_glm <- function(formula_str, data, label = "") {
  f <- as.formula(formula_str)
  d <- model.frame(f, data = data, na.action = na.omit)
  n_used <- nrow(d)
  fit <- tryCatch(
    glm(f, data = data, family = binomial(link = "logit"), na.action = na.omit),
    error = function(e) {
      cat("  [warn] glm failed:", conditionMessage(e), "\n"); NULL
    }
  )
  if (is.null(fit)) return(list(tidy = NULL, model = NULL, n = n_used))
  tidy_df <- broom::tidy(fit, exponentiate = TRUE, conf.int = TRUE) |>
    filter(term != "(Intercept)") |>
    mutate(
      n_used    = n_used,
      aic       = AIC(fit),
      bic       = BIC(fit),
      model     = label,
      converged = fit$converged
    ) |>
    rename(
      odds_ratio  = estimate,
      ci_lower_95 = conf.low,
      ci_upper_95 = conf.high,
      p_value     = p.value,
      std_error   = std.error
    )
  list(tidy = tidy_df, model = fit, n = n_used)
}

build_formula <- function(predictors, outcome, data) {
  present   <- intersect(predictors, names(data))
  cat_vars  <- intersect(present, CATEGORICAL_VARS)
  other_vars <- setdiff(present, cat_vars)
  terms <- c(other_vars, if (length(cat_vars) > 0) cat_vars)
  paste0(outcome, " ~ ", paste(terms, collapse = " + "))
}

# Likelihood-ratio test p-value for one variable, comparing full vs reduced model
lrt_global_p <- function(variable, formula_full, data) {
  f_red <- tryCatch(
    update(formula_full, paste(". ~ . -", variable)),
    error = function(e) NULL
  )
  if (is.null(f_red)) return(NA_real_)
  tryCatch({
    m_full <- glm(formula_full, data = data, family = binomial(), na.action = na.omit)
    m_red  <- glm(f_red, data = data, family = binomial(), na.action = na.omit)
    lrt    <- anova(m_red, m_full, test = "LRT")
    lrt[2, "Pr(>Chi)"]
  }, error = function(e) NA_real_)
}

# ============================================================
# 1. Univariate logistic regression (all 17 candidate predictors)
# ============================================================
run_univariate <- function(predictor, df, outcome) {
  if (!predictor %in% names(df)) return(NULL)
  formula_str <- paste0(outcome, " ~ ", predictor)
  res <- fit_glm(formula_str, df, label = predictor)
  if (is.null(res$tidy)) return(NULL)
  res$tidy |> mutate(predictor_univar = predictor)
}

uni_results <- map_dfr(
  intersect(CANDIDATE_PREDICTORS, names(df)),
  ~ run_univariate(.x, df, OUTCOME)
)
write_csv(uni_results, file.path(TABLES_DIR, "univariate_logistic_results.csv"))
log_sample_size("04_logistic", "univariate_all", nrow(df), nrow(df))

null_model <- glm(as.formula(paste0(OUTCOME, " ~ 1")),
                  data = df, family = binomial(), na.action = na.omit)

uni_var_summary <- map_dfr(
  intersect(CANDIDATE_PREDICTORS, names(df)),
  function(v) {
    df_v   <- df[, c(v, OUTCOME)] |> na.omit()
    n_used <- nrow(df_v)
    f_full <- as.formula(paste0(OUTCOME, " ~ ", v))
    global_p <- tryCatch({
      m_full <- glm(f_full, data = df, family = binomial(), na.action = na.omit)
      m_null <- glm(as.formula(paste0(OUTCOME, " ~ 1")),
                    data = df, family = binomial(), na.action = na.omit)
      lrt <- anova(m_null, m_full, test = "LRT")
      lrt[2, "Pr(>Chi)"]
    }, error = function(e) NA_real_)
    data.frame(variable = v, n_used = n_used,
               global_p_value = round(global_p, 5),
               stringsAsFactors = FALSE)
  }
)
write_csv(uni_var_summary,
          file.path(TABLES_DIR, "univariate_logistic_variable_level_summary.csv"))

# ============================================================
# 2. Fixed paper-derived models
# ============================================================
fit_and_save <- function(features, label, rds_name, csv_name) {
  present_feats <- intersect(features, names(df))
  if (length(present_feats) == 0) {
    cat("  [warn] No features present for model:", label, "\n")
    return(invisible(NULL))
  }
  formula_str <- build_formula(present_feats, OUTCOME, df)
  cat("  Fitting", label, "...  Formula:", formula_str, "\n")
  res <- fit_glm(formula_str, df, label = label)
  if (!is.null(res$tidy)) {
    write_csv(res$tidy, file.path(TABLES_DIR, csv_name))
    cat("   AIC =", round(res$tidy$aic[1], 2),
        " BIC =", round(res$tidy$bic[1], 2),
        " N =", res$n, "\n")
  }
  if (!is.null(res$model))
    saveRDS(res$model, file.path(MODELS_DIR, rds_name))
  log_sample_size("04_logistic", label, nrow(df), res$n)
  invisible(res$model)
}

model_a <- fit_and_save(
  FEATURES_PAPER_TEXT, "paper_text",
  "logistic_model_paper_text.rds",
  "multivariable_logistic_results_paper_text.csv"
)

model_b <- fit_and_save(
  FEATURES_NOMOGRAM_FIGURE, "nomogram_figure",
  "logistic_model_nomogram_figure.rds",
  "multivariable_logistic_results_nomogram_figure.csv"
)

# ============================================================
# 3. Pipeline-selected model
#   A. LASSO lambda.1se shortlist
#   B. RF support table (audit only, not a hard filter)
#   C. Univariate LR (computed above)
#   D. Multivariable LR on the LASSO 1se shortlist
#   E. Final predictors: LASSO 1se AND multivariable global p < 0.05
# ============================================================
lasso_1se_path <- file.path(MODELS_DIR, "lasso_selected_vars_lambda_1se.rds")
lasso_min_path <- file.path(MODELS_DIR, "lasso_selected_vars_lambda_min.rds")

lasso_1se_vars <- tryCatch(readRDS(lasso_1se_path), error = function(e) {
  cat("  [warn] lasso_selected_vars_lambda_1se.rds not found. Run script 02 first.\n")
  character(0)
})
lasso_min_vars <- tryCatch(readRDS(lasso_min_path), error = function(e) character(0))

if (length(lasso_1se_vars) == 0) {
  cat("  [STOP] No LASSO lambda.1se variables found. Cannot build pipeline_selected model.\n")
  model_ps <- NULL
} else {

  rf_support_path <- file.path(TABLES_DIR, "rf_support_for_lasso_1se.csv")
  rf_support <- tryCatch(
    read_csv(rf_support_path, show_col_types = FALSE),
    error = function(e) {
      cat("  [warn] rf_support_for_lasso_1se.csv not found. Run script 03 first.\n")
      NULL
    }
  )

  # Primary candidate set for multivariable LR = LASSO lambda.1se shortlist.
  # RF support is an audit column, not a hard exclusion filter.
  pipeline_candidates <- intersect(lasso_1se_vars, names(df))

  formula_mv <- as.formula(build_formula(pipeline_candidates, OUTCOME, df))
  cat("  [pipeline] Multivariable formula:", paste(deparse(formula_mv), collapse = " "), "\n")

  res_mv <- fit_glm(paste(deparse(formula_mv), collapse = " "), df, label = "pipeline_mv")

  if (!is.null(res_mv$tidy)) {
    write_csv(res_mv$tidy,
              file.path(TABLES_DIR, "multivariable_logistic_results_pipeline_selected.csv"))
    log_sample_size("04_logistic", "pipeline_mv", nrow(df), res_mv$n)
  }

  mv_var_summary <- map_dfr(pipeline_candidates, function(v) {
    global_p <- lrt_global_p(v, formula_mv, df)
    data.frame(
      variable          = v,
      included_in_pipeline_multivariable = TRUE,
      multivariable_global_p = round(global_p, 5),
      stringsAsFactors  = FALSE
    )
  })
  write_csv(mv_var_summary,
            file.path(TABLES_DIR,
                      "multivariable_logistic_variable_level_summary_pipeline_selected.csv"))

  final_ps_vars <- mv_var_summary |>
    filter(!is.na(multivariable_global_p), multivariable_global_p < 0.05) |>
    pull(variable)

  cat("  [pipeline] Final selected (p<0.05):", paste(final_ps_vars, collapse = ", "), "\n")

  # ---- Comprehensive pipeline selection audit table --------------------------
  all_candidate_vars <- intersect(CANDIDATE_PREDICTORS, names(df))

  rf_imp_path <- file.path(TABLES_DIR, "random_forest_importance.csv")
  rf_imp <- tryCatch(
    read_csv(rf_imp_path, show_col_types = FALSE) |>
      mutate(
        rf_rank_mda  = rank(-mean_decrease_accuracy, ties.method = "first"),
        rf_rank_gini = rank(-mean_decrease_gini,     ties.method = "first")
      ),
    error = function(e) NULL
  )

  summary_rows <- map_dfr(all_candidate_vars, function(v) {
    sel_min <- v %in% lasso_min_vars
    sel_1se <- v %in% lasso_1se_vars

    rf_row  <- if (!is.null(rf_imp)) rf_imp[rf_imp$variable == v, ] else NULL
    rf_mda  <- if (!is.null(rf_row) && nrow(rf_row) > 0) rf_row$mean_decrease_accuracy else NA_real_
    rf_gini <- if (!is.null(rf_row) && nrow(rf_row) > 0) rf_row$mean_decrease_gini else NA_real_
    rf_rank_mda  <- if (!is.null(rf_row) && nrow(rf_row) > 0) rf_row$rf_rank_mda else NA_integer_
    rf_rank_gini <- if (!is.null(rf_row) && nrow(rf_row) > 0) rf_row$rf_rank_gini else NA_integer_

    rf_supp_row <- if (!is.null(rf_support)) rf_support[rf_support$variable == v, ] else NULL
    rf_supp <- if (!is.null(rf_supp_row) && nrow(rf_supp_row) > 0) rf_supp_row$rf_supported_lasso else NA

    uni_p_row <- uni_var_summary[uni_var_summary$variable == v, ]
    uni_p     <- if (nrow(uni_p_row) > 0) uni_p_row$global_p_value else NA_real_

    in_mv   <- v %in% pipeline_candidates
    mv_row  <- if (!is.null(res_mv$tidy)) mv_var_summary[mv_var_summary$variable == v, ] else NULL
    mv_p    <- if (!is.null(mv_row) && nrow(mv_row) > 0) mv_row$multivariable_global_p else NA_real_

    selected <- v %in% final_ps_vars

    note <- if (!sel_1se && selected) {
      "ERROR: selected but not in LASSO 1se"
    } else if (!sel_1se) {
      "Not in LASSO 1se shortlist — excluded before multivariable step"
    } else if (is.na(mv_p)) {
      "In LASSO 1se; multivariable p not available"
    } else if (mv_p >= 0.05) {
      paste0("In LASSO 1se; multivariable global p=", round(mv_p, 4), " >= 0.05 — not retained")
    } else {
      paste0("In LASSO 1se; multivariable global p=", round(mv_p, 4), " < 0.05 — SELECTED")
    }

    data.frame(
      variable                        = v,
      in_17_candidate_set             = TRUE,
      selected_by_lasso_lambda_min    = sel_min,
      selected_by_lasso_lambda_1se    = sel_1se,
      rf_mda                          = round(rf_mda, 4),
      rf_gini                         = round(rf_gini, 4),
      rf_rank_mda                     = rf_rank_mda,
      rf_rank_gini                    = rf_rank_gini,
      rf_supported_lasso              = rf_supp,
      univariate_global_p             = round(uni_p, 5),
      included_in_pipeline_multivariable = in_mv,
      multivariable_global_p          = round(mv_p, 5),
      selected_in_pipeline_final      = selected,
      selection_notes                 = note,
      stringsAsFactors                = FALSE
    )
  })

  write_csv(summary_rows, file.path(TABLES_DIR, "pipeline_selection_summary.csv"))

  if (length(final_ps_vars) == 0) {
    cat("\n  [STOP] No pipeline-selected variables met global p < 0.05.",
        "Saving LASSO-only exploratory model instead.\n")
    fit_and_save(
      pipeline_candidates, "pipeline_lasso_only_exploratory",
      "logistic_model_pipeline_lasso_only_exploratory.rds",
      "multivariable_logistic_results_pipeline_lasso_only_exploratory.csv"
    )
    saveRDS(pipeline_candidates,
            file.path(MODELS_DIR, "pipeline_lasso_only_exploratory_features.rds"))
    model_ps <- NULL
  } else {
    model_ps <- fit_and_save(
      final_ps_vars, "pipeline_selected",
      "logistic_model_pipeline_selected.rds",
      "multivariable_logistic_results_pipeline_selected_final.csv"
    )
    saveRDS(final_ps_vars, file.path(MODELS_DIR, "pipeline_selected_features.rds"))
    write_csv(data.frame(variable = final_ps_vars),
              file.path(TABLES_DIR, "pipeline_selected_features.csv"))
    cat("  Final pipeline-selected variables (", length(final_ps_vars), "):",
        paste(final_ps_vars, collapse = ", "), "\n")
  }
}

# ============================================================
# 4. Final model (nomogram basis)
# ============================================================
final_model <- switch(FINAL_MODEL_CHOICE,
  paper_text      = model_a,
  nomogram_figure = model_b,
  model_b   # default
)

if (!is.null(final_model)) {
  saveRDS(final_model, file.path(MODELS_DIR, "logistic_model_final.rds"))
}

# ============================================================
# 5. Pipeline feature-selection audit table (stage-level summary)
# ============================================================
make_audit_row <- function(stage, vars) {
  data.frame(
    stage       = stage,
    n_variables = length(vars),
    variables   = paste(vars, collapse = ", "),
    stringsAsFactors = FALSE
  )
}

final_ps_vars_for_audit <- tryCatch(
  readRDS(file.path(MODELS_DIR, "pipeline_selected_features.rds")),
  error = function(e) character(0)
)
lasso_min_vars_audit <- tryCatch(
  readRDS(file.path(MODELS_DIR, "lasso_selected_vars_lambda_min.rds")),
  error = function(e) character(0)
)
lasso_1se_vars_audit <- tryCatch(
  readRDS(file.path(MODELS_DIR, "lasso_selected_vars_lambda_1se.rds")),
  error = function(e) character(0)
)

audit_tbl <- bind_rows(
  make_audit_row("LASSO lambda.min",       lasso_min_vars_audit),
  make_audit_row("LASSO lambda.1se",       lasso_1se_vars_audit),
  make_audit_row("Pipeline-selected final", final_ps_vars_for_audit),
  make_audit_row("Nomogram Figure",        FEATURES_NOMOGRAM_FIGURE),
  make_audit_row("Paper Text",             FEATURES_PAPER_TEXT),
  make_audit_row("AJCC T/N baseline",      FEATURES_TNM_BASELINE)
)

write_csv(audit_tbl, file.path(TABLES_DIR, "pipeline_feature_selection_audit.csv"))
print(audit_tbl[, c("stage", "n_variables")])

cat("=== 04 complete ===\n\n")
