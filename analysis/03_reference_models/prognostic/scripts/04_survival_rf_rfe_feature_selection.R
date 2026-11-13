# scripts/04_survival_rf_rfe_feature_selection.R
# Survival RF recursive feature elimination -> fig4_rf_rfe_rmse.
#
# Candidate set = all estimable predictors (RF_RFE_CANDIDATE_MODE in config).
# RFE error metric = K-fold cross-validated 1 - Harrell's C-index: for each
# k-variable model, fit a survival RF on training folds and evaluate
# concordance on the held-out fold. This reproduces the "RMSE
# (Cross-Validation)" curve in Li et al. (2023) Figure 4, which does not
# specify how RMSE is computed for censored survival data.

cat("=== 04_survival_rf_rfe_feature_selection.R ===\n")

source("config/config.R")

for (pkg in c("survival", "ggplot2", "dplyr", "readr")) {
  if (!requireNamespace(pkg, quietly = TRUE))
    stop("Install '", pkg, "' with: install.packages('", pkg, "')")
}
suppressPackageStartupMessages({
  library(survival); library(ggplot2); library(dplyr); library(readr)
})

set.seed(SEED)

df   <- readRDS(file.path(MODELS_DIR, "prognostic_data_clean.rds"))
df_t <- df[df$cohort == "Training", ]
cat("  Training N =", nrow(df_t), "| Events =", sum(df_t[[EVENT_VAR]]), "\n")

# ---- Build candidate set -----------------------------------------------------
uni_sig  <- tryCatch({
  tbl <- read_csv(file.path(TABLES_DIR, "univariate_cox_variable_level_summary.csv"),
                  show_col_types = FALSE)
  tbl |> filter(!is.na(global_lrt_p), global_lrt_p < 0.05) |> pull(predictor)
}, error = function(e) { cat("  [warn] uni summary not found.\n"); character(0) })

aic_vars <- tryCatch(
  readRDS(file.path(MODELS_DIR, "cox_aic_selected_vars.rds")),
  error = function(e) { cat("  [warn] cox_aic_selected_vars.rds not found.\n"); character(0) }
)

preds_all  <- intersect(CANDIDATE_PREDICTORS, names(df_t))
zero_v <- sapply(preds_all, function(v) {
  if (is.factor(df_t[[v]])) length(unique(droplevels(df_t[[v]]))) <= 1
  else FALSE
})
preds_est <- preds_all[!zero_v]

candidate_mode <- if (exists("RF_RFE_CANDIDATE_MODE")) RF_RFE_CANDIDATE_MODE else "screened_union"

if (candidate_mode == "all_estimable") {
  candidates <- preds_est
} else if (candidate_mode == "screened_union") {
  candidates <- unique(intersect(union(uni_sig, aic_vars), preds_est))
} else {
  stop("Invalid RF_RFE_CANDIDATE_MODE: ", candidate_mode)
}

if (length(candidates) == 0) {
  cat("  [note] Empty candidate set; using all estimable predictors.\n")
  candidates <- preds_est
}

cat("  RF/RFE candidate mode:", candidate_mode, "\n")
cat("  Candidates (", length(candidates), "):",
    paste(candidates, collapse = ", "), "\n")

df_rf <- df_t |>
  select(all_of(c(candidates, TIME_VAR, EVENT_VAR))) |>
  na.omit() |>
  as.data.frame()
df_rf[[TIME_VAR]]  <- as.numeric(df_rf[[TIME_VAR]])
df_rf[[EVENT_VAR]] <- as.integer(df_rf[[EVENT_VAR]])
cat("  RF modeling sample:", nrow(df_rf), "\n")

use_rfsrc  <- requireNamespace("randomForestSRC", quietly = TRUE)
use_ranger <- !use_rfsrc && requireNamespace("ranger", quietly = TRUE)
if (!use_rfsrc && !use_ranger) {
  stop("Install randomForestSRC or ranger: install.packages('randomForestSRC')")
}
if (use_rfsrc) suppressPackageStartupMessages(library(randomForestSRC))
if (use_ranger) suppressPackageStartupMessages(library(ranger))

# ---- Full survival RF: variable importance ranking ---------------------------
cat("  Fitting full survival RF for variable ranking ...\n")
rf_formula <- as.formula(
  paste0("Surv(", TIME_VAR, ", ", EVENT_VAR, ") ~ ",
         paste(candidates, collapse = " + "))
)

rf_full     <- NULL
ranked_vars <- NULL
imp_tbl     <- NULL
rf_pkg      <- NULL

if (use_rfsrc) {
  rf_full <- tryCatch(
    rfsrc(rf_formula, data = df_rf, ntree = 500,
          importance = TRUE, seed = SEED, verbose = FALSE),
    error = function(e) { cat("  [warn]", conditionMessage(e), "\n"); NULL }
  )
  if (!is.null(rf_full)) {
    vimp <- sort(rf_full$importance, decreasing = TRUE)
    ranked_vars <- names(vimp)
    imp_tbl <- data.frame(variable = ranked_vars, importance = as.numeric(vimp),
                          rank = seq_along(ranked_vars), stringsAsFactors = FALSE)
    rf_pkg <- "randomForestSRC"
  }
}

if (is.null(ranked_vars) && use_ranger) {
  rf_full <- tryCatch(
    ranger::ranger(
      formula = rf_formula, data = df_rf, num.trees = 500,
      importance = "permutation", seed = SEED, verbose = FALSE
    ),
    error = function(e) { cat("  [warn]", conditionMessage(e), "\n"); NULL }
  )
  if (!is.null(rf_full)) {
    vimp <- sort(rf_full$variable.importance, decreasing = TRUE)
    ranked_vars <- names(vimp)
    imp_tbl <- data.frame(variable = ranked_vars, importance = as.numeric(vimp),
                          rank = seq_along(ranked_vars), stringsAsFactors = FALSE)
    rf_pkg <- "ranger"
  }
}

if (is.null(ranked_vars)) stop("Survival RF fitting failed.")
saveRDS(rf_full, file.path(MODELS_DIR, "rf_survival_model_full.rds"))
write_csv(imp_tbl, file.path(TABLES_DIR, "rf_variable_importance_survival.csv"))
cat("  Top variables by importance:\n"); print(head(imp_tbl, 10))

# ---- K-fold CV error (1 - C-index per fold) -----------------------------------
K        <- RF_RFE_CV_FOLDS
n_fold   <- nrow(df_rf)
fold_ids <- sample(rep(seq_len(K), length.out = n_fold))

cat("  Running", K, "-fold CV (1 - C-index) for k = 1 to",
    length(ranked_vars), "...\n")

rfe_rows <- vector("list", length(ranked_vars))

for (k in seq_along(ranked_vars)) {
  vars_k <- ranked_vars[1:k]
  f_k    <- as.formula(
    paste0("Surv(", TIME_VAR, ", ", EVENT_VAR, ") ~ ",
           paste(vars_k, collapse = " + "))
  )

  ci_folds <- numeric(K)

  for (fold in seq_len(K)) {
    train_f <- df_rf[fold_ids != fold, , drop = FALSE]
    test_f  <- df_rf[fold_ids == fold, , drop = FALSE]

    rf_f <- tryCatch({
      if (use_rfsrc) {
        rfsrc(f_k, data = train_f, ntree = 300,
              importance = FALSE, seed = SEED + fold, verbose = FALSE)
      } else {
        ranger::ranger(f_k, data = train_f, num.trees = 300,
                       importance = "none", seed = SEED + fold, verbose = FALSE)
      }
    }, error = function(e) NULL)

    if (is.null(rf_f)) { ci_folds[fold] <- NA_real_; next }

    risk <- tryCatch({
      if (use_rfsrc) {
        -predict(rf_f, newdata = test_f)$predicted  # negate: higher = lower mortality risk
      } else {
        pred <- predict(rf_f, data = test_f)
        rowSums(pred$survival)  # higher = more integrated survival = lower risk
      }
    }, error = function(e) NULL)

    if (is.null(risk) || all(is.na(risk))) { ci_folds[fold] <- NA_real_; next }

    tv <- test_f[[TIME_VAR]]
    ev <- test_f[[EVENT_VAR]]
    ci_folds[fold] <- tryCatch(
      1 - survival::concordance(Surv(tv, ev) ~ risk)$concordance,
      error = function(e) NA_real_
    )
  }

  avg_err <- mean(ci_folds, na.rm = TRUE)
  rfe_rows[[k]] <- data.frame(
    k        = k,
    rmse_cv  = avg_err,
    rmse_sd  = if (sum(!is.na(ci_folds)) > 1) sd(ci_folds, na.rm = TRUE) else NA_real_,
    n_obs    = nrow(df_rf),
    top_vars = paste(vars_k, collapse = "; "),
    stringsAsFactors = FALSE
  )

  if (k %% 3 == 0 || k == length(ranked_vars)) {
    cat("   k =", k, "| CV error =", round(avg_err, 4), "\n")
  }
}

rfe_df  <- bind_rows(rfe_rows)

best_k          <- rfe_df$k[which.min(rfe_df$rmse_cv)]
best_rmse       <- rfe_df$rmse_cv[best_k]
selected_vars   <- ranked_vars[1:best_k]
rfe_df$selected <- rfe_df$k == best_k

cat("  Best k =", best_k, "| CV error =", round(best_rmse, 4), "\n")
cat("  RF/RFE-selected vars:", paste(selected_vars, collapse = ", "), "\n")

write_csv(rfe_df, file.path(TABLES_DIR, "rf_rfe_feature_selection_results.csv"))
saveRDS(selected_vars, file.path(MODELS_DIR, "rf_rfe_selected_vars.rds"))
write_csv(data.frame(variable = selected_vars),
          file.path(TABLES_DIR, "rf_rfe_selected_vars.csv"))

rmse_curve_df <- rfe_df |>
  rename(rmse_cv_mean = rmse_cv, rmse_cv_sd = rmse_sd, n_obs_mean = n_obs)
write_csv(rmse_curve_df,  file.path(TABLES_DIR, "rf_rfe_rmse_curve.csv"))
write_csv(data.frame(variable = selected_vars),
          file.path(TABLES_DIR, "rf_rfe_selected_variables.csv"))

method_note_text <- paste0(
  "The 'RMSE (Cross-Validation)' metric in Li et al. (2023) Figure 4 is reproduced ",
  "here as ", K, "-fold CV error = 1 - Harrell's C-index. For each k-variable model: ",
  "fit RSF on training folds, extract risk scores (ensemble mortality for rfsrc; ",
  "negative integrated survival for ranger), compute concordance on the held-out fold, ",
  "then average 1 - C-index across folds. RF/RFE candidate mode: ", candidate_mode, "."
)
write_csv(
  data.frame(section     = "Figure 4 analogue (RF/RFE RMSE = CV 1-C-index)",
             method_used = paste0("kfold_cv_1_minus_cindex_", K, "fold"),
             note        = method_note_text,
             stringsAsFactors = FALSE),
  file.path(TABLES_DIR, "rf_rfe_method_notes.csv")
)
cat("  Saved rf_rfe outputs (rmse_curve, selected_variables, method_notes).\n")

# ---- fig4_rf_rfe_rmse: RMSE (Cross-Validation) vs Number of Variables --------
y_label      <- "RMSE (Cross-Validation)"
fig_subtitle <- paste0(K, "-fold CV | 1 - C-index per fold | ",
                        rf_pkg, " | N=", nrow(df_rf))

n_max_k      <- max(rfe_df$k)
x_step       <- if (n_max_k <= 12) 2L else if (n_max_k <= 20) 3L else 5L
x_int_breaks <- seq(x_step, n_max_k, by = x_step)
if (!(n_max_k %in% x_int_breaks)) x_int_breaks <- sort(c(x_int_breaks, n_max_k))

y_max_rmse   <- max(rfe_df$rmse_cv, na.rm = TRUE)
y_min_rmse   <- min(rfe_df$rmse_cv, na.rm = TRUE)
y_anno       <- y_min_rmse + (y_max_rmse - y_min_rmse) * 0.88
x_offset     <- max(1.2, n_max_k * 0.08)
x_anno       <- if (best_k > n_max_k * 0.75) best_k - x_offset else best_k + x_offset
hjust_anno   <- if (best_k > n_max_k * 0.75) 1 else 0

p_rfe <- ggplot(rfe_df, aes(x = k, y = rmse_cv)) +
  geom_line(color = "steelblue", linewidth = 1.0) +
  geom_point(color = "steelblue", size = 2.5) +
  geom_point(data = rfe_df[rfe_df$selected, ],
             aes(x = k, y = rmse_cv),
             color = "firebrick", size = 4, shape = 18) +
  geom_vline(xintercept = best_k, color = "firebrick",
             linetype = "dashed", linewidth = 0.8) +
  annotate(
    "text",
    x     = x_anno,
    y     = y_anno,
    label = paste0("Minimum RMSE\nk = ", best_k),
    color = "firebrick",
    hjust = hjust_anno,
    vjust = 1,
    size  = 3.5
  ) +
  scale_x_continuous(
    breaks = x_int_breaks,
    limits = c(1, n_max_k + 0.5)
  ) +
  coord_cartesian(clip = "off") +
  labs(
    title    = "RF Feature Selection: RMSE vs Number of Variables",
    subtitle = fig_subtitle,
    x        = "Number of Variables",
    y        = y_label
  ) +
  theme_bw(base_size = 12) +
  theme(plot.title    = element_text(face = "bold"),
        plot.subtitle = element_text(size = 9, color = "grey40"))

ggsave(file.path(FIGURES_DIR, "fig4_rf_rfe_rmse.png"),
       p_rfe, width = 8, height = 5, dpi = 150)
ggsave(file.path(FIGURES_DIR, "fig4_rf_rfe_rmse.pdf"),
       p_rfe, width = 8, height = 5)
cat("  Saved fig4_rf_rfe_rmse figures.\n")

log_method_note(
  "RF RFE feature selection",
  paste0(K, "-fold CV 1-C-index via ", rf_pkg),
  paste0(method_note_text, " Best k=", best_k, ": ",
         paste(selected_vars, collapse = ", "))
)
log_sample_size("04_rf_rfe", "rf_full", nrow(df), nrow(df_rf))

cat("=== 04 complete ===\n\n")
