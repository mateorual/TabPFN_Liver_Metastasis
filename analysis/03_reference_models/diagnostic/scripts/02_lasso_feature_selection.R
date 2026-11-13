# scripts/02_lasso_feature_selection.R
# LASSO logistic regression feature selection -> Figure 1A/B (fig1_ab_lasso_combined).
#
# Primary shortlist: lambda.1se (paper cites a "minimum criterion" sparse solution
# of ~7 non-zero coefficients; lambda.min retains nearly all variables, so
# lambda.1se is used as the primary sparse criterion). lambda.min is kept for audit.

cat("=== 02_lasso_feature_selection.R ===\n")

source("config/config.R")
source("scripts/helpers_diagnostic.R")

for (pkg in c("glmnet", "ggplot2", "dplyr", "readr", "stringr", "patchwork")) {
  if (!requireNamespace(pkg, quietly = TRUE))
    stop("Install '", pkg, "' with: install.packages('", pkg, "')")
}
suppressPackageStartupMessages({
  library(glmnet); library(ggplot2); library(dplyr)
  library(readr);  library(stringr);  library(patchwork)
})

# ---- Load data --------------------------------------------------------------
df <- readRDS(file.path(MODELS_DIR, "diagnostic_data_clean.rds"))
lasso_preds <- intersect(CANDIDATE_PREDICTORS, names(df))

df_model <- df |> select(all_of(c(lasso_preds, OUTCOME))) |> na.omit()
n_drop   <- nrow(df) - nrow(df_model)
if (n_drop > 0) cat("  Dropped", n_drop, "rows (complete-case).\n")
log_sample_size("02_lasso", "lasso_cv", nrow(df), nrow(df_model))

X <- model.matrix(
  as.formula(paste("~", paste(lasso_preds, collapse = " + "))),
  data = df_model
)[, -1]
y <- df_model[[OUTCOME]]

# Map a dummy term back to its original variable name
map_to_original <- function(terms, preds) {
  sapply(terms, function(t) {
    for (v in preds) if (startsWith(t, v)) return(v)
    t
  })
}

# Count retained ORIGINAL features (not dummy columns) at each lambda
orig_nonzero_path <- function(beta_mat, term_names, preds) {
  orig <- map_to_original(term_names, preds)
  apply(beta_mat != 0, 2, function(z) length(unique(orig[z])))
}

extract_coefs <- function(fit, lambda_val, label) {
  coefs <- coef(fit, s = lambda_val)
  data.frame(
    term        = rownames(coefs),
    coefficient = as.numeric(coefs),
    stringsAsFactors = FALSE
  ) |>
    filter(term != "(Intercept)", coefficient != 0) |>
    arrange(desc(abs(coefficient))) |>
    mutate(lambda_used  = label, lambda_value = lambda_val)
}

# ---- 10-fold CV (MSE) --------------------------------------------------------
set.seed(SEED)
cv_dev <- tryCatch(
  cv.glmnet(X, y, family = "binomial", alpha = 1, nfolds = 10,
            type.measure = "mse"),
  error = function(e) {
    set.seed(SEED)
    cv.glmnet(X, y, family = "binomial", alpha = 1, nfolds = 10,
              type.measure = "deviance")
  }
)
cv_measure_label <- if (!is.null(attr(cv_dev, "type.measure")) &&
                         attr(cv_dev, "type.measure") == "deviance")
  "Mean Cross-Validated Deviance" else "Mean-Squared Error"

lambda_min <- cv_dev$lambda.min
lambda_1se <- cv_dev$lambda.1se

coefs_min <- extract_coefs(cv_dev, lambda_min, "lambda.min")
coefs_1se <- extract_coefs(cv_dev, lambda_1se, "lambda.1se")
coefs_min$original_variable <- map_to_original(coefs_min$term, lasso_preds)
coefs_1se$original_variable <- map_to_original(coefs_1se$term, lasso_preds)

n_orig_min <- length(unique(coefs_min$original_variable))
n_orig_1se <- length(unique(coefs_1se$original_variable))
vars_1se   <- unique(coefs_1se$original_variable)
vars_min   <- unique(coefs_min$original_variable)

cat("  lambda.min =", round(lambda_min, 6), "->", n_orig_min, "vars:",
    paste(vars_min, collapse = ", "), "\n")
cat("  lambda.1se =", round(lambda_1se, 6), "->", n_orig_1se, "vars:",
    paste(vars_1se, collapse = ", "), "\n")

write_csv(coefs_min, file.path(TABLES_DIR, "lasso_selected_features_lambda_min.csv"))
write_csv(coefs_1se, file.path(TABLES_DIR, "lasso_selected_features_lambda_1se.csv"))
saveRDS(vars_1se, file.path(MODELS_DIR, "lasso_selected_vars_lambda_1se.rds"))
saveRDS(vars_min,  file.path(MODELS_DIR, "lasso_selected_vars_lambda_min.rds"))
write_csv(data.frame(variable = vars_1se),
          file.path(TABLES_DIR, "lasso_selected_vars_lambda_1se.csv"))
write_csv(data.frame(variable = vars_min),
          file.path(TABLES_DIR, "lasso_selected_vars_lambda_min.csv"))

# ---- Full LASSO path (for top-axis original-feature counts) -----------------
fit_full  <- glmnet(X, y, family = "binomial", alpha = 1)
beta_mat  <- as.matrix(coef(fit_full))[-1, , drop = FALSE]
feat_terms <- rownames(beta_mat)

n_orig_path <- orig_nonzero_path(beta_mat, feat_terms, lasso_preds)
lambda_path_df <- data.frame(
  log_lambda     = log(fit_full$lambda),
  n_orig_nonzero = n_orig_path
)

get_n_orig_at <- function(lambda_val) {
  idx <- which.min(abs(fit_full$lambda - lambda_val))
  n_orig_path[idx]
}
n_orig_at_min <- get_n_orig_at(lambda_min)
n_orig_at_1se <- get_n_orig_at(lambda_1se)

ps_feat_path <- file.path(MODELS_DIR, "pipeline_selected_features.rds")
n_ps <- if (file.exists(ps_feat_path)) length(readRDS(ps_feat_path)) else NULL

# =============================================================
# Panel A: CV lambda plot
# =============================================================
cv_df_dev <- data.frame(
  log_lambda = log(cv_dev$lambda),
  cvm        = cv_dev$cvm,
  cvup       = cv_dev$cvup,
  cvlo       = cv_dev$cvlo
)

n_pts      <- nrow(lambda_path_df)
sel_idx    <- unique(round(seq(1, n_pts, length.out = min(15, n_pts))))
top_breaks <- lambda_path_df$log_lambda[sel_idx]
top_labels <- lambda_path_df$n_orig_nonzero[sel_idx]

y_top  <- max(cv_df_dev$cvup, na.rm = TRUE)
y_step <- diff(range(cv_df_dev$cvm, na.rm = TRUE)) * 0.12

subtitle_lambda <- paste0(
  "lambda.min = ", round(lambda_min, 4),
  " (", n_orig_at_min, " retained original variables, audit/literal minimum)  |  ",
  "lambda.1se = ", round(lambda_1se, 4),
  " (", n_orig_at_1se, " retained original variables, primary sparse solution)"
)

p_lambda <- ggplot(cv_df_dev, aes(x = log_lambda, y = cvm)) +
  geom_ribbon(aes(ymin = cvlo, ymax = cvup), fill = "grey80", alpha = 0.6) +
  geom_line(color = "steelblue", linewidth = 0.9) +
  geom_vline(xintercept = log(lambda_min), color = "grey40",
             linetype = "dashed", linewidth = 0.7) +
  annotate("text", x = log(lambda_min), y = y_top - y_step * 0.1,
           label = paste0("lambda.min\n", n_orig_at_min,
                          " retained\noriginal variables"),
           color = "grey40", size = 2.6, hjust = 1.06, vjust = 1) +
  geom_vline(xintercept = log(lambda_1se), color = "#1f77b4",
             linetype = "solid", linewidth = 1.0) +
  annotate("text", x = log(lambda_1se), y = y_top - y_step * 0.1,
           label = paste0("lambda.1se\n", n_orig_at_1se,
                          " retained\noriginal variables\n[PRIMARY]"),
           color = "#1f77b4", size = 2.6, hjust = -0.07, vjust = 1) +
  scale_x_continuous(
    name     = "log(lambda)",
    sec.axis = sec_axis(~ ., breaks = top_breaks, labels = top_labels,
                        name = "Retained original clinical features")
  ) +
  labs(
    title    = "A  Optimal penalization coefficient",
    y        = cv_measure_label
  ) +
  theme_bw(base_size = 12) +
  theme(plot.title = element_text(face = "bold", size = 11))

# =============================================================
# Panel B: coefficient profiles
# =============================================================
profile_df <- data.frame(
  log_lambda  = rep(log(fit_full$lambda), each = length(feat_terms)),
  term        = rep(feat_terms, times = length(fit_full$lambda)),
  coefficient = as.numeric(beta_mat)
)
profile_df$original_variable <- map_to_original(profile_df$term, lasso_preds)

palette_17 <- c(
  "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
  "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
  "#aec7e8","#ffbb78","#98df8a","#ff9896","#c5b0d5",
  "#c49c94","#f7b6d2"
)
n_vars    <- length(lasso_preds)
color_map <- setNames(palette_17[seq_len(n_vars)], lasso_preds)

sel_pretty <- str_wrap(
  paste0("Selected at lambda.1se:\n",
         paste(pretty_var_label(vars_1se), collapse = ", ")),
  width = 42
)

p_profiles <- ggplot(profile_df,
                     aes(x = log_lambda, y = coefficient,
                         group = term, color = original_variable)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black", linewidth = 0.4) +
  geom_line(linewidth = 0.45, alpha = 0.9) +
  geom_vline(xintercept = log(lambda_min), color = "grey50",
             linetype = "dashed", linewidth = 0.55) +
  geom_vline(xintercept = log(lambda_1se), color = "#1f77b4",
             linetype = "solid", linewidth = 1.0) +
  annotate("text", x = log(lambda_1se), y = Inf,
           label = paste0("LASSO lambda.1se:\n",
                          n_orig_at_1se, " retained original variables",
                          if (!is.null(n_ps))
                            paste0("\nPipeline-selected final: ", n_ps,
                                   " variables") else ""),
           color = "#1f77b4", size = 2.6, hjust = -0.06, vjust = 1.3) +
  annotate("label", x = -Inf, y = -Inf, label = sel_pretty,
           hjust = -0.04, vjust = -0.15, size = 2.4, label.size = 0.25,
           fill = "white", color = "#1f77b4", alpha = 0.85) +
  scale_color_manual(
    values = color_map, name = "Clinical feature",
    guide  = guide_legend(ncol = 1, keyheight = 0.8,
                          override.aes = list(linewidth = 1.6, alpha = 1))
  ) +
  scale_x_continuous(
    name     = "log(lambda)",
    sec.axis = sec_axis(~ ., breaks = top_breaks, labels = top_labels,
                        name = "Retained original clinical features")
  ) +
  labs(title = "B  Coefficient profiles", y = "Coefficient") +
  theme_bw(base_size = 11) +
  theme(
    plot.title      = element_text(face = "bold", size = 11),
    legend.position = "right",
    legend.text     = element_text(size = 7.5),
    legend.key.size = unit(0.45, "cm")
  )

# ---- Combine into Figure 1A/B ------------------------------------------------
fig1_ab <- p_lambda + p_profiles +
  plot_layout(widths = c(1, 1.35)) +
  plot_annotation(
    subtitle = str_wrap(subtitle_lambda, width = 130),
    theme    = theme(plot.subtitle = element_text(size = 8.5, color = "grey30"))
  )

ggsave(file.path(FIGURES_DIR, "fig1_ab_lasso_combined.png"), fig1_ab,
       width = 17, height = 7, dpi = 150)
ggsave(file.path(FIGURES_DIR, "fig1_ab_lasso_combined.pdf"), fig1_ab,
       width = 17, height = 7)
cat("  Saved fig1_ab_lasso_combined figures.\n")

saveRDS(cv_dev, file.path(MODELS_DIR, "lasso_cv_model.rds"))

cat("=== 02 complete ===\n\n")
