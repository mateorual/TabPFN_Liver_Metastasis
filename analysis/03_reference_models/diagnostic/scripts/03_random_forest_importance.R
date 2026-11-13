# scripts/03_random_forest_importance.R
# Random forest feature importance -> Figure 1C/D (fig1_cd_rf_importance_combined).

cat("=== 03_random_forest_importance.R ===\n")

source("config/config.R")
source("scripts/helpers_diagnostic.R")

for (pkg in c("randomForest", "ggplot2", "dplyr", "readr", "patchwork", "tibble")) {
  if (!requireNamespace(pkg, quietly = TRUE))
    stop("Install '", pkg, "' with: install.packages('", pkg, "')")
}
suppressPackageStartupMessages({
  library(randomForest); library(ggplot2); library(dplyr)
  library(readr);        library(patchwork)
})

df <- readRDS(file.path(MODELS_DIR, "diagnostic_data_clean.rds"))
rf_preds <- intersect(CANDIDATE_PREDICTORS, names(df))

df_model <- df |>
  select(all_of(c(rf_preds, OUTCOME))) |>
  na.omit() |>
  mutate(outcome_f = factor(.data[[OUTCOME]], levels = c(0, 1), labels = c("No", "Yes")))

n_drop <- nrow(df) - nrow(df_model)
if (n_drop > 0) cat("  Dropped", n_drop, "rows (complete-case).\n")

build_bar_plot <- function(data, x_var, x_label, title) {
  data <- data |>
    mutate(variable = pretty_var_label(variable)) |>
    arrange(.data[[x_var]]) |>
    mutate(variable = factor(variable, levels = variable))
  ggplot(data, aes(x = .data[[x_var]], y = variable)) +
    geom_col(fill = "steelblue", color = "white", width = 0.7) +
    labs(title = title, x = x_label, y = NULL) +
    theme_bw(base_size = 12) +
    theme(plot.title  = element_text(face = "bold"),
          axis.text.y = element_text(size = 10))
}

top_n_plot <- min(20, length(rf_preds))

cat("  Fitting randomForest (ntree=1000) ...\n")
set.seed(SEED)
rf_fit <- randomForest(
  x          = df_model[, rf_preds],
  y          = df_model$outcome_f,
  ntree      = 1000,
  importance = TRUE
)
cat("  OOB error:", round(rf_fit$err.rate[1000, "OOB"] * 100, 2), "%\n")

imp <- as.data.frame(importance(rf_fit, type = NULL, scale = TRUE)) |>
  tibble::rownames_to_column("variable") |>
  rename(mean_decrease_accuracy = MeanDecreaseAccuracy,
         mean_decrease_gini     = MeanDecreaseGini) |>
  arrange(desc(mean_decrease_gini))

write_csv(imp, file.path(TABLES_DIR, "random_forest_importance.csv"))

p_acc <- build_bar_plot(
  imp |> slice_max(mean_decrease_accuracy, n = top_n_plot),
  "mean_decrease_accuracy", "Mean Decrease in Accuracy",
  "C  Mean Decrease Accuracy"
)
p_gini <- build_bar_plot(
  imp |> slice_max(mean_decrease_gini, n = top_n_plot),
  "mean_decrease_gini", "Mean Decrease in Gini",
  "D  Mean Decrease Gini"
)

fig1_cd <- (p_acc + labs(tag = "C") +
              theme(plot.tag = element_text(face = "bold", size = 14),
                    plot.margin = ggplot2::margin(5.5, 18, 5.5, 5.5))) +
  (p_gini + labs(tag = "D") +
     theme(plot.tag = element_text(face = "bold", size = 14),
           plot.margin = ggplot2::margin(5.5, 5.5, 5.5, 18))) +
  plot_layout(ncol = 2)

ggsave(file.path(FIGURES_DIR, "fig1_cd_rf_importance_combined.png"), fig1_cd,
       width = 15, height = 6, dpi = 150)
ggsave(file.path(FIGURES_DIR, "fig1_cd_rf_importance_combined.pdf"), fig1_cd,
       width = 15, height = 6)
cat("  Saved fig1_cd_rf_importance_combined figures.\n")

saveRDS(rf_fit, file.path(MODELS_DIR, "random_forest_diagnostic_model.rds"))
rf_top_vars <- imp |> arrange(desc(mean_decrease_gini)) |> pull(variable)
saveRDS(rf_top_vars, file.path(MODELS_DIR, "rf_top_vars.rds"))

log_sample_size("03_rf", "random_forest", nrow(df), nrow(df_model))

# ---- RF support table for the LASSO lambda.1se shortlist (audit only) -------
lasso_1se_path <- file.path(MODELS_DIR, "lasso_selected_vars_lambda_1se.rds")
if (file.exists(lasso_1se_path)) {
  lasso_1se_vars <- readRDS(lasso_1se_path)
  rf_support <- imp |>
    mutate(
      rf_rank_mda        = rank(-mean_decrease_accuracy, ties.method = "first"),
      rf_rank_gini       = rank(-mean_decrease_gini,     ties.method = "first"),
      lasso_1se_selected = variable %in% lasso_1se_vars,
      rf_positive_mda    = mean_decrease_accuracy > 0,
      rf_positive_gini   = mean_decrease_gini > 0,
      rf_supported_lasso = lasso_1se_selected & rf_positive_mda & rf_positive_gini
    ) |>
    arrange(rf_rank_mda)

  write_csv(rf_support, file.path(TABLES_DIR, "rf_support_for_lasso_1se.csv"))
  cat("  RF-supported vars within LASSO 1se shortlist:",
      paste(rf_support$variable[rf_support$rf_supported_lasso], collapse = ", "), "\n")
} else {
  cat("  [skip] lasso_selected_vars_lambda_1se.rds not found. Run script 02 first.\n")
}

cat("=== 03 complete ===\n\n")
