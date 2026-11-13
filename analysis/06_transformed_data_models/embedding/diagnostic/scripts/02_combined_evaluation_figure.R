# scripts/02_combined_evaluation_figure.R
# Combined 3-panel evaluation figure for the embedding model vs AJCC T/N
# baseline: A) ROC, B) Calibration (embedding model only), C) DCA.
# Mirrors the clinical pipeline's fig3_abc_pipeline_selected_evaluation
# structure/style so the two approaches are visually comparable.

cat("=== 02_combined_evaluation_figure.R ===\n")

source("config/config.R")

for (pkg in c("pROC", "ggplot2", "dplyr", "readr", "patchwork")) {
  if (!requireNamespace(pkg, quietly = TRUE))
    stop("Install '", pkg, "' with: install.packages('", pkg, "')")
}
suppressPackageStartupMessages({
  library(pROC); library(ggplot2); library(dplyr); library(readr); library(patchwork)
})

set.seed(SEED)

pred <- read_csv(file.path(TABLES_DIR, "embedding_diagnostic_predictions.csv"),
                 show_col_types = FALSE)
n_common <- nrow(pred)

model_colors <- c(embedding_model = "#2ca02c", tnm_baseline = "#9467bd")
model_labels <- c(
  embedding_model = "TabPFN embedding model (512-dim, ridge logistic, OOF)",
  tnm_baseline    = "AJCC T/N baseline (2 vars: T stage + N stage)"
)

# ============================================================
# Panel A — ROC
# ============================================================
roc_list <- list()
auc_info <- list()
for (nm in c("embedding_model", "tnm_baseline")) {
  roc_obj <- pROC::roc(pred[[OUTCOME]], pred[[nm]], direction = "<", quiet = TRUE)
  ci_obj  <- tryCatch(
    pROC::ci.auc(roc_obj, method = "bootstrap", boot.n = 1000),
    error = function(e) pROC::ci.auc(roc_obj, method = "delong")
  )
  roc_list[[nm]] <- roc_obj
  auc_info[[nm]] <- list(auc = round(as.numeric(pROC::auc(roc_obj)), 3),
                         lo  = round(as.numeric(ci_obj[1]), 3),
                         hi  = round(as.numeric(ci_obj[3]), 3))
  cat("  ", nm, ": AUC =", auc_info[[nm]]$auc,
      "(", auc_info[[nm]]$lo, "-", auc_info[[nm]]$hi, ")\n")
}

auc_tbl <- bind_rows(lapply(names(auc_info), function(nm) {
  data.frame(model = nm, label = model_labels[nm],
             auc = auc_info[[nm]]$auc,
             ci_lower_95 = auc_info[[nm]]$lo, ci_upper_95 = auc_info[[nm]]$hi,
             n = n_common, stringsAsFactors = FALSE)
}))
write_csv(auc_tbl, file.path(TABLES_DIR, "roc_auc_embedding_diagnostic.csv"))

roc_label <- function(nm) {
  ai <- auc_info[[nm]]
  paste0(model_labels[nm], "\n  AUC = ", ai$auc, " (", ai$lo, "–", ai$hi, ")")
}
roc_df <- bind_rows(lapply(names(roc_list), function(nm) {
  r <- roc_list[[nm]]
  data.frame(model_key = nm, specificity = rev(r$specificities),
             sensitivity = rev(r$sensitivities), stringsAsFactors = FALSE)
}))

panel_a <- ggplot(roc_df, aes(x = 1 - specificity, y = sensitivity, color = model_key)) +
  geom_line(linewidth = 1.1) +
  geom_abline(slope = 1, intercept = 0, color = "grey60", linetype = "dashed", linewidth = 0.5) +
  scale_color_manual(values = model_colors, labels = sapply(names(roc_list), roc_label),
                     name = NULL, guide = guide_legend(ncol = 1, byrow = TRUE)) +
  coord_equal() +
  labs(tag = "A", title = "ROC", subtitle = paste0("N=", n_common),
       x = "1 − Specificity (FPR)", y = "Sensitivity (TPR)") +
  theme_bw(base_size = 11) +
  theme(plot.tag = element_text(face = "bold", size = 14),
        plot.title = element_text(face = "bold"),
        plot.subtitle = element_text(size = 8.5, color = "grey40"),
        legend.position = "bottom",
        legend.text = element_text(size = 8, lineheight = 0.95),
        legend.key.size = unit(0.5, "cm"))

# ============================================================
# Panel B — Calibration (embedding model, OOF predictions vs observed)
# ============================================================
n_deciles <- 10
cal_df <- pred |>
  mutate(decile = ntile(embedding_model, n_deciles)) |>
  group_by(decile) |>
  summarise(mean_predicted = mean(embedding_model),
            observed       = mean(.data[[OUTCOME]]),
            n              = n(), .groups = "drop")
write_csv(cal_df, file.path(TABLES_DIR, "calibration_embedding_diagnostic_deciles.csv"))

cal_lm    <- lm(observed ~ mean_predicted, data = cal_df)
cal_slope <- round(coef(cal_lm)[2], 3)
cal_int   <- round(coef(cal_lm)[1], 3)
brier     <- mean((pred[[OUTCOME]] - pred$embedding_model)^2)
write_csv(
  data.frame(metric = c("brier_score", "calibration_intercept_deciles", "calibration_slope_deciles"),
             value  = c(round(brier, 4), cal_int, cal_slope)),
  file.path(TABLES_DIR, "calibration_metrics_embedding_diagnostic.csv")
)
cat("  Brier =", round(brier, 4), "| decile-level slope =", cal_slope, "intercept =", cal_int, "\n")

long_ideal <- data.frame(mean_predicted = c(0, 1), observed = c(0, 1), curve = "Ideal")
long_app   <- cal_df |> select(mean_predicted, observed) |> mutate(curve = "Apparent (deciles)")
long_df    <- bind_rows(long_ideal, long_app)
long_df$curve <- factor(long_df$curve, levels = c("Ideal", "Apparent (deciles)"))

panel_b <- ggplot(long_df, aes(x = mean_predicted, y = observed, color = curve, linetype = curve)) +
  geom_line() +
  geom_point(data = long_df |> filter(curve != "Ideal"), size = 2, show.legend = FALSE) +
  scale_color_manual(values = c("Ideal" = "gray50", "Apparent (deciles)" = "steelblue"), name = NULL) +
  scale_linetype_manual(values = c("Ideal" = "dashed", "Apparent (deciles)" = "solid"), name = NULL) +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  labs(tag = "B", title = "Calibration",
       subtitle = "Model: TabPFN embedding (OOF)",
       x = "Predicted Pr{liver=1}", y = "Actual Probability") +
  theme_bw(base_size = 11) +
  theme(plot.tag = element_text(face = "bold", size = 14),
        plot.title = element_text(face = "bold"),
        plot.subtitle = element_text(size = 8.5, color = "grey40"),
        legend.position = "bottom",
        legend.text = element_text(size = 9))

# ============================================================
# Panel C — DCA
# ============================================================
thresholds <- seq(0.01, 0.99, by = 0.01)
prevalence <- mean(pred[[OUTCOME]])

compute_nb <- function(y, probs, thresholds) {
  sapply(thresholds, function(t) {
    tp <- sum(probs >= t & y == 1)
    fp <- sum(probs >= t & y == 0)
    tp / length(y) - fp / length(y) * (t / (1 - t))
  })
}

dca_rows <- list(
  "Treat All"  = data.frame(threshold = thresholds,
                             net_benefit = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds)),
                             strategy = "Treat All"),
  "Treat None" = data.frame(threshold = thresholds, net_benefit = 0, strategy = "Treat None"),
  embedding_model = data.frame(threshold = thresholds,
                                net_benefit = compute_nb(pred[[OUTCOME]], pred$embedding_model, thresholds),
                                strategy = "embedding_model"),
  tnm_baseline = data.frame(threshold = thresholds,
                             net_benefit = compute_nb(pred[[OUTCOME]], pred$tnm_baseline, thresholds),
                             strategy = "tnm_baseline")
)
dca_df <- bind_rows(dca_rows)
write_csv(dca_df, file.path(TABLES_DIR, "dca_embedding_diagnostic_results.csv"))

dca_keys   <- c("embedding_model", "tnm_baseline", "Treat All", "Treat None")
dca_colors <- c(model_colors, "Treat All" = "#333333", "Treat None" = "#999999")
dca_linetypes <- c(embedding_model = "solid", tnm_baseline = "solid",
                   "Treat All" = "dashed", "Treat None" = "dotted")
dca_display_lbls <- c(model_labels, "Treat All" = "Treat All", "Treat None" = "Treat None")

sub_dca <- dca_df |> filter(strategy %in% dca_keys) |> mutate(strategy = factor(strategy, levels = dca_keys))
max_nb  <- max(sub_dca$net_benefit[sub_dca$strategy != "Treat None"], na.rm = TRUE)
y_upper <- max(0.05, ceiling(max_nb * 110) / 100)

panel_c <- ggplot(sub_dca, aes(x = threshold, y = net_benefit, color = strategy, linetype = strategy)) +
  geom_line(linewidth = 1.0) +
  scale_color_manual(values = dca_colors[dca_keys], labels = dca_display_lbls[dca_keys],
                     name = NULL, guide = guide_legend(ncol = 1)) +
  scale_linetype_manual(values = dca_linetypes[dca_keys], labels = dca_display_lbls[dca_keys],
                        name = NULL, guide = guide_legend(ncol = 1)) +
  coord_cartesian(ylim = c(-0.02, y_upper)) +
  labs(tag = "C", title = "Decision Curve Analysis", subtitle = paste0("N=", n_common),
       x = "Threshold Probability", y = "Net Benefit") +
  theme_bw(base_size = 11) +
  theme(plot.tag = element_text(face = "bold", size = 14),
        plot.title = element_text(face = "bold"),
        plot.subtitle = element_text(size = 8.5, color = "grey40"),
        legend.position = "bottom",
        legend.text = element_text(size = 8, lineheight = 0.95))

# ============================================================
# Combine and save
# ============================================================
fig <- panel_a + panel_b + panel_c + plot_layout(ncol = 3)
ggsave(file.path(FIGURES_DIR, "fig_embedding_diagnostic_evaluation.png"), fig,
       width = 21, height = 8.6, dpi = 150)
ggsave(file.path(FIGURES_DIR, "fig_embedding_diagnostic_evaluation.pdf"), fig,
       width = 21, height = 8.6)
cat("  Saved fig_embedding_diagnostic_evaluation figures.\n")

cat("=== 02 complete ===\n\n")
