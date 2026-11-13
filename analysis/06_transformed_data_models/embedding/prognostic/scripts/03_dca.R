# scripts/03_dca.R
# Decision curve analysis for survival prediction at 1-, 2-, and 3-year OS
# -> fig7-style 6-panel figure (embedding Cox model vs AJCC T/N baseline vs
# Treat All vs Treat None, Training + Internal validation).
#
# Method: manual net benefit with complete-case binary outcome at each
# horizon (event by time t; patients censored before t excluded) — identical
# approximation to the clinical prognostic pipeline's script 09, applied here
# to the predicted survival probabilities from script 01.

cat("=== 03_dca.R ===\n")

source("config/config.R")

for (pkg in c("dplyr", "readr", "ggplot2", "patchwork")) {
  if (!requireNamespace(pkg, quietly = TRUE))
    stop("Install '", pkg, "' with: install.packages('", pkg, "')")
}
suppressPackageStartupMessages({
  library(dplyr); library(readr); library(ggplot2); library(patchwork)
})

set.seed(SEED)

surv_long <- read_csv(file.path(TABLES_DIR, "embedding_prognostic_survival_predictions.csv"),
                      show_col_types = FALSE) |>
  rename(horizon_months = time_months)
risk_tbl  <- read_csv(file.path(TABLES_DIR, "embedding_prognostic_risk_scores.csv"),
                      show_col_types = FALSE) |>
  select(patient_id, cohort, followup_months = time_months, event)

dat <- surv_long |> inner_join(risk_tbl, by = c("patient_id", "cohort"))

model_colors <- c(embedding_model = "#2ca02c", tn_baseline = "#9467bd",
                  "Treat All" = "#333333", "Treat None" = "#999999")
model_linetypes <- c(embedding_model = "solid", tn_baseline = "solid",
                     "Treat All" = "dashed", "Treat None" = "dotted")
display_labels <- c(embedding_model = "TabPFN embedding Cox (1024-dim, ridge)",
                    tn_baseline = "AJCC T/N baseline (2 vars)",
                    "Treat All" = "Treat All", "Treat None" = "Treat None")

thresholds <- seq(0.01, 0.99, by = 0.01)

compute_nb_binary <- function(y_bin, risk, thresholds) {
  sapply(thresholds, function(t) {
    tp <- sum(risk >= t & y_bin == 1)
    fp <- sum(risk >= t & y_bin == 0)
    tp / length(y_bin) - fp / length(y_bin) * (t / (1 - t))
  })
}

all_dca_rows <- list()
dca_plots    <- list()

for (cohort_name in c("Training", "Internal validation")) {
  coh_key <- gsub(" ", "_", tolower(cohort_name))

  for (hi in seq_along(TIME_HORIZONS_MONTHS)) {
    t_star <- TIME_HORIZONS_MONTHS[hi]
    t_lbl  <- TIME_HORIZONS_LABELS[hi]

    sub <- dat |> filter(cohort == cohort_name, horizon_months == t_star)
    tv <- sub$followup_months
    ev <- sub$event
    eligible <- (tv >= t_star) | (ev == 1 & tv < t_star)
    sub_e <- sub[eligible, ]
    tv_e  <- tv[eligible]; ev_e <- ev[eligible]
    y_bin <- as.integer(ev_e == 1 & tv_e <= t_star)
    prev  <- mean(y_bin)
    n_elig <- nrow(sub_e)

    cat("  ", cohort_name, t_lbl, ": N=", n_elig, "| events=", sum(y_bin), "\n")
    if (n_elig < 20 || sum(y_bin) < 5) { cat("  [skip] Insufficient events.\n"); next }

    rows_this <- list(
      "Treat All"  = data.frame(threshold = thresholds,
                                 net_benefit = prev - (1 - prev) * (thresholds / (1 - thresholds)),
                                 strategy = "Treat All"),
      "Treat None" = data.frame(threshold = thresholds, net_benefit = 0, strategy = "Treat None"),
      embedding_model = data.frame(threshold = thresholds,
                                    net_benefit = compute_nb_binary(y_bin, 1 - sub_e$embedding_surv, thresholds),
                                    strategy = "embedding_model"),
      tn_baseline = data.frame(threshold = thresholds,
                                net_benefit = compute_nb_binary(y_bin, 1 - sub_e$tn_baseline_surv, thresholds),
                                strategy = "tn_baseline")
    )
    dca_df <- bind_rows(rows_this)
    dca_df$cohort <- cohort_name; dca_df$time_label <- t_lbl; dca_df$time_months <- t_star
    all_dca_rows[[paste(coh_key, t_lbl)]] <- dca_df

    keys_plot <- c("embedding_model", "tn_baseline", "Treat All", "Treat None")
    sub_dca <- dca_df |> filter(strategy %in% keys_plot) |> mutate(strategy = factor(strategy, levels = keys_plot))
    max_nb  <- max(sub_dca$net_benefit[sub_dca$strategy != "Treat None"], na.rm = TRUE)
    y_upper <- max(0.05, ceiling(max_nb * 110) / 100)

    p_dca <- ggplot(sub_dca, aes(x = threshold, y = net_benefit, color = strategy, linetype = strategy)) +
      geom_line(linewidth = 1.0) +
      scale_color_manual(values = model_colors[keys_plot], labels = display_labels[keys_plot],
                         name = NULL, guide = guide_legend(ncol = 1)) +
      scale_linetype_manual(values = model_linetypes[keys_plot], labels = display_labels[keys_plot],
                            name = NULL, guide = guide_legend(ncol = 1)) +
      coord_cartesian(ylim = c(-0.02, y_upper)) +
      labs(title = paste0(t_lbl, " DCA - ", cohort_name),
           subtitle = paste0("N=", n_elig, " | Events=", sum(y_bin)),
           x = "Threshold Probability", y = "Net Benefit") +
      theme_bw(base_size = 10) +
      theme(plot.title = element_text(face = "bold", size = 10),
            plot.subtitle = element_text(size = 8, color = "grey40"),
            legend.position = "bottom", legend.text = element_text(size = 7),
            legend.key.size = unit(0.4, "cm"))

    dca_plots[[paste0(coh_key, "_", gsub("-year", "yr", t_lbl))]] <- p_dca
  }
}

dca_all <- bind_rows(all_dca_rows)
write_csv(dca_all, file.path(TABLES_DIR, "dca_embedding_prognostic_results.csv"))
cat("  Saved dca_embedding_prognostic_results.csv\n")

all_dca_plots_6 <- c(dca_plots[paste0("training_", c("1yr", "2yr", "3yr"))],
                     dca_plots[paste0("internal_validation_", c("1yr", "2yr", "3yr"))])
all_dca_plots_6 <- Filter(Negate(is.null), all_dca_plots_6)
if (length(all_dca_plots_6) > 0) {
  combined6 <- wrap_plots(all_dca_plots_6, ncol = 3) +
    plot_annotation(tag_levels = "A", theme = theme(plot.tag = element_text(face = "bold", size = 13)))
  ggsave(file.path(FIGURES_DIR, "fig7_dca_training_internal_6panel.png"), combined6, width = 18, height = 10, dpi = 150)
  ggsave(file.path(FIGURES_DIR, "fig7_dca_training_internal_6panel.pdf"), combined6, width = 18, height = 10)
  cat("  Saved fig7_dca_training_internal_6panel\n")
}

cat("=== 03 complete ===\n\n")
