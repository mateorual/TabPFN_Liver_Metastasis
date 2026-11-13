# scripts/04_km_risk_groups.R
# Stratify patients into Low-risk / High-risk using the embedding Cox model's
# risk score -> fig8_km_risk_groups. Cutpoint = median TRAINING linear
# predictor, applied unchanged to internal validation (same rule as the
# clinical prognostic pipeline).

cat("=== 04_km_risk_groups.R ===\n")

source("config/config.R")

for (pkg in c("survival", "ggplot2", "dplyr", "readr", "patchwork")) {
  if (!requireNamespace(pkg, quietly = TRUE))
    stop("Install '", pkg, "' with: install.packages('", pkg, "')")
}
suppressPackageStartupMessages({
  library(survival); library(ggplot2); library(dplyr)
  library(readr);    library(patchwork)
})

risk_tbl <- read_csv(file.path(TABLES_DIR, "embedding_prognostic_risk_scores.csv"),
                     show_col_types = FALSE)

lp_train  <- risk_tbl$embedding_lp[risk_tbl$cohort == "Training"]
cutpoint  <- median(lp_train, na.rm = TRUE)
cat("  Median training LP (cutpoint):", round(cutpoint, 4), "\n")

cutpoint_tbl <- data.frame(
  method = "Median training linear predictor (embedding Cox model)",
  cutpoint = round(cutpoint, 6), n_training = length(lp_train),
  stringsAsFactors = FALSE
)
write_csv(cutpoint_tbl, file.path(TABLES_DIR, "risk_cutpoint_summary_embedding_prognostic.csv"))

risk_tbl <- risk_tbl |>
  mutate(risk_group = factor(if_else(embedding_lp <= cutpoint, "Low-risk", "High-risk"),
                             levels = c("Low-risk", "High-risk")))
write_csv(risk_tbl, file.path(TABLES_DIR, "risk_group_assignment_embedding_prognostic.csv"))
cat("  Risk group counts:\n"); print(table(risk_tbl$cohort, risk_tbl$risk_group))

GROUP_COLORS <- c("High-risk" = "#2166ac", "Low-risk" = "#d6604d")

make_risk_table <- function(km_fit, time_breaks, x_max) {
  risk_sum <- tryCatch(summary(km_fit, times = time_breaks, extend = TRUE), error = function(e) NULL)
  if (is.null(risk_sum)) return(NULL)
  risk_df <- data.frame(time = risk_sum$time, n_risk = risk_sum$n.risk,
                        risk_group = gsub("risk_group=", "", as.character(risk_sum$strata)),
                        stringsAsFactors = FALSE)
  risk_df$risk_group <- factor(risk_df$risk_group, levels = c("High-risk", "Low-risk"))
  ggplot(risk_df, aes(x = time, y = risk_group, label = n_risk)) +
    geom_text(size = 3.2, color = "black") +
    scale_x_continuous(limits = c(0, x_max), breaks = time_breaks,
                       expand = expansion(mult = c(0.01, 0.02))) +
    scale_y_discrete(drop = FALSE) +
    labs(x = "Time (months)", y = NULL, title = "Number at risk") +
    theme_bw(base_size = 10) +
    theme(plot.title = element_text(size = 9, face = "bold", hjust = 0),
          panel.grid = element_blank(), axis.ticks.y = element_blank(),
          axis.text.y = element_text(size = 9), axis.text.x = element_text(size = 9))
}

make_km_plot <- function(data, title, panel_label = NULL) {
  d <- data[!is.na(data$risk_group), ]
  if (nrow(d) < 10) return(NULL)
  display_title <- if (!is.null(panel_label)) paste0(panel_label, "   ", title) else title

  km_fit  <- survfit(Surv(time_months, event) ~ risk_group, data = d)
  lr_test <- tryCatch(survdiff(Surv(time_months, event) ~ risk_group, data = d), error = function(e) NULL)
  p_val   <- if (!is.null(lr_test)) pchisq(lr_test$chisq, df = 1, lower.tail = FALSE) else NA_real_
  p_label <- if (!is.na(p_val)) { if (p_val < 0.001) "p < 0.001" else paste0("p = ", round(p_val, 3)) } else ""

  strata_names <- gsub("risk_group=", "", names(km_fit$strata))
  km_df <- data.frame(time = km_fit$time, surv = km_fit$surv, lower = km_fit$lower, upper = km_fit$upper,
                      risk_group = rep(strata_names, km_fit$strata), stringsAsFactors = FALSE)
  t0 <- data.frame(time = 0, surv = 1, lower = 1, upper = 1, risk_group = strata_names, stringsAsFactors = FALSE)
  km_df <- rbind(t0, km_df)
  km_df$risk_group <- factor(km_df$risk_group, levels = c("Low-risk", "High-risk"))
  km_df <- km_df[order(km_df$risk_group, km_df$time), ]

  x_max <- max(d$time_months, na.rm = TRUE)
  x_plot_max <- ceiling(x_max / 25) * 25
  time_breaks <- seq(0, x_plot_max, by = 25)

  p_main <- ggplot(km_df, aes(x = time, y = surv, color = risk_group, fill = risk_group)) +
    geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.15, color = NA) +
    geom_step(linewidth = 0.9) +
    scale_color_manual(values = GROUP_COLORS, name = NULL) +
    scale_fill_manual(values = GROUP_COLORS, name = NULL) +
    scale_y_continuous(limits = c(0, 1), labels = scales::percent_format(1),
                       expand = expansion(mult = c(0, 0.02))) +
    scale_x_continuous(limits = c(0, x_plot_max), breaks = time_breaks,
                       expand = expansion(mult = c(0.01, 0.02))) +
    annotate("text", x = x_plot_max * 0.62, y = 0.92, label = p_label, size = 3.5, color = "black") +
    labs(title = display_title, x = NULL, y = "Survival probability") +
    theme_bw(base_size = 11) +
    theme(plot.title = element_text(face = "bold"),
          legend.position = c(0.83, 0.85), legend.background = element_rect(fill = "white", color = NA),
          legend.text = element_text(size = 9), axis.text.x = element_blank(), axis.ticks.x = element_blank())

  p_risk <- make_risk_table(km_fit, time_breaks, x_plot_max)
  combined <- if (!is.null(p_risk)) p_main / p_risk + plot_layout(heights = c(4, 1.5)) else p_main

  med_tbl <- tryCatch({
    med_obj <- summary(km_fit)$table
    data.frame(risk_group = gsub("risk_group=", "", rownames(med_obj)),
               n_patients = as.integer(med_obj[, "records"]), n_events = as.integer(med_obj[, "events"]),
               median_os = as.numeric(med_obj[, "median"]), stringsAsFactors = FALSE)
  }, error = function(e) NULL)

  list(plot = combined, logrank_p = p_val, n = nrow(d), median_tbl = med_tbl)
}

cohort_defs   <- list(Training = risk_tbl[risk_tbl$cohort == "Training", ],
                      `Internal validation` = risk_tbl[risk_tbl$cohort == "Internal validation", ])
cohort_keys   <- c("training", "internal_validation")
cohort_labels <- c("A", "B")

km_results <- list(); logrank_rows <- list(); summary_rows <- list()

for (i in seq_along(cohort_defs)) {
  cn <- names(cohort_defs)[i]; ck <- cohort_keys[i]
  res <- make_km_plot(cohort_defs[[cn]], title = cn, panel_label = cohort_labels[i])
  if (is.null(res)) next
  km_results[[ck]] <- res$plot
  logrank_rows[[ck]] <- data.frame(cohort = cn, cohort_key = ck, n = res$n,
                                   logrank_p = round(res$logrank_p, 5), stringsAsFactors = FALSE)
  if (!is.null(res$median_tbl)) {
    med <- res$median_tbl; med$cohort <- cn; med$logrank_p <- round(res$logrank_p, 5); med$cutpoint <- round(cutpoint, 4)
    summary_rows[[ck]] <- med
  }
  cat("  ", cn, "| p =", round(res$logrank_p, 4), "\n")
}

logrank_tbl <- bind_rows(logrank_rows)
write_csv(logrank_tbl, file.path(TABLES_DIR, "km_logrank_results_embedding_prognostic.csv"))

km_summary <- bind_rows(summary_rows) |>
  select(cohort, risk_group, n_patients, n_events, median_os, logrank_p, cutpoint)
write_csv(km_summary, file.path(TABLES_DIR, "km_risk_group_summary_embedding_prognostic.csv"))
print(km_summary)

panel_order <- c("training", "internal_validation")
panels <- Filter(Negate(is.null), km_results[panel_order])
if (length(panels) > 0) {
  n_panels <- length(panels)
  combined_km <- wrap_plots(panels, ncol = n_panels)
  ggsave(file.path(FIGURES_DIR, "fig8_km_risk_groups.png"), combined_km, width = 8 * n_panels, height = 7, dpi = 150)
  ggsave(file.path(FIGURES_DIR, "fig8_km_risk_groups.pdf"), combined_km, width = 8 * n_panels, height = 7)
  cat("  Saved fig8_km_risk_groups\n")
}

cat("=== 04 complete ===\n\n")
