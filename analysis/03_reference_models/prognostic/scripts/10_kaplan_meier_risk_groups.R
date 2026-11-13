# scripts/10_kaplan_meier_risk_groups.R
# Stratify patients into Low-risk / High-risk using the primary Cox model
# -> fig8_km_risk_groups.
# Cutpoint = median linear predictor in the TRAINING cohort, applied unchanged
# to all cohorts. KM curves with 95% CI ribbons and number-at-risk tables.

cat("=== 10_kaplan_meier_risk_groups.R ===\n")

source("config/config.R")

for (pkg in c("survival", "ggplot2", "dplyr", "readr", "patchwork")) {
  if (!requireNamespace(pkg, quietly = TRUE))
    stop("Install '", pkg, "' with: install.packages('", pkg, "')")
}
suppressPackageStartupMessages({
  library(survival); library(ggplot2); library(dplyr)
  library(readr);    library(patchwork)
})

df   <- readRDS(file.path(MODELS_DIR, "prognostic_data_clean.rds"))
df_t <- df[df$cohort == "Training", ]
df_v <- df[df$cohort == "Internal validation", ]
cat("  Training N =", nrow(df_t), "| Validation N =", nrow(df_v), "\n")

cox_nom <- tryCatch(
  readRDS(file.path(MODELS_DIR, "cox_model_nomogram_figure.rds")),
  error = function(e) { stop("cox_model_nomogram_figure.rds not found.") }
)

lp_t <- predict(cox_nom, newdata = df_t, type = "lp")
lp_v <- predict(cox_nom, newdata = df_v, type = "lp")
lp_a <- predict(cox_nom, newdata = df,   type = "lp")

cutpoint <- median(lp_t, na.rm = TRUE)
cat("  Median training LP (cutpoint):", round(cutpoint, 4), "\n")

cutpoint_note <- paste0(
  "Risk groups were defined by the median training-cohort linear predictor because ",
  "the paper did not specify the cutoff rule. The median (", round(cutpoint, 4),
  ") was computed from N=", nrow(df_t), " training patients and applied to all cohorts."
)
cutpoint_tbl <- data.frame(
  method     = "Median training linear predictor",
  cutpoint   = round(cutpoint, 6),
  n_training = nrow(df_t),
  note       = cutpoint_note,
  stringsAsFactors = FALSE
)
write_csv(cutpoint_tbl, file.path(TABLES_DIR, "risk_cutpoint_summary.csv"))
cat("  Saved risk_cutpoint_summary.csv\n")
log_method_note("KM risk groups", "Median training LP cutpoint", cutpoint_note)

assign_group <- function(lp) {
  ifelse(is.na(lp), NA_character_,
         ifelse(lp <= cutpoint, "Low-risk", "High-risk"))
}

df <- df |>
  mutate(
    lp           = lp_a,
    risk_group   = factor(assign_group(lp_a),
                          levels = c("Low-risk", "High-risk"))
  )

risk_assign_tbl <- df |>
  select(patient_id = any_of("patient_id"),
         cohort, risk_group,
         lp_value = lp,
         all_of(c(TIME_VAR, EVENT_VAR)))
write_csv(risk_assign_tbl, file.path(TABLES_DIR, "risk_group_assignment.csv"))
cat("  Saved risk_group_assignment.csv\n")
cat("  Risk group counts:\n")
print(table(df$cohort, df$risk_group, useNA = "ifany"))

GROUP_COLORS <- c("High-risk" = "#2166ac", "Low-risk" = "#d6604d")

make_risk_table <- function(km_fit, time_breaks, x_max) {
  risk_sum <- tryCatch(
    summary(km_fit, times = time_breaks, extend = TRUE),
    error = function(e) NULL
  )
  if (is.null(risk_sum)) return(NULL)

  risk_df <- data.frame(
    time       = risk_sum$time,
    n_risk     = risk_sum$n.risk,
    risk_group = gsub("risk_group=", "", as.character(risk_sum$strata)),
    stringsAsFactors = FALSE
  )
  risk_df$risk_group <- factor(risk_df$risk_group, levels = c("High-risk", "Low-risk"))

  ggplot(risk_df, aes(x = time, y = risk_group, label = n_risk)) +
    geom_text(size = 3.2, color = "black") +
    scale_x_continuous(limits = c(0, x_max),
                       breaks = time_breaks,
                       expand = expansion(mult = c(0.01, 0.02))) +
    scale_y_discrete(drop = FALSE) +
    labs(x = "Time (months)", y = NULL, title = "Number at risk") +
    theme_bw(base_size = 10) +
    theme(
      plot.title   = element_text(size = 9, face = "bold", hjust = 0),
      panel.grid   = element_blank(),
      axis.ticks.y = element_blank(),
      axis.text.y  = element_text(size = 9),
      axis.text.x  = element_text(size = 9)
    )
}

make_km_plot <- function(data, title, panel_label = NULL) {
  valid <- !is.na(data$risk_group)
  d     <- data[valid, ]
  if (nrow(d) < 10) return(NULL)
  display_title <- if (!is.null(panel_label)) paste0(panel_label, "   ", title) else title

  km_fit <- survfit(
    as.formula(paste0("Surv(", TIME_VAR, ", ", EVENT_VAR, ") ~ risk_group")),
    data = d
  )

  lr_test <- tryCatch(
    survdiff(as.formula(paste0("Surv(", TIME_VAR, ", ", EVENT_VAR,
                               ") ~ risk_group")), data = d),
    error = function(e) NULL
  )
  p_val <- if (!is.null(lr_test)) {
    pchisq(lr_test$chisq, df = 1, lower.tail = FALSE)
  } else NA_real_
  p_label <- if (!is.na(p_val)) {
    if (p_val < 0.001) "p < 0.001" else paste0("p = ", round(p_val, 3))
  } else ""

  strata_names <- gsub("risk_group=", "", names(km_fit$strata))
  km_df <- data.frame(
    time       = km_fit$time,
    surv       = km_fit$surv,
    lower      = km_fit$lower,
    upper      = km_fit$upper,
    risk_group = rep(strata_names, km_fit$strata),
    stringsAsFactors = FALSE
  )
  t0 <- data.frame(time = 0, surv = 1, lower = 1, upper = 1,
                   risk_group = strata_names, stringsAsFactors = FALSE)
  km_df      <- rbind(t0, km_df)
  km_df$risk_group <- factor(km_df$risk_group, levels = c("Low-risk", "High-risk"))
  km_df      <- km_df[order(km_df$risk_group, km_df$time), ]

  x_max       <- max(d[[TIME_VAR]], na.rm = TRUE)
  x_plot_max  <- ceiling(x_max / 25) * 25
  time_breaks <- seq(0, x_plot_max, by = 25)

  p_main <- ggplot(km_df, aes(x = time, y = surv,
                               color = risk_group, fill = risk_group)) +
    geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.15, color = NA) +
    geom_step(linewidth = 0.9) +
    scale_color_manual(values = GROUP_COLORS, name = NULL) +
    scale_fill_manual( values = GROUP_COLORS, name = NULL) +
    scale_y_continuous(limits = c(0, 1),
                       labels = scales::percent_format(1),
                       expand = expansion(mult = c(0, 0.02))) +
    scale_x_continuous(limits = c(0, x_plot_max),
                       breaks = time_breaks,
                       expand = expansion(mult = c(0.01, 0.02))) +
    annotate("text", x = x_plot_max * 0.62, y = 0.92,
             label = p_label, size = 3.5, color = "black") +
    labs(title = display_title, x = NULL, y = "Survival probability") +
    theme_bw(base_size = 11) +
    theme(
      plot.title        = element_text(face = "bold"),
      legend.position   = c(0.83, 0.85),
      legend.background = element_rect(fill = "white", color = NA),
      legend.text       = element_text(size = 9),
      axis.text.x       = element_blank(),
      axis.ticks.x      = element_blank()
    )

  p_risk <- make_risk_table(km_fit, time_breaks, x_plot_max)

  combined <- if (!is.null(p_risk)) {
    p_main / p_risk + plot_layout(heights = c(4, 1.5))
  } else {
    p_main
  }

  med_tbl <- tryCatch({
    med_obj <- summary(km_fit)$table
    data.frame(
      risk_group = gsub("risk_group=", "", rownames(med_obj)),
      n_patients = as.integer(med_obj[, "records"]),
      n_events   = as.integer(med_obj[, "events"]),
      median_os  = as.numeric(med_obj[, "median"]),
      stringsAsFactors = FALSE
    )
  }, error = function(e) NULL)

  list(plot = combined, logrank_p = p_val, n = nrow(d), median_tbl = med_tbl)
}

cohort_defs   <- list(
  Training              = df[df$cohort == "Training", ],
  `Internal validation` = df[df$cohort == "Internal validation", ]
)
cohort_keys   <- c("training", "internal_validation")
cohort_labels <- c("A", "B")

km_results   <- list()
logrank_rows <- list()
summary_rows <- list()

for (i in seq_along(cohort_defs)) {
  cn  <- names(cohort_defs)[i]
  ck  <- cohort_keys[i]
  res <- make_km_plot(cohort_defs[[cn]], title = cn, panel_label = cohort_labels[i])
  if (is.null(res)) next

  km_results[[ck]]   <- res$plot
  logrank_rows[[ck]] <- data.frame(
    cohort     = cn,
    cohort_key = ck,
    n          = res$n,
    logrank_p  = round(res$logrank_p, 5),
    stringsAsFactors = FALSE
  )

  if (!is.null(res$median_tbl)) {
    med          <- res$median_tbl
    med$cohort   <- cn
    med$logrank_p <- round(res$logrank_p, 5)
    med$cutpoint  <- round(cutpoint, 4)
    summary_rows[[ck]] <- med
  }

  cat("  ", cn, "| p =", round(res$logrank_p, 4), "\n")
}

logrank_tbl <- bind_rows(logrank_rows)
write_csv(logrank_tbl, file.path(TABLES_DIR, "km_logrank_results.csv"))
cat("  Saved km_logrank_results.csv\n")
print(logrank_tbl)

km_summary <- bind_rows(summary_rows) |>
  select(cohort, risk_group, n_patients, n_events, median_os, logrank_p, cutpoint)
write_csv(km_summary, file.path(TABLES_DIR, "km_risk_group_summary.csv"))
cat("  Saved km_risk_group_summary.csv\n")
print(km_summary)

# ---- fig8_km_risk_groups: Training + Internal validation, 2-panel ------------
panel_order <- c("training", "internal_validation")
panels      <- Filter(Negate(is.null), km_results[panel_order])

if (length(panels) > 0) {
  n_panels    <- length(panels)
  combined_km <- wrap_plots(panels, ncol = n_panels)
  ggsave(file.path(FIGURES_DIR, "fig8_km_risk_groups.png"),
         combined_km, width = 8 * n_panels, height = 7, dpi = 150)
  ggsave(file.path(FIGURES_DIR, "fig8_km_risk_groups.pdf"),
         combined_km, width = 8 * n_panels, height = 7)
  cat("  Saved fig8_km_risk_groups\n")
}

cat("=== 10 complete ===\n\n")
