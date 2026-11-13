# scripts/09_survival_dca.R
# Decision curve analysis for survival prediction at 1-, 2-, and 3-year OS
# -> fig7_dca_training_internal_6panel.
#
# Method: manual net benefit with complete-case binary outcome at each horizon.
#   risk_t = 1 - predicted_survival_at_t
#   Binary outcome: event occurred at or before t (patients censored before t
#   are excluded for that horizon). Documented as an approximation — IPCW was
#   not used since the paper does not specify the exact DCA method.
#
# Compares nomogram_figure vs AJCC T/N baseline vs Treat All vs Treat None,
# across Training and Internal validation cohorts.

cat("=== 09_survival_dca.R ===\n")

source("config/config.R")

for (pkg in c("survival", "rms", "ggplot2", "dplyr", "readr", "patchwork")) {
  if (!requireNamespace(pkg, quietly = TRUE))
    stop("Install '", pkg, "' with: install.packages('", pkg, "')")
}
suppressPackageStartupMessages({
  library(survival); library(rms); library(ggplot2)
  library(dplyr); library(readr); library(patchwork)
})

set.seed(SEED)

df   <- readRDS(file.path(MODELS_DIR, "prognostic_data_clean.rds"))
df_t <- df[df$cohort == "Training", ]
df_v <- df[df$cohort == "Internal validation", ]

cox_nom <- tryCatch(readRDS(file.path(MODELS_DIR, "cox_model_nomogram_figure.rds")),
                    error = function(e) NULL)
cox_tn  <- tryCatch(readRDS(file.path(MODELS_DIR, "cox_model_tn_baseline.rds")),
                    error = function(e) NULL)
if (is.null(cox_nom)) stop("cox_model_nomogram_figure.rds not found.")
if (is.null(cox_tn))  stop("cox_model_tn_baseline.rds not found.")

cph_fit <- tryCatch(readRDS(file.path(MODELS_DIR, "prognostic_nomogram_cph_model.rds")),
                    error = function(e) NULL)

get_pred_surv <- function(cox_fit, cph_fit, newdata, t_star) {
  if (!is.null(cph_fit)) {
    tryCatch({
      present <- intersect(FEATURES_NOMOGRAM_FIGURE, names(newdata))
      df_sub  <- newdata[, c(present, TIME_VAR, EVENT_VAR), drop = FALSE]
      df_sub  <- df_sub[complete.cases(df_sub), , drop = FALSE]
      df_sub  <- as.data.frame(df_sub) |>
        dplyr::mutate(dplyr::across(where(is.ordered),
                                    ~ factor(.x, ordered = FALSE)))
      sf_obj  <- rms::survest(cph_fit, newdata = df_sub, times = t_star)
      list(surv = as.numeric(if (is.list(sf_obj)) sf_obj$surv else sf_obj),
           n    = nrow(df_sub))
    }, error = function(e) NULL)
  } else NULL
}

get_pred_surv_cox <- function(cox_fit, newdata, t_star) {
  tryCatch({
    sf  <- survfit(cox_fit, newdata = newdata)
    idx <- findInterval(t_star, sf$time)
    s   <- if (idx == 0) rep(1.0, nrow(newdata)) else as.numeric(sf$surv[idx, ])
    list(surv = s, n = nrow(newdata))
  }, error = function(e) NULL)
}

thresholds <- seq(0.01, 0.99, by = 0.01)

compute_nb_binary <- function(y_bin, risk, thresholds) {
  valid <- !is.na(risk) & !is.na(y_bin)
  y_v   <- y_bin[valid]; p_v <- risk[valid]; n_v <- length(y_v)
  sapply(thresholds, function(t) {
    tp <- sum(p_v >= t & y_v == 1)
    fp <- sum(p_v >= t & y_v == 0)
    tp / n_v - fp / n_v * (t / (1 - t))
  })
}

model_colors    <- c(nomogram_figure = "#ff7f0e", tn_baseline = "#9467bd",
                     "Treat All" = "#333333", "Treat None" = "#999999")
model_linetypes <- c(nomogram_figure = "solid",  tn_baseline = "solid",
                     "Treat All" = "dashed",  "Treat None" = "dotted")
nom_lbl <- model_feature_label("Nomogram", FEATURES_NOMOGRAM_FIGURE)
tn_lbl  <- model_feature_label("AJCC T/N baseline", FEATURES_TN_BASELINE)
display_labels <- c(nomogram_figure = nom_lbl, tn_baseline = tn_lbl,
                    "Treat All" = "Treat All", "Treat None" = "Treat None")

all_dca_rows  <- list()
dca_plots     <- list()

cohorts <- list(
  Training = list(df = df_t, key = "training"),
  `Internal validation` = list(df = df_v, key = "internal_validation")
)

for (cohort_name in names(cohorts)) {
  coh     <- cohorts[[cohort_name]]
  coh_key <- coh$key
  coh_df  <- coh$df

  for (hi in seq_along(TIME_HORIZONS_MONTHS)) {
    t_star <- TIME_HORIZONS_MONTHS[hi]
    t_lbl  <- TIME_HORIZONS_LABELS[hi]

    # Exclude patients censored before t_star
    tv    <- as.numeric(coh_df[[TIME_VAR]])
    ev    <- as.integer(coh_df[[EVENT_VAR]])
    eligible <- (tv >= t_star) | (ev == 1 & tv < t_star)
    df_elig  <- coh_df[eligible, ]
    tv_e     <- tv[eligible]; ev_e <- ev[eligible]
    y_bin    <- as.integer(ev_e == 1 & tv_e <= t_star)
    prev     <- mean(y_bin)
    n_elig   <- nrow(df_elig)

    cat("  ", cohort_name, t_lbl, ": N=", n_elig,
        "| events=", sum(y_bin), "\n")
    if (n_elig < 20 || sum(y_bin) < 5) {
      cat("  [skip] Insufficient events.\n"); next
    }

    rows_this <- list()

    rows_this[["Treat All"]]  <- data.frame(
      threshold = thresholds,
      net_benefit = prev - (1 - prev) * (thresholds / (1 - thresholds)),
      strategy = "Treat All", stringsAsFactors = FALSE
    )
    rows_this[["Treat None"]] <- data.frame(
      threshold = thresholds, net_benefit = 0,
      strategy = "Treat None", stringsAsFactors = FALSE
    )

    for (nm in c("nomogram_figure", "tn_baseline")) {
      cx_fit  <- if (nm == "nomogram_figure") cox_nom else cox_tn
      cph_obj <- if (nm == "nomogram_figure") cph_fit else NULL

      surv_res <- if (!is.null(cph_obj)) {
        get_pred_surv(cx_fit, cph_obj, df_elig, t_star)
      } else NULL
      if (is.null(surv_res)) {
        surv_res <- get_pred_surv_cox(cx_fit, df_elig, t_star)
      }
      if (is.null(surv_res)) next

      risk_t <- 1 - surv_res$surv
      if (length(risk_t) != n_elig) risk_t <- rep(NA_real_, n_elig)

      nb_vals <- compute_nb_binary(y_bin, risk_t, thresholds)
      rows_this[[nm]] <- data.frame(
        threshold = thresholds, net_benefit = nb_vals,
        strategy = nm, stringsAsFactors = FALSE
      )
    }

    dca_df <- bind_rows(rows_this)
    dca_df$cohort      <- cohort_name
    dca_df$time_label  <- t_lbl
    dca_df$time_months <- t_star

    all_dca_rows[[paste(coh_key, t_lbl)]] <- dca_df

    # Build panel (kept in memory for the fig7 6-panel assembly below)
    keys_plot <- intersect(c("nomogram_figure", "tn_baseline", "Treat All", "Treat None"),
                           unique(dca_df$strategy))
    sub_dca   <- dca_df |>
      filter(strategy %in% keys_plot) |>
      mutate(strategy = factor(strategy, levels = keys_plot))

    max_nb  <- max(sub_dca$net_benefit[sub_dca$strategy != "Treat None"],
                   na.rm = TRUE)
    y_upper <- max(0.05, ceiling(max_nb * 110) / 100)

    p_dca <- ggplot(sub_dca,
                    aes(x = threshold, y = net_benefit,
                        color = strategy, linetype = strategy)) +
      geom_line(linewidth = 1.0) +
      scale_color_manual(values    = model_colors[keys_plot],
                         labels    = display_labels[keys_plot],
                         name      = NULL,
                         guide     = guide_legend(ncol = 1)) +
      scale_linetype_manual(values = model_linetypes[keys_plot],
                            labels = display_labels[keys_plot],
                            name   = NULL,
                            guide  = guide_legend(ncol = 1)) +
      coord_cartesian(ylim = c(-0.02, y_upper)) +
      labs(
        title    = paste0(t_lbl, " DCA - ", cohort_name),
        subtitle = paste0("N=", n_elig, " | Events=", sum(y_bin)),
        x = "Threshold Probability",
        y = "Net Benefit"
      ) +
      theme_bw(base_size = 10) +
      theme(plot.title      = element_text(face = "bold", size = 10),
            plot.subtitle   = element_text(size = 8, color = "grey40"),
            legend.position = "bottom",
            legend.text     = element_text(size = 7),
            legend.key.size = unit(0.4, "cm"))

    dca_plots[[paste0(coh_key, "_", gsub("-year", "yr", t_lbl))]] <- p_dca
  }
}

dca_all <- bind_rows(all_dca_rows)
write_csv(dca_all, file.path(TABLES_DIR, "dca_survival_results.csv"))
cat("  Saved dca_survival_results.csv\n")

legend_audit <- data.frame(
  model_key   = c("nomogram_figure", "tn_baseline", "Treat All", "Treat None"),
  display_label = unname(display_labels[c("nomogram_figure", "tn_baseline",
                                          "Treat All", "Treat None")]),
  stringsAsFactors = FALSE
)
write_csv(legend_audit, file.path(TABLES_DIR, "dca_survival_legend_audit.csv"))
cat("  Saved dca_survival_legend_audit.csv\n")

# ---- fig7_dca_training_internal_6panel: Training (A-C) + Validation (D-F) ----
all_dca_plots_6 <- c(
  dca_plots[paste0("training_",            c("1yr", "2yr", "3yr"))],
  dca_plots[paste0("internal_validation_", c("1yr", "2yr", "3yr"))]
)
all_dca_plots_6 <- Filter(Negate(is.null), all_dca_plots_6)
if (length(all_dca_plots_6) > 0) {
  combined6 <- wrap_plots(all_dca_plots_6, ncol = 3) +
    plot_annotation(
      tag_levels = "A",
      theme      = theme(plot.tag = element_text(face = "bold", size = 13))
    )
  ggsave(file.path(FIGURES_DIR, "fig7_dca_training_internal_6panel.png"),
         combined6, width = 18, height = 10, dpi = 150)
  ggsave(file.path(FIGURES_DIR, "fig7_dca_training_internal_6panel.pdf"),
         combined6, width = 18, height = 10)
  cat("  Saved fig7_dca_training_internal_6panel\n")
}

log_method_note(
  "Survival DCA",
  "Complete-case binary outcome at horizon",
  paste0("Survival DCA approximated by converting survival predictions into ",
         "horizon-specific mortality risk (risk_t = 1 - S_hat(t)) and excluding ",
         "patients censored before the horizon. IPCW was not used. ",
         "This is an approximation; the paper does not specify the exact DCA method.")
)

cat("=== 09 complete ===\n\n")
