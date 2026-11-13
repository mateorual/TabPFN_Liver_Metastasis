# scripts/02_time_dependent_roc.R
# Time-dependent ROC curves for 1-, 2-, and 3-year OS -> fig6-style 6-panel
# figure (embedding Cox risk score vs AJCC T/N baseline, Training + Internal
# validation x 1/2/3-year). Uses timeROC on the risk scores from script 01.

cat("=== 02_time_dependent_roc.R ===\n")

source("config/config.R")

for (pkg in c("survival", "ggplot2", "dplyr", "readr", "patchwork")) {
  if (!requireNamespace(pkg, quietly = TRUE))
    stop("Install '", pkg, "' with: install.packages('", pkg, "')")
}
suppressPackageStartupMessages({
  library(survival); library(ggplot2); library(dplyr)
  library(readr);    library(patchwork)
})

use_timeROC <- requireNamespace("timeROC", quietly = TRUE)
if (!use_timeROC) stop("Install timeROC: install.packages('timeROC')")
suppressPackageStartupMessages(library(timeROC))

set.seed(SEED)

risk_tbl <- read_csv(file.path(TABLES_DIR, "embedding_prognostic_risk_scores.csv"),
                     show_col_types = FALSE)

model_colors <- c(embedding_model = "#2ca02c", tn_baseline = "#9467bd")
model_labels <- c(embedding_model = "TabPFN embedding Cox (1024-dim, ridge)",
                  tn_baseline    = "AJCC T/N baseline (2 vars)")

compute_troc <- function(time_var, event_var, marker, t_star) {
  tryCatch({
    out <- timeROC::timeROC(T = time_var, delta = event_var, marker = marker,
                            cause = 1, times = t_star, iid = TRUE, weighting = "marginal")
    t_idx  <- which(out$times == t_star)
    auc    <- out$AUC[t_idx]
    se_auc <- tryCatch(out$inference$vect_sd_1[[t_idx]], error = function(e) NA_real_)
    ci     <- if (!is.na(se_auc) && se_auc > 0) c(auc - 1.96 * se_auc, auc + 1.96 * se_auc) else c(NA, NA)
    list(auc = auc, auc_lower = ci[1], auc_upper = ci[2],
         sensitivity = out$TP[, t_idx], one_minus_specificity = out$FP[, t_idx])
  }, error = function(e) { cat("  [warn] timeROC failed at t=", t_star, ":", conditionMessage(e), "\n"); NULL })
}

fmt_auc_ci <- function(auc, lo, hi) {
  if (!is.na(lo) && !is.na(hi)) paste0(round(auc, 3), " (95% CI: ", round(lo, 3), "–", round(hi, 3), ")")
  else as.character(round(auc, 3))
}

cohorts <- list(Training = risk_tbl |> filter(cohort == "Training"),
               `Internal validation` = risk_tbl |> filter(cohort == "Internal validation"))

auc_rows <- list()
roc_plots <- list()

for (cohort_name in names(cohorts)) {
  coh <- cohorts[[cohort_name]]
  coh_key <- gsub(" ", "_", tolower(cohort_name))

  for (hi in seq_along(TIME_HORIZONS_MONTHS)) {
    t_star <- TIME_HORIZONS_MONTHS[hi]
    t_lbl  <- TIME_HORIZONS_LABELS[hi]

    roc_emb <- compute_troc(coh$time_months, coh$event, coh$embedding_lp, t_star)
    roc_tn  <- compute_troc(coh$time_months, coh$event, coh$tn_baseline_lp, t_star)

    for (nm in c("embedding_model", "tn_baseline")) {
      r <- if (nm == "embedding_model") roc_emb else roc_tn
      if (is.null(r)) next
      auc_rows[[paste(cohort_name, t_lbl, nm)]] <- data.frame(
        cohort = cohort_name, time_label = t_lbl, time_months = t_star, model = nm,
        auc = round(r$auc, 4), auc_lower_95 = round(r$auc_lower, 4), auc_upper_95 = round(r$auc_upper, 4),
        n = nrow(coh), n_events = sum(coh$event[coh$time_months <= t_star & coh$event == 1]),
        stringsAsFactors = FALSE
      )
    }

    roc_df_panel <- bind_rows(
      if (!is.null(roc_emb)) data.frame(model = "embedding_model", x = roc_emb$one_minus_specificity, y = roc_emb$sensitivity),
      if (!is.null(roc_tn))  data.frame(model = "tn_baseline",     x = roc_tn$one_minus_specificity,  y = roc_tn$sensitivity)
    )
    if (nrow(roc_df_panel) == 0) next

    lbl_emb <- if (!is.null(roc_emb)) paste0(model_labels["embedding_model"], "\n  AUC = ",
                                              fmt_auc_ci(roc_emb$auc, roc_emb$auc_lower, roc_emb$auc_upper)) else model_labels["embedding_model"]
    lbl_tn  <- if (!is.null(roc_tn))  paste0(model_labels["tn_baseline"], "\n  AUC = ",
                                              fmt_auc_ci(roc_tn$auc, roc_tn$auc_lower, roc_tn$auc_upper)) else model_labels["tn_baseline"]

    p_panel <- ggplot(roc_df_panel, aes(x = x, y = y, color = model)) +
      geom_line(linewidth = 1.1) +
      geom_abline(slope = 1, intercept = 0, color = "grey60", linetype = "dashed", linewidth = 0.5) +
      scale_color_manual(
        values = model_colors[unique(roc_df_panel$model)],
        labels = c(embedding_model = lbl_emb, tn_baseline = lbl_tn)[unique(roc_df_panel$model)],
        name = NULL, guide = guide_legend(ncol = 1)
      ) +
      coord_equal() +
      labs(title = paste0(t_lbl, " ROC - ", cohort_name), subtitle = paste0("N=", nrow(coh)),
           x = "1 - Specificity (FPR)", y = "Sensitivity (TPR)") +
      theme_bw(base_size = 10) +
      theme(plot.title = element_text(face = "bold", size = 10),
            plot.subtitle = element_text(size = 8, color = "grey40"),
            legend.position = "bottom", legend.text = element_text(size = 7),
            legend.key.size = unit(0.4, "cm"))

    roc_plots[[paste0(coh_key, "_", gsub("-year", "yr", t_lbl))]] <- p_panel
  }
}

auc_df <- bind_rows(auc_rows)
write_csv(auc_df, file.path(TABLES_DIR, "time_dependent_roc_auc_embedding_prognostic.csv"))
cat("  AUC summary:\n"); print(auc_df |> select(cohort, time_label, model, auc))

all_panels <- c(roc_plots[paste0("training_", c("1yr", "2yr", "3yr"))],
               roc_plots[paste0("internal_validation_", c("1yr", "2yr", "3yr"))])
all_panels <- Filter(Negate(is.null), all_panels)
if (length(all_panels) > 0) {
  combined6 <- wrap_plots(all_panels, ncol = 3) +
    plot_annotation(tag_levels = "A", theme = theme(plot.tag = element_text(face = "bold", size = 13)))
  ggsave(file.path(FIGURES_DIR, "fig6_roc_training_internal_6panel.png"), combined6, width = 18, height = 11, dpi = 150)
  ggsave(file.path(FIGURES_DIR, "fig6_roc_training_internal_6panel.pdf"), combined6, width = 18, height = 11)
  cat("  Saved fig6_roc_training_internal_6panel\n")
}

cat("=== 02 complete ===\n\n")
