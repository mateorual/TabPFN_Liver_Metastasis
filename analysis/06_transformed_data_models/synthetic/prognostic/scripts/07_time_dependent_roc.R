# scripts/07_time_dependent_roc.R
# Time-dependent ROC curves for 1-, 2-, and 3-year OS -> fig6_roc_training_internal_6panel.
# Compares nomogram_figure vs AJCC T/N baseline, across Training and Internal
# validation cohorts (3 horizons x 2 cohorts = 6 panels).
# Uses timeROC package (preferred) or survivalROC as fallback.

cat("=== 07_time_dependent_roc.R ===\n")

source("config/config.R")

for (pkg in c("survival", "ggplot2", "dplyr", "readr", "patchwork")) {
  if (!requireNamespace(pkg, quietly = TRUE))
    stop("Install '", pkg, "' with: install.packages('", pkg, "')")
}
suppressPackageStartupMessages({
  library(survival); library(ggplot2); library(dplyr)
  library(readr);    library(patchwork)
})

set.seed(SEED)

df   <- readRDS(file.path(MODELS_DIR, "prognostic_data_clean.rds"))
df_t <- df[df$cohort == "Training", ]
df_v <- df[df$cohort == "Internal validation", ]
cat("  Training N =", nrow(df_t), "| Validation N =", nrow(df_v), "\n")

load_cox <- function(name) {
  p <- file.path(MODELS_DIR, paste0("cox_model_", name, ".rds"))
  if (file.exists(p)) readRDS(p) else NULL
}
cox_nom <- load_cox("nomogram_figure")
cox_tn  <- load_cox("tn_baseline")

if (is.null(cox_nom)) stop("cox_model_nomogram_figure.rds not found.")
if (is.null(cox_tn))  stop("cox_model_tn_baseline.rds not found.")

lp_t_nom <- predict(cox_nom, newdata = df_t, type = "lp")
lp_v_nom <- predict(cox_nom, newdata = df_v, type = "lp")
lp_t_tn  <- predict(cox_tn,  newdata = df_t, type = "lp")
lp_v_tn  <- predict(cox_tn,  newdata = df_v, type = "lp")

use_timeROC      <- requireNamespace("timeROC", quietly = TRUE)
use_survivalROC  <- !use_timeROC && requireNamespace("survivalROC", quietly = TRUE)

if (!use_timeROC && !use_survivalROC) {
  stop("Install timeROC or survivalROC: install.packages('timeROC')")
}
if (use_timeROC)     { suppressPackageStartupMessages(library(timeROC)) }
if (use_survivalROC) { suppressPackageStartupMessages(library(survivalROC)) }

# ---- Time-dependent AUC/ROC with 95% CI ---------------------------------------
compute_troc <- function(time_var, event_var, marker, data, t_star) {
  tv <- as.numeric(data[[time_var]])
  ev <- as.integer(data[[event_var]])
  m  <- as.numeric(marker)

  if (use_timeROC) {
    tryCatch({
      out <- timeROC::timeROC(
        T         = tv,
        delta     = ev,
        marker    = m,
        cause     = 1,
        times     = t_star,
        iid       = TRUE,
        weighting = "marginal"
      )
      t_idx   <- which(out$times == t_star)
      auc_val <- out$AUC[t_idx]
      # 95% CI from iid-based SE
      se_auc <- tryCatch(out$inference$vect_sd_1[[t_idx]], error = function(e) NA_real_)
      ci_res <- if (!is.na(se_auc) && se_auc > 0) {
        c(auc_val - 1.96 * se_auc, auc_val + 1.96 * se_auc)
      } else {
        c(NA_real_, NA_real_)
      }
      sens   <- out$TP[, t_idx]
      spec_1 <- out$FP[, t_idx]
      list(auc = auc_val, auc_lower = ci_res[1], auc_upper = ci_res[2],
           sensitivity = sens, one_minus_specificity = spec_1)
    }, error = function(e) {
      cat("  [warn] timeROC failed at t=", t_star, ":", conditionMessage(e), "\n")
      NULL
    })
  } else {
    tryCatch({
      out <- survivalROC::survivalROC(
        Stime = tv, status = ev, marker = m, predict.time = t_star, method = "KM"
      )
      list(auc = out$AUC, auc_lower = NA_real_, auc_upper = NA_real_,
           sensitivity = out$TP, one_minus_specificity = out$FP)
    }, error = function(e) {
      cat("  [warn] survivalROC failed at t=", t_star, ":", conditionMessage(e), "\n")
      NULL
    })
  }
}

model_colors <- c(nomogram_figure = "#ff7f0e", tn_baseline = "#9467bd")
nom_label    <- model_feature_label("Nomogram", FEATURES_NOMOGRAM_FIGURE)
tn_label     <- model_feature_label("AJCC T/N baseline", FEATURES_TN_BASELINE)

fmt_auc_ci <- function(auc, lo, hi) {
  if (!is.na(lo) && !is.na(hi))
    paste0(auc, " (95% CI: ", lo, "–", hi, ")")
  else as.character(auc)
}

cohorts <- list(Training = list(df = df_t, lp_nom = lp_t_nom, lp_tn = lp_t_tn),
               `Internal validation` = list(df = df_v, lp_nom = lp_v_nom, lp_tn = lp_v_tn))

auc_rows   <- list()
curve_rows <- list()
roc_plots  <- list()

for (cohort_name in names(cohorts)) {
  coh     <- cohorts[[cohort_name]]
  coh_key <- gsub(" ", "_", tolower(cohort_name))

  for (hi in seq_along(TIME_HORIZONS_MONTHS)) {
    t_star <- TIME_HORIZONS_MONTHS[hi]
    t_lbl  <- TIME_HORIZONS_LABELS[hi]

    roc_nom <- compute_troc(TIME_VAR, EVENT_VAR, coh$lp_nom, coh$df, t_star)
    roc_tn  <- compute_troc(TIME_VAR, EVENT_VAR, coh$lp_tn,  coh$df, t_star)

    for (nm in c("nomogram_figure", "tn_baseline")) {
      r     <- if (nm == "nomogram_figure") roc_nom else roc_tn
      if (is.null(r)) next

      auc_rows[[paste(cohort_name, t_lbl, nm)]] <- data.frame(
        cohort       = cohort_name,
        time_label   = t_lbl,
        time_months  = t_star,
        model        = nm,
        auc          = round(r$auc, 4),
        auc_lower_95 = round(if (!is.null(r$auc_lower)) r$auc_lower else NA_real_, 4),
        auc_upper_95 = round(if (!is.null(r$auc_upper)) r$auc_upper else NA_real_, 4),
        n            = nrow(coh$df),
        n_events     = sum(coh$df[[EVENT_VAR]][coh$df[[TIME_VAR]] <= t_star &
                                               coh$df[[EVENT_VAR]] == 1]),
        stringsAsFactors = FALSE
      )

      curve_rows[[paste(cohort_name, t_lbl, nm)]] <- data.frame(
        cohort      = cohort_name,
        time_label  = t_lbl,
        time_months = t_star,
        model       = nm,
        sensitivity        = r$sensitivity,
        one_minus_specificity = r$one_minus_specificity,
        stringsAsFactors   = FALSE
      )
    }

    # Build single panel (kept in memory for the fig6 6-panel assembly below)
    roc_df_panel <- bind_rows(
      if (!is.null(roc_nom)) data.frame(
        model = "nomogram_figure",
        x = roc_nom$one_minus_specificity, y = roc_nom$sensitivity) else NULL,
      if (!is.null(roc_tn)) data.frame(
        model = "tn_baseline",
        x = roc_tn$one_minus_specificity,  y = roc_tn$sensitivity)  else NULL
    )

    if (nrow(roc_df_panel) == 0) next

    auc_nom_val <- if (!is.null(roc_nom)) round(roc_nom$auc, 3) else NA
    auc_tn_val  <- if (!is.null(roc_tn))  round(roc_tn$auc,  3) else NA
    auc_nom_lo  <- if (!is.null(roc_nom) && !is.null(roc_nom$auc_lower)) round(roc_nom$auc_lower, 3) else NA_real_
    auc_nom_hi  <- if (!is.null(roc_nom) && !is.null(roc_nom$auc_upper)) round(roc_nom$auc_upper, 3) else NA_real_
    auc_tn_lo   <- if (!is.null(roc_tn)  && !is.null(roc_tn$auc_lower))  round(roc_tn$auc_lower,  3) else NA_real_
    auc_tn_hi   <- if (!is.null(roc_tn)  && !is.null(roc_tn$auc_upper))  round(roc_tn$auc_upper,  3) else NA_real_

    panel_lbl_nom <- paste0(nom_label, "\n  AUC = ", fmt_auc_ci(auc_nom_val, auc_nom_lo, auc_nom_hi))
    panel_lbl_tn  <- paste0(tn_label,  "\n  AUC = ", fmt_auc_ci(auc_tn_val,  auc_tn_lo,  auc_tn_hi))

    p_panel <- ggplot(roc_df_panel, aes(x = x, y = y, color = model)) +
      geom_line(linewidth = 1.1) +
      geom_abline(slope = 1, intercept = 0, color = "grey60",
                  linetype = "dashed", linewidth = 0.5) +
      scale_color_manual(
        values = model_colors[unique(roc_df_panel$model)],
        labels = c(nomogram_figure = panel_lbl_nom, tn_baseline = panel_lbl_tn)[
          unique(roc_df_panel$model)],
        name = NULL,
        guide = guide_legend(ncol = 1)
      ) +
      coord_equal() +
      labs(
        title    = paste0(t_lbl, " ROC - ", cohort_name),
        subtitle = paste0("N=", nrow(coh$df)),
        x = "1 - Specificity (FPR)", y = "Sensitivity (TPR)"
      ) +
      theme_bw(base_size = 10) +
      theme(plot.title      = element_text(face = "bold", size = 10),
            plot.subtitle   = element_text(size = 8, color = "grey40"),
            legend.position = "bottom",
            legend.text     = element_text(size = 7),
            legend.key.size = unit(0.4, "cm"))

    plot_key <- paste0(coh_key, "_", gsub("-year", "yr", t_lbl))
    roc_plots[[plot_key]] <- p_panel
  }
}

auc_df   <- bind_rows(auc_rows)
curve_df <- bind_rows(curve_rows)
write_csv(auc_df,   file.path(TABLES_DIR, "time_dependent_roc_auc.csv"))
write_csv(curve_df, file.path(TABLES_DIR, "time_dependent_roc_curve_data.csv"))
cat("  Saved time_dependent_roc_auc.csv / _curve_data.csv\n")
cat("  AUC summary:\n")
print(auc_df |> select(cohort, time_label, model, auc))

legend_audit <- data.frame(
  model_key   = c("nomogram_figure", "tn_baseline"),
  model_label = c(nom_label, tn_label),
  stringsAsFactors = FALSE
)
write_csv(legend_audit, file.path(TABLES_DIR, "roc_model_legend_audit.csv"))
cat("  Saved roc_model_legend_audit.csv\n")

# ---- fig6_roc_training_internal_6panel: Training (A-C) + Validation (D-F) ----
all_panels <- c(
  roc_plots[paste0("training_", c("1yr", "2yr", "3yr"))],
  roc_plots[paste0("internal_validation_", c("1yr", "2yr", "3yr"))]
)
all_panels <- Filter(Negate(is.null), all_panels)
if (length(all_panels) > 0) {
  combined6 <- wrap_plots(all_panels, ncol = 3) +
    plot_annotation(
      tag_levels = "A",
      theme      = theme(plot.tag = element_text(face = "bold", size = 13))
    )
  ggsave(file.path(FIGURES_DIR, "fig6_roc_training_internal_6panel.png"),
         combined6, width = 18, height = 11, dpi = 150)
  ggsave(file.path(FIGURES_DIR, "fig6_roc_training_internal_6panel.pdf"),
         combined6, width = 18, height = 11)
  cat("  Saved fig6_roc_training_internal_6panel\n")
}

meth_note <- data.frame(
  component = "Time-dependent ROC",
  method    = if (use_timeROC) "timeROC (IPCW marginal)" else "survivalROC (KM method)",
  note      = paste0("Package: ", if (use_timeROC) "timeROC" else "survivalROC"),
  stringsAsFactors = FALSE
)
meth_path <- file.path(TABLES_DIR, "methodological_notes_prognostic.csv")
if (file.exists(meth_path)) {
  prev <- read.csv(meth_path, stringsAsFactors = FALSE)
  write.csv(rbind(prev, meth_note), meth_path, row.names = FALSE)
} else {
  write.csv(meth_note, meth_path, row.names = FALSE)
}

cat("=== 07 complete ===\n\n")
