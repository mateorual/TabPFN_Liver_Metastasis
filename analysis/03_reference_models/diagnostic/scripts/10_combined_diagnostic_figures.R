# scripts/10_combined_diagnostic_figures.R
# Combined three-panel Figure 3 for the pipeline-selected model:
#   A - ROC (pipeline_selected vs AJCC T/N baseline)
#   B - Calibration (pipeline_selected: Ideal / Apparent / Bias-corrected)
#   C - DCA (pipeline_selected + AJCC T/N baseline + Treat All + Treat None)
# Requires scripts 04 and 00 to have been run first.

cat("=== 10_combined_diagnostic_figures.R ===\n")

source("config/config.R")
source("scripts/helpers_diagnostic.R")

for (pkg in c("pROC", "rms", "ggplot2", "dplyr", "readr", "patchwork")) {
  if (!requireNamespace(pkg, quietly = TRUE))
    stop("Install '", pkg, "' with: install.packages('", pkg, "')")
}
suppressPackageStartupMessages({
  library(pROC); library(rms)
  library(ggplot2); library(dplyr); library(readr); library(patchwork)
})

set.seed(SEED)

df <- readRDS(file.path(MODELS_DIR, "diagnostic_data_clean.rds"))

ps_feats <- tryCatch(
  readRDS(file.path(MODELS_DIR, "pipeline_selected_features.rds")),
  error = function(e) { cat("  [warn] pipeline_selected_features.rds not found.\n"); character(0) }
)
if (length(ps_feats) == 0) {
  cat("  [STOP] No pipeline-selected features found. Run script 04 first.\n")
  quit(save = "no", status = 0)
}

ps_glm_path <- file.path(MODELS_DIR, "logistic_model_pipeline_selected.rds")
if (!file.exists(ps_glm_path)) {
  cat("  [STOP] logistic_model_pipeline_selected.rds not found. Run script 04 first.\n")
  quit(save = "no", status = 0)
}
ps_glm <- readRDS(ps_glm_path)
cat("  Pipeline-selected features (", length(ps_feats), "):",
    paste(ps_feats, collapse = ", "), "\n")

tnm_feats <- intersect(FEATURES_TNM_BASELINE, names(df))

common_vars <- unique(c(ps_feats, tnm_feats, OUTCOME))
common_vars <- intersect(common_vars, names(df))
df_common   <- df[complete.cases(df[, common_vars]), ]
n_common    <- nrow(df_common)
cat("  Common evaluation set (pipeline_selected + TNM): N =", n_common, "\n")

# AJCC T/N baseline: the comparator fitted and saved by script 04.
tnm_path <- file.path(MODELS_DIR, "logistic_model_tnm_baseline.rds")
if (!file.exists(tnm_path)) stop("logistic_model_tnm_baseline.rds not found. Run script 04 first.")
tnm_glm <- readRDS(tnm_path)

# ============================================================
# Panel A — ROC
# ============================================================
get_probs <- function(model, data) {
  tryCatch(as.numeric(predict(model, newdata = data, type = "response",
                              na.action = na.exclude)),
           error = function(e) NULL)
}

roc_models <- Filter(Negate(is.null), list(
  pipeline_selected = ps_glm,
  tnm_baseline      = tnm_glm
))

roc_list <- list()
auc_info <- list()

for (nm in names(roc_models)) {
  probs <- get_probs(roc_models[[nm]], df_common)
  if (is.null(probs)) next
  valid  <- !is.na(probs)
  roc_obj <- tryCatch(
    pROC::roc(df_common[[OUTCOME]][valid], probs[valid], direction = "<", quiet = TRUE),
    error = function(e) NULL
  )
  if (is.null(roc_obj)) next
  ci_obj <- tryCatch(
    pROC::ci.auc(roc_obj, method = "bootstrap", boot.n = 1000),
    error = function(e) pROC::ci.auc(roc_obj, method = "delong")
  )
  roc_list[[nm]] <- roc_obj
  auc_info[[nm]] <- list(
    auc  = round(as.numeric(pROC::auc(roc_obj)), 3),
    lo   = round(as.numeric(ci_obj[1]), 3),
    hi   = round(as.numeric(ci_obj[3]), 3)
  )
  cat("  ", nm, ": AUC =", auc_info[[nm]]$auc,
      "(", auc_info[[nm]]$lo, "-", auc_info[[nm]]$hi, ")\n")
}

roc_colors <- c(pipeline_selected = "#2ca02c", tnm_baseline = "#9467bd")

wrap_legend_label <- function(x, width = 42) {
  paste(strwrap(x, width = width), collapse = "\n")
}

roc_label <- function(nm) {
  ai  <- auc_info[[nm]]
  base <- if (nm == "pipeline_selected")
            model_display_label("Pipeline-selected", ps_feats, max_chars = 500)
          else
            model_display_label("AJCC T/N baseline", tnm_feats, max_chars = 500)
  wrap_legend_label(paste0(base, " | AUC=", ai$auc, " (", ai$lo, "–", ai$hi, ")"), width = 62)
}

roc_df <- bind_rows(lapply(names(roc_list), function(nm) {
  r <- roc_list[[nm]]
  data.frame(model_key   = nm,
             specificity = rev(r$specificities),
             sensitivity = rev(r$sensitivities),
             stringsAsFactors = FALSE)
}))

panel_a <- ggplot(roc_df,
                  aes(x = 1 - specificity, y = sensitivity, color = model_key)) +
  geom_line(linewidth = 1.1) +
  geom_abline(slope = 1, intercept = 0, color = "grey60",
              linetype = "dashed", linewidth = 0.5) +
  scale_color_manual(
    values = roc_colors[names(roc_list)],
    labels = sapply(names(roc_list), roc_label),
    name   = NULL,
    guide  = guide_legend(ncol = 1, byrow = TRUE)
  ) +
  coord_equal() +
  labs(
    tag      = "A",
    title    = "ROC",
    subtitle = paste0("Common evaluation set: N=", n_common),
    x        = "1 − Specificity (FPR)",
    y        = "Sensitivity (TPR)"
  ) +
  theme_bw(base_size = 11) +
  theme(plot.tag        = element_text(face = "bold", size = 14),
        plot.title      = element_text(face = "bold"),
        plot.subtitle   = element_text(size = 8.5, color = "grey40"),
        legend.position = "bottom",
        legend.text     = element_text(size = 8.2, lineheight = 0.95),
        legend.key.size = unit(0.5, "cm"),
        legend.key.width = unit(0.7, "cm"),
        plot.margin     = ggplot2::margin(5.5, 5.5, 5.5, 5.5))

# ============================================================
# Panel B — Calibration (pipeline-selected)
# ============================================================
panel_b <- tryCatch({
  ps_feats_present <- intersect(ps_feats, names(df))
  df_ps <- df |>
    select(all_of(c(ps_feats_present, OUTCOME))) |>
    na.omit()

  dd_ps <- rms::datadist(df_ps)
  options(datadist = "dd_ps")

  lrm_ps <- rms::lrm(
    as.formula(paste0(OUTCOME, " ~ ", paste(ps_feats_present, collapse = " + "))),
    data = df_ps, x = TRUE, y = TRUE
  )

  cat("  Running rms::calibrate() for pipeline_selected (B=1000) ...\n")
  cal_ps <- tryCatch(
    rms::calibrate(lrm_ps, method = "boot", B = 1000),
    error = function(e) {
      cat("  [warn] calibrate boot failed:", conditionMessage(e), "\n")
      NULL
    }
  )
  if (is.null(cal_ps)) stop("Calibration returned NULL.")

  cal_mat        <- unclass(cal_ps)
  available_cols <- colnames(cal_mat)
  pred_col <- intersect(c("predy", "mean.predicted"), available_cols)[1]
  obs_col  <- intersect(c("calibrated.orig", "calibrated.obs"), available_cols)[1]
  if (is.na(pred_col) || is.na(obs_col)) stop("Cannot identify calibration columns.")

  keep <- !is.na(cal_mat[, pred_col]) & !is.na(cal_mat[, obs_col])
  cal_df <- data.frame(
    mean_predicted = as.numeric(cal_mat[keep, pred_col]),
    apparent       = as.numeric(cal_mat[keep, obs_col])
  )
  has_bcorr <- "calibrated.corrected" %in% available_cols
  if (has_bcorr)
    cal_df$bias_corrected <- as.numeric(cal_mat[keep, "calibrated.corrected"])
  else
    cal_df$bias_corrected <- NA_real_

  curve_levels <- c("Ideal", "Apparent", "Bias-corrected")
  long_ideal   <- data.frame(mean_predicted = c(0, 1), observed = c(0, 1), curve = "Ideal")
  long_app     <- data.frame(mean_predicted = cal_df$mean_predicted,
                              observed = cal_df$apparent, curve = "Apparent")
  if (has_bcorr && !all(is.na(cal_df$bias_corrected))) {
    long_bc <- data.frame(mean_predicted = cal_df$mean_predicted,
                           observed = cal_df$bias_corrected, curve = "Bias-corrected")
    long_df <- bind_rows(long_ideal, long_app, long_bc)
  } else {
    long_df  <- bind_rows(long_ideal, long_app)
    curve_levels <- c("Ideal", "Apparent")
  }
  long_df$curve <- factor(long_df$curve, levels = curve_levels)

  color_map    <- c("Ideal" = "gray50",    "Apparent" = "steelblue",
                    "Bias-corrected" = "darkorange")
  linetype_map <- c("Ideal" = "dashed",    "Apparent" = "solid",
                    "Bias-corrected" = "dashed")
  linewidth_map <- c("Ideal" = 0.8,        "Apparent" = 1.2,
                     "Bias-corrected" = 1.0)

  ggplot(long_df,
         aes(x = mean_predicted, y = observed,
             color = curve, linetype = curve, linewidth = curve)) +
    geom_line() +
    geom_point(data = long_df |> filter(curve != "Ideal"),
               size = 1.8, show.legend = FALSE) +
    scale_color_manual(   values = color_map[levels(long_df$curve)],    name = NULL) +
    scale_linetype_manual(values = linetype_map[levels(long_df$curve)], name = NULL) +
    scale_linewidth_manual(values = linewidth_map[levels(long_df$curve)], name = NULL) +
    coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
    labs(
      tag      = "B",
      title    = "Calibration",
      subtitle = paste0("Model: pipeline_selected (", length(ps_feats), " vars)"),
      x        = "Predicted Pr{liver=1}",
      y        = "Actual Probability"
    ) +
    theme_bw(base_size = 11) +
    theme(plot.tag        = element_text(face = "bold", size = 14),
          plot.title      = element_text(face = "bold"),
          plot.subtitle   = element_text(size = 8.5, color = "grey40"),
          legend.position = "bottom",
          legend.text     = element_text(size = 9),
          legend.key.size = unit(0.5, "cm")) +
    guides(linewidth = "none")
}, error = function(e) {
  cat("  [warn] Panel B calibration failed:", conditionMessage(e), "\n")
  NULL
})

# ============================================================
# Panel C — DCA
# ============================================================
thresholds    <- seq(0.01, 0.99, by = 0.01)
prevalence    <- mean(df_common[[OUTCOME]], na.rm = TRUE)

compute_nb <- function(y, probs, thresholds) {
  valid <- !is.na(probs)
  y_v   <- y[valid]; p_v <- as.numeric(probs)[valid]; n_v <- length(y_v)
  sapply(thresholds, function(t) {
    tp <- sum(p_v >= t & y_v == 1); fp <- sum(p_v >= t & y_v == 0)
    tp / n_v - fp / n_v * (t / (1 - t))
  })
}

treat_all_nb <- prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))

dca_rows <- list(
  "Treat All"  = data.frame(threshold = thresholds, net_benefit = treat_all_nb,
                             strategy = "Treat All"),
  "Treat None" = data.frame(threshold = thresholds, net_benefit = 0,
                             strategy = "Treat None")
)

for (nm in names(roc_models)) {
  probs <- get_probs(roc_models[[nm]], df_common)
  if (is.null(probs)) next
  nb_vals <- compute_nb(df_common[[OUTCOME]], probs, thresholds)
  dca_rows[[nm]] <- data.frame(threshold = thresholds, net_benefit = nb_vals, strategy = nm)
}

dca_df <- bind_rows(dca_rows)
dca_keys <- intersect(
  c("pipeline_selected", "tnm_baseline", "Treat All", "Treat None"),
  unique(dca_df$strategy)
)

dca_colors <- c(
  pipeline_selected = "#2ca02c",
  tnm_baseline      = "#9467bd",
  "Treat All"       = "#333333",
  "Treat None"      = "#999999"
)
dca_linetypes <- c(
  pipeline_selected = "solid",
  tnm_baseline      = "solid",
  "Treat All"       = "dashed",
  "Treat None"      = "dotted"
)
dca_display_lbls <- c(
  pipeline_selected = wrap_legend_label(
                        model_display_label("Pipeline-selected", ps_feats, max_chars = 500),
                        width = 62),
  tnm_baseline      = wrap_legend_label(
                        model_display_label("AJCC T/N baseline", tnm_feats, max_chars = 500),
                        width = 62),
  "Treat All"       = "Treat All",
  "Treat None"      = "Treat None"
)

sub_dca <- dca_df |>
  filter(strategy %in% dca_keys) |>
  mutate(strategy = factor(strategy, levels = dca_keys))

max_nb  <- max(sub_dca$net_benefit[sub_dca$strategy != "Treat None"], na.rm = TRUE)
y_upper <- max(0.05, ceiling(max_nb * 110) / 100)

panel_c <- ggplot(sub_dca,
                  aes(x = threshold, y = net_benefit,
                      color = strategy, linetype = strategy)) +
  geom_line(linewidth = 1.0) +
  scale_color_manual(
    values = dca_colors[dca_keys],
    labels = dca_display_lbls[dca_keys],
    name   = NULL, guide = guide_legend(ncol = 1, byrow = TRUE)
  ) +
  scale_linetype_manual(
    values = dca_linetypes[dca_keys],
    labels = dca_display_lbls[dca_keys],
    name   = NULL, guide = guide_legend(ncol = 1, byrow = TRUE)
  ) +
  coord_cartesian(ylim = c(-0.02, y_upper)) +
  labs(
    tag      = "C",
    title    = "Decision Curve Analysis",
    subtitle = paste0("Common evaluation set: N=", n_common),
    x        = "Threshold Probability",
    y        = "Net Benefit"
  ) +
  theme_bw(base_size = 11) +
  theme(plot.tag        = element_text(face = "bold", size = 14),
        plot.title      = element_text(face = "bold"),
        plot.subtitle   = element_text(size = 8.5, color = "grey40"),
        legend.position = "bottom",
        legend.text     = element_text(size = 8.2, lineheight = 0.95),
        legend.key.size = unit(0.5, "cm"),
        legend.key.width = unit(0.7, "cm"),
        plot.margin     = ggplot2::margin(5.5, 5.5, 5.5, 5.5))

# ============================================================
# Combine panels and save
# ============================================================
panel_list <- Filter(Negate(is.null), list(panel_a, panel_b, panel_c))
n_panels   <- length(panel_list)

if (n_panels == 3) {
  fig3 <- panel_list[[1]] + panel_list[[2]] + panel_list[[3]] + plot_layout(ncol = 3)
  fig_width <- 21
  fig_height <- 8.6
} else if (n_panels == 2) {
  fig3 <- panel_list[[1]] + panel_list[[2]] + plot_layout(ncol = 2)
  fig_width <- 14
  fig_height <- 8.6
  cat("  [warn] Only 2 panels assembled (panel B calibration may have failed).\n")
} else {
  cat("  [warn] Fewer than 2 panels available; skipping combined figure.\n")
  fig3 <- NULL
}

if (!is.null(fig3)) {
  ggsave(file.path(FIGURES_DIR, "fig3_abc_pipeline_selected_evaluation.png"),
         fig3, width = fig_width, height = fig_height, dpi = 150)
  ggsave(file.path(FIGURES_DIR, "fig3_abc_pipeline_selected_evaluation.pdf"),
         fig3, width = fig_width, height = fig_height)
  cat("  Saved fig3_abc_pipeline_selected_evaluation figures.\n")
}

cat("=== 10 complete ===\n\n")
