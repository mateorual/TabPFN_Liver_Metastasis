# scripts/08_dca_analysis.R
# Decision curve analysis — manual net-benefit table for all diagnostic models.
#
# Net benefit: NB = TP/N - FP/N * threshold/(1-threshold)
# Treat All:   prevalence - (1-prevalence) * threshold/(1-threshold)
# Treat None:  0

cat("=== 08_dca_analysis.R ===\n")

source("config/config.R")
source("scripts/helpers_diagnostic.R")

for (pkg in c("dplyr", "readr")) {
  if (!requireNamespace(pkg, quietly = TRUE))
    stop("Install '", pkg, "' with: install.packages('", pkg, "')")
}
suppressPackageStartupMessages({ library(dplyr); library(readr) })

set.seed(SEED)

df <- readRDS(file.path(MODELS_DIR, "diagnostic_data_clean.rds"))

load_model <- function(name) {
  p <- file.path(MODELS_DIR, paste0("logistic_model_", name, ".rds"))
  if (!file.exists(p)) return(NULL)
  readRDS(p)
}

ps_feats <- tryCatch(
  readRDS(file.path(MODELS_DIR, "pipeline_selected_features.rds")),
  error = function(e) character(0)
)

models_list <- Filter(Negate(is.null), list(
  paper_text        = load_model("paper_text"),
  nomogram_figure   = load_model("nomogram_figure"),
  pipeline_selected = load_model("pipeline_selected")
))

# AJCC T/N baseline: the comparator fitted and saved by script 04.
tnm_m <- load_model("tnm_baseline")
if (is.null(tnm_m)) stop("logistic_model_tnm_baseline.rds not found. Run script 04 first.")
models_list[["tnm_baseline"]] <- tnm_m
cat("  Models loaded:", paste(names(models_list), collapse = ", "), "\n")

model_labels <- c(
  paper_text        = model_display_label("Paper Text", FEATURES_PAPER_TEXT, max_chars = 120),
  nomogram_figure   = model_display_label("Nomogram Figure", FEATURES_NOMOGRAM_FIGURE, max_chars = 120),
  pipeline_selected = model_display_label("Pipeline-selected",
                        if (length(ps_feats) > 0) ps_feats else character(0), max_chars = 120),
  tnm_baseline      = model_display_label("AJCC T/N baseline", FEATURES_TNM_BASELINE, max_chars = 120),
  "Treat All"       = "Treat All",
  "Treat None"      = "Treat None"
)

all_dca_vars <- unique(c(
  FEATURES_PAPER_TEXT, FEATURES_NOMOGRAM_FIGURE,
  if (length(ps_feats) > 0) ps_feats else character(0),
  FEATURES_TNM_BASELINE, OUTCOME
))
all_dca_vars  <- intersect(all_dca_vars, names(df))
df_common     <- df[complete.cases(df[, all_dca_vars]), ]
n_common      <- nrow(df_common)
y_common      <- df_common[[OUTCOME]]
cat("  Common evaluation set: N =", n_common, "\n")

thresholds <- seq(0.01, 0.99, by = 0.01)

compute_nb <- function(y, probs, thresholds) {
  valid <- !is.na(probs)
  y_v   <- y[valid]
  p_v   <- as.numeric(probs)[valid]
  n_v   <- length(y_v)
  sapply(thresholds, function(t) {
    tp <- sum(p_v >= t & y_v == 1)
    fp <- sum(p_v >= t & y_v == 0)
    tp / n_v - fp / n_v * (t / (1 - t))
  })
}

prevalence <- mean(y_common, na.rm = TRUE)
treat_all_nb <- prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))

dca_rows <- list(
  "Treat All"  = data.frame(threshold = thresholds, net_benefit = treat_all_nb, strategy = "Treat All"),
  "Treat None" = data.frame(threshold = thresholds, net_benefit = 0, strategy = "Treat None")
)

for (nm in names(models_list)) {
  probs <- tryCatch(
    predict(models_list[[nm]], newdata = df_common, type = "response", na.action = na.exclude),
    error = function(e) NULL
  )
  if (is.null(probs)) next
  nb_vals <- compute_nb(y_common, probs, thresholds)
  dca_rows[[nm]] <- data.frame(threshold = thresholds, net_benefit = nb_vals, strategy = nm)
  log_sample_size("08_dca", nm, nrow(df), sum(!is.na(probs)))
}

dca_df <- bind_rows(dca_rows)
dca_df$display_label <- model_labels[dca_df$strategy]
dca_df$display_label[is.na(dca_df$display_label)] <- dca_df$strategy[is.na(dca_df$display_label)]

write_csv(dca_df, file.path(TABLES_DIR, "dca_diagnostic_results.csv"))

write_csv(
  data.frame(figure = "dca_diagnostic_results", eval_set = "common_all_models",
             n = n_common, vars_included = paste(setdiff(all_dca_vars, OUTCOME), collapse = "; "),
             stringsAsFactors = FALSE),
  file.path(TABLES_DIR, "dca_evaluation_sample_sizes.csv")
)

cat("=== 08 complete ===\n\n")
