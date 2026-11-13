# scripts/08_survival_calibration.R
# Calibration tables for predicted 1-, 2-, and 3-year OS.
#   Training cohort:   rms::calibrate() with bootstrap
#   Validation cohort: grouped observed KM vs predicted survival

cat("=== 08_survival_calibration.R ===\n")

source("config/config.R")

for (pkg in c("rms", "survival", "dplyr", "readr")) {
  if (!requireNamespace(pkg, quietly = TRUE))
    stop("Install '", pkg, "' with: install.packages('", pkg, "')")
}
suppressPackageStartupMessages({
  library(rms); library(survival); library(dplyr); library(readr)
})

set.seed(SEED)

df   <- readRDS(file.path(MODELS_DIR, "prognostic_data_clean.rds"))
df_t <- df[df$cohort == "Training", ]
df_v <- df[df$cohort == "Internal validation", ]
cat("  Training N =", nrow(df_t), "| Validation N =", nrow(df_v), "\n")

features <- FEATURES_NOMOGRAM_FIGURE
present  <- intersect(features, names(df_t))

df_t_mod <- df_t |>
  select(all_of(c(present, TIME_VAR, EVENT_VAR))) |>
  na.omit() |>
  mutate(across(where(is.ordered), ~ factor(.x, ordered = FALSE)))

zero_var_cal <- sapply(present, function(v) {
  if (is.factor(df_t_mod[[v]])) length(unique(droplevels(df_t_mod[[v]]))) <= 1
  else isTRUE(var(as.numeric(df_t_mod[[v]]), na.rm = TRUE) == 0)
})
if (any(zero_var_cal)) {
  dropped_cal <- present[zero_var_cal]
  cat("  [note] Removing zero-variance variables for calibration:",
      paste(dropped_cal, collapse = ", "), "\n")
  present   <- present[!zero_var_cal]
  df_t_mod  <- df_t_mod |> select(all_of(c(present, TIME_VAR, EVENT_VAR)))
}

old_contrasts <- getOption("contrasts")
options(contrasts = c("contr.treatment", "contr.treatment"))

dd <- rms::datadist(df_t_mod)
assign("dd", dd, envir = .GlobalEnv)
options(datadist = "dd")

cph_formula <- as.formula(
  paste0("Surv(", TIME_VAR, ", ", EVENT_VAR, ") ~ ",
         paste(present, collapse = " + "))
)

cph_fit <- tryCatch(
  rms::cph(cph_formula, data = df_t_mod, x = TRUE, y = TRUE,
           surv = TRUE, time.inc = 12),
  error = function(e) {
    cat("  [warn] rms::cph failed:", conditionMessage(e), "\n"); NULL
  }
)
if (is.null(cph_fit)) stop("rms::cph model fitting failed.")

# Rebuild without singular variables if any are detected
singular_terms <- which(is.na(coef(cph_fit)))
if (length(singular_terms) > 0) {
  singular_vars <- unique(gsub("=.*$", "", names(singular_terms)))
  cat("  [note] Singular variables detected:", paste(singular_vars, collapse = ", "),
      "\n  Rebuilding calibration model without singular variables.\n")
  present_cal <- setdiff(present, singular_vars)
  if (length(present_cal) == 0) {
    stop("No non-singular predictors available for calibration model.")
  }
  df_t_mod <- df_t |>
    select(all_of(c(present_cal, TIME_VAR, EVENT_VAR))) |>
    na.omit() |>
    mutate(across(where(is.ordered), ~ factor(.x, ordered = FALSE)))
  dd_cal <- rms::datadist(df_t_mod)
  assign("dd", dd_cal, envir = .GlobalEnv)
  options(datadist = "dd")
  cph_formula_cal <- as.formula(
    paste0("Surv(", TIME_VAR, ", ", EVENT_VAR, ") ~ ",
           paste(present_cal, collapse = " + "))
  )
  cph_fit <- tryCatch(
    rms::cph(cph_formula_cal, data = df_t_mod, x = TRUE, y = TRUE,
             surv = TRUE, time.inc = 12),
    error = function(e) { cat("  [warn] Fallback cph failed:", conditionMessage(e), "\n"); NULL }
  )
  if (is.null(cph_fit)) stop("Calibration model fallback failed.")
  cat("  [note] Calibration uses (", length(present_cal), "vars):",
      paste(present_cal, collapse = ", "), "\n")
}

cat("  rms::cph fitted on training (N =", nrow(df_t_mod), ")\n")

# ---- Training calibration via rms::calibrate() at each horizon -----------------
B_boot <- CALIBRATION_BOOTSTRAP_B
cat("  Running training calibration (B =", B_boot, "bootstrap) ...\n")
grouped_rows <- list()
metrics_rows <- list()

for (hi in seq_along(TIME_HORIZONS_MONTHS)) {
  t_star <- TIME_HORIZONS_MONTHS[hi]
  t_lbl  <- TIME_HORIZONS_LABELS[hi]
  cat("   Training calibration:", t_lbl, "...\n")

  cal_obj <- tryCatch(
    rms::calibrate(cph_fit, u = t_star, ctype = "hare", B = B_boot),
    error = function(e) {
      cat("  [warn] rms::calibrate (hare) failed:", conditionMessage(e), "\n")
      tryCatch(
        rms::calibrate(cph_fit, u = t_star, B = B_boot),
        error = function(e2) { cat("  [warn] calibrate fallback failed:", conditionMessage(e2), "\n"); NULL }
      )
    }
  )
  if (is.null(cal_obj)) next

  cal_mat  <- tryCatch(unclass(cal_obj), error = function(e) NULL)
  if (is.null(cal_mat)) next

  avail_cols <- colnames(cal_mat)
  pred_col   <- intersect(c("mean.predicted", "pred"), avail_cols)[1]
  obs_col    <- intersect(c("calibrated", "KM", "obs"), avail_cols)[1]
  if (is.na(pred_col) || is.na(obs_col)) {
    cat("  [warn] Cannot identify calibration columns:", paste(avail_cols, collapse = ", "), "\n")
    next
  }

  pred_s <- as.numeric(cal_mat[, pred_col])
  obs_s  <- as.numeric(cal_mat[, obs_col])
  keep   <- !is.na(pred_s) & !is.na(obs_s)

  grouped_rows[[paste("training", t_lbl)]] <- data.frame(
    cohort = "Training", time_label = t_lbl, time_months = t_star,
    group = seq_len(sum(keep)),
    predicted = pred_s[keep], observed = obs_s[keep],
    stringsAsFactors = FALSE
  )

  if (sum(keep) >= 3) {
    cal_lm <- tryCatch(lm(obs_s[keep] ~ pred_s[keep]), error = function(e) NULL)
    if (!is.null(cal_lm)) {
      metrics_rows[[paste("training", t_lbl)]] <- data.frame(
        cohort = "Training", time_label = t_lbl, time_months = t_star,
        cal_slope = round(coef(cal_lm)[2], 4),
        cal_intercept = round(coef(cal_lm)[1], 4),
        stringsAsFactors = FALSE
      )
    }
  }
}

# ---- Validation calibration: grouped KM vs predicted survival ------------------
cat("  Running validation calibration (grouped KM) ...\n")

df_v_mod <- df_v |>
  select(all_of(c(present, TIME_VAR, EVENT_VAR))) |>
  na.omit()

for (hi in seq_along(TIME_HORIZONS_MONTHS)) {
  t_star <- TIME_HORIZONS_MONTHS[hi]
  t_lbl  <- TIME_HORIZONS_LABELS[hi]
  cat("   Validation calibration:", t_lbl, "...\n")

  pred_surv <- tryCatch({
    sf_obj <- rms::survest(cph_fit, newdata = df_v_mod, times = t_star)
    if (is.list(sf_obj) && "surv" %in% names(sf_obj)) {
      as.numeric(sf_obj$surv)
    } else {
      as.numeric(sf_obj)
    }
  }, error = function(e) {
    cox_obj <- readRDS(file.path(MODELS_DIR, "cox_model_nomogram_figure.rds"))
    tryCatch({
      sf <- survfit(cox_obj, newdata = df_v_mod)
      idx <- findInterval(t_star, sf$time)
      if (idx == 0) rep(1.0, nrow(df_v_mod))
      else as.numeric(sf$surv[idx, ])
    }, error = function(e2) NULL)
  })

  if (is.null(pred_surv)) next

  valid_idx <- !is.na(pred_surv)
  if (sum(valid_idx) < 10) next

  pred_v <- pred_surv[valid_idx]
  tv     <- as.numeric(df_v_mod[[TIME_VAR]])[valid_idx]
  ev     <- as.integer(df_v_mod[[EVENT_VAR]])[valid_idx]

  n_groups <- min(10, floor(sum(valid_idx) / 5))
  if (n_groups < 3) n_groups <- 3
  groups   <- cut(pred_v, breaks = quantile(pred_v, probs = seq(0, 1, 1/n_groups)),
                  include.lowest = TRUE, labels = FALSE)

  km_rows <- lapply(seq_len(n_groups), function(g) {
    idx_g <- groups == g
    if (sum(idx_g) < 3) return(NULL)
    tv_g  <- tv[idx_g]; ev_g <- ev[idx_g]
    km_g  <- tryCatch(survfit(Surv(tv_g, ev_g) ~ 1), error = function(e) NULL)
    if (is.null(km_g)) return(NULL)
    km_idx <- max(which(km_g$time <= t_star), 0)
    obs_s  <- if (km_idx == 0) 1.0 else km_g$surv[km_idx]
    data.frame(group = g, predicted = mean(pred_v[idx_g]), observed = obs_s)
  })
  km_tbl <- bind_rows(Filter(Negate(is.null), km_rows))
  if (nrow(km_tbl) < 3) next

  grouped_rows[[paste("validation", t_lbl)]] <- data.frame(
    cohort = "Internal validation", time_label = t_lbl, time_months = t_star,
    group = km_tbl$group,
    predicted = km_tbl$predicted, observed = km_tbl$observed,
    stringsAsFactors = FALSE
  )

  if (nrow(km_tbl) >= 3) {
    cal_lm2 <- tryCatch(lm(observed ~ predicted, data = km_tbl), error = function(e) NULL)
    if (!is.null(cal_lm2)) {
      metrics_rows[[paste("validation", t_lbl)]] <- data.frame(
        cohort = "Internal validation", time_label = t_lbl, time_months = t_star,
        cal_slope = round(coef(cal_lm2)[2], 4),
        cal_intercept = round(coef(cal_lm2)[1], 4),
        stringsAsFactors = FALSE
      )
    }
  }
}

grouped_all <- bind_rows(grouped_rows)
write_csv(grouped_all, file.path(TABLES_DIR, "calibration_survival_grouped.csv"))

metrics_all <- bind_rows(metrics_rows)
write_csv(metrics_all, file.path(TABLES_DIR, "calibration_survival_metrics.csv"))
cat("  Saved calibration tables.\n")

meth_note <- data.frame(
  component = "Survival calibration",
  method    = "Training: rms::calibrate bootstrap; Validation: grouped KM vs predicted",
  note      = "Validation calibration uses quantile groups on predicted survival probability.",
  stringsAsFactors = FALSE
)
meth_path <- file.path(TABLES_DIR, "methodological_notes_prognostic.csv")
if (file.exists(meth_path)) {
  prev <- read.csv(meth_path, stringsAsFactors = FALSE)
  write.csv(rbind(prev, meth_note), meth_path, row.names = FALSE)
} else {
  write.csv(meth_note, meth_path, row.names = FALSE)
}

cat("=== 08 complete ===\n\n")
