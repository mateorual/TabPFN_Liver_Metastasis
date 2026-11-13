# scripts/06_prognostic_nomogram.R
# Prognostic nomogram for 1-, 2-, and 3-year OS using rms::cph() -> fig5_prognostic_nomogram.
# Training cohort only. Zero-variance variables (e.g. mets_brain absent from
# training) are automatically excluded with a warning before rms fitting.

cat("=== 06_prognostic_nomogram.R ===\n")

source("config/config.R")

for (pkg in c("rms", "survival", "dplyr", "readr")) {
  if (!requireNamespace(pkg, quietly = TRUE))
    stop("Install '", pkg, "' with: install.packages('", pkg, "')")
}
suppressPackageStartupMessages({
  library(rms); library(survival); library(dplyr); library(readr)
})

df   <- readRDS(file.path(MODELS_DIR, "prognostic_data_clean.rds"))
df_t <- df[df$cohort == "Training", ]
cat("  Training N =", nrow(df_t), "| Events =", sum(df_t[[EVENT_VAR]]), "\n")

present <- intersect(FEATURES_NOMOGRAM_FIGURE, names(df_t))

# Zero-variance predictors in training (e.g. mets_brain) cannot be estimated
# by rms and are excluded, with a method-note audit trail.
zero_var <- sapply(present, function(v) {
  if (is.factor(df_t[[v]])) length(unique(droplevels(df_t[[v]]))) <= 1
  else isTRUE(var(as.numeric(df_t[[v]]), na.rm = TRUE) == 0)
})
dropped <- present[zero_var]

if (length(dropped) > 0) {
  absent_info <- sapply(dropped, function(v) {
    n_full  <- if (is.factor(df[[v]]))   sum(df[[v]]   != levels(df[[v]])[1],   na.rm = TRUE) else 0L
    n_train <- if (is.factor(df_t[[v]])) sum(df_t[[v]] != levels(df_t[[v]])[1], na.rm = TRUE) else 0L
    paste0(v, " (full=", n_full, ", training=", n_train, ")")
  })
  cat("  [note] Unestimable vars in training for nomogram_figure nomogram:",
      paste(absent_info, collapse = "; "), "\n")
  log_method_note(
    "Nomogram (nomogram_figure) — zero-variance exclusion",
    "Variable absent/constant in training cohort",
    paste0("The following Figure 5 variables cannot be estimated from training data ",
           "and were excluded from rms nomogram fitting: ",
           paste(absent_info, collapse = "; "),
           ". Cox model predictions (script 05) use all specified variables via survival::coxph.")
  )
  write_csv(
    data.frame(variable = dropped, reason = "zero-variance in training", stringsAsFactors = FALSE),
    file.path(TABLES_DIR, "dropped_nomogram_vars_nomogram_figure.csv")
  )
  present <- present[!zero_var]
}

df_mod <- df_t |>
  select(all_of(c(present, TIME_VAR, EVENT_VAR))) |>
  na.omit() |>
  mutate(across(where(is.ordered), ~ factor(.x, ordered = FALSE)))

cat("  N =", nrow(df_mod), "| Features:", paste(present, collapse = ", "), "\n")

old_contrasts <- getOption("contrasts")
options(contrasts = c("contr.treatment", "contr.treatment"))
on.exit(options(contrasts = old_contrasts), add = TRUE)

dd <- rms::datadist(df_mod)
assign("dd", dd, envir = .GlobalEnv)
options(datadist = "dd")

cph_formula <- as.formula(
  paste0("Surv(", TIME_VAR, ", ", EVENT_VAR, ") ~ ",
         paste(present, collapse = " + "))
)

warn_log <- character(0)
cph_fit <- tryCatch(
  withCallingHandlers(
    rms::cph(cph_formula, data = df_mod, x = TRUE, y = TRUE,
             surv = TRUE, time.inc = 12),
    warning = function(w) { warn_log <<- c(warn_log, conditionMessage(w));
                             invokeRestart("muffleWarning") }
  ),
  error = function(e) {
    cat("  [warn] rms::cph failed:", conditionMessage(e), "\n")
    NULL
  }
)
if (!is.null(cph_fit) && length(warn_log) > 0) {
  cat("  Warnings:", paste(warn_log, collapse = "; "), "\n")
}
if (is.null(cph_fit)) stop("rms::cph model fitting failed — nomogram skipped.")

saveRDS(cph_fit, file.path(MODELS_DIR, "prognostic_nomogram_cph_model.rds"))
cat("  Saved prognostic_nomogram_cph_model.rds\n")

surv_fn <- rms::Survival(cph_fit)
surv_1yr <- function(x) surv_fn(12, x)
surv_2yr <- function(x) surv_fn(24, x)
surv_3yr <- function(x) surv_fn(36, x)

nom <- rms::nomogram(
  cph_fit,
  fun        = list(surv_1yr, surv_2yr, surv_3yr),
  funlabel   = c("1-year survival probability",
                 "2-year survival probability",
                 "3-year survival probability"),
  fun.at     = list(
    c(.05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95),
    c(.05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95),
    c(.05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95)
  ),
  maxscale   = 100,
  lp         = FALSE
)

png(file.path(FIGURES_DIR, "fig5_prognostic_nomogram.png"),
    width = 2400, height = 1800, res = 150)
plot(nom, xfrac = 0.4, label.every = 1, col.grid = gray(c(0.8, 0.95)),
     main = paste0("Prognostic Nomogram: 1-, 2-, 3-year OS\n(nomogram_figure | ",
                   length(present), " vars)"))
dev.off()

pdf(file.path(FIGURES_DIR, "fig5_prognostic_nomogram.pdf"), width = 16, height = 12)
plot(nom, xfrac = 0.4, label.every = 1, col.grid = gray(c(0.8, 0.95)),
     main = paste0("Prognostic Nomogram: 1-, 2-, 3-year OS\n(nomogram_figure | ",
                   length(present), " vars)"))
dev.off()
cat("  Saved fig5_prognostic_nomogram figures.\n")

var_levels <- do.call(rbind, lapply(present, function(v) {
  lvls <- if (is.factor(df_mod[[v]])) paste(levels(df_mod[[v]]), collapse = " | ")
          else paste0("continuous [",
                      round(min(df_mod[[v]], na.rm = TRUE), 1), " - ",
                      round(max(df_mod[[v]], na.rm = TRUE), 1), "]")
  data.frame(variable = v, levels_or_range = lvls, stringsAsFactors = FALSE)
}))
write_csv(var_levels, file.path(TABLES_DIR, "prognostic_nomogram_variable_levels.csv"))
log_sample_size("06_nomogram", "nomogram_figure", nrow(df), nrow(df_mod))

cat("=== 06 complete ===\n\n")
