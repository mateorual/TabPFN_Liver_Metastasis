# src/external_validation_helpers.R
# Shared functions for scripts/02_run_external_validation.R (frozen-model external
# validation of the reference-paper reproduction models on the UK-Erlangen cohort).
#
# Rules enforced here:
#   * no model is (re)fitted; every model, including the diagnostic AJCC T/N comparator,
#     is the object saved by the reference training code (01_RefPaper_*);
#   * factor levels/reference levels come from each model's stored xlevels;
#   * predictions are never altered by external calibration metrics;
#   * the bootstrap schedule is generated once, shared by all models/metrics,
#     and failed draws are counted, never redrawn.

EV_REQUIRED_PKGS <- c("survival", "rms", "timeROC", "pROC", "riskRegression", "prodlim", "ggplot2",
                      "patchwork", "jsonlite", "digest", "readr", "dplyr", "scales", "ragg")

ev_require <- function(pkgs = EV_REQUIRED_PKGS) {
  miss <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (length(miss)) stop("Missing R packages (not installing automatically): ", paste(miss, collapse = ", "))
  # Attached because riskRegression::Score resolves Hist()/Surv() in its formula by name.
  suppressPackageStartupMessages({
    library(survival); library(prodlim); library(riskRegression); library(rms)
  })
  invisible(TRUE)
}

# ==================================================================================
# 1. Infrastructure: paths, hashing, atomic writes
# ==================================================================================

ev_paths <- function(config_path) {
  cfg_file <- normalizePath(config_path, winslash = "/", mustWork = TRUE)
  module <- dirname(dirname(cfg_file))
  repo <- dirname(dirname(module))
  list(config = cfg_file, module = module, repo = repo,
       out = file.path(module, "outputs"), local = file.path(module, "outputs", "local_only"),
       script = file.path(module, "scripts", "02_run_external_validation.R"),
       helpers = file.path(module, "src", "external_validation_helpers.R"))
}

ev_repo <- function(paths, rel) file.path(paths$repo, rel)
ev_strip_prefix <- function(prefix, x) {
  x <- normalizePath(x, winslash = "/", mustWork = FALSE); p <- paste0(prefix, "/")
  ifelse(startsWith(x, p), substring(x, nchar(p) + 1), x)
}
ev_rel <- function(paths, abs) ev_strip_prefix(paths$repo, abs)

ev_sha256_file <- function(path) digest::digest(file = path, algo = "sha256")
ev_sha256_text <- function(x) digest::digest(paste(x, collapse = "\n"), algo = "sha256", serialize = FALSE)

ev_replace <- function(tmp, path) {
  if (file.exists(path)) file.remove(path)
  if (!file.rename(tmp, path)) stop("atomic rename failed for ", path)
  invisible(path)
}

ev_write_csv <- function(df, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  tmp <- paste0(path, ".tmp-", Sys.getpid())
  readr::write_csv(as.data.frame(df), tmp, na = "")
  ev_replace(tmp, path)
}

ev_json_text <- function(x) jsonlite::toJSON(x, auto_unbox = TRUE, pretty = TRUE, digits = NA, null = "null", na = "null")

ev_write_json <- function(x, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  tmp <- paste0(path, ".tmp-", Sys.getpid())
  writeLines(ev_json_text(x), tmp, useBytes = TRUE)
  ev_replace(tmp, path)
}

ev_write_text <- function(lines, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  tmp <- paste0(path, ".tmp-", Sys.getpid())
  con <- file(tmp, open = "wb"); writeLines(enc2utf8(lines), con, useBytes = TRUE); close(con)
  ev_replace(tmp, path)
}

ev_save_rds <- function(obj, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  tmp <- paste0(path, ".tmp-", Sys.getpid())
  saveRDS(obj, tmp)
  ev_replace(tmp, path)
}

ev_read_csv <- function(path) as.data.frame(readr::read_csv(path, show_col_types = FALSE, guess_max = 1e6,
                                                            progress = FALSE))

ev_now <- function() format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z")

ev_load_cfg <- function(paths) jsonlite::fromJSON(paths$config, simplifyVector = FALSE)

ev_chr <- function(x) unlist(x, use.names = FALSE)

ev_pkg_versions <- function(pkgs = EV_REQUIRED_PKGS) {
  stats::setNames(lapply(pkgs, function(p) as.character(utils::packageVersion(p))), pkgs)
}

# ==================================================================================
# 2. Status vocabulary and interval support
# ==================================================================================

EV_STATUS <- c("supported", "fragile_interval", "exploratory_small_sample", "unsupported_small_sample",
               "not_estimable_no_class_variation", "not_estimable_no_event_structure",
               "not_estimable_censoring_positivity", "not_estimable_model_incompatibility",
               "not_evaluable_missing_predictor")

ev_interval_support <- function(n_valid, B, rules) {
  if (n_valid >= rules$supported_min_valid_draws && n_valid / B >= rules$supported_min_valid_fraction) {
    "supported"
  } else if (n_valid >= rules$fragile_min_valid_draws) {
    "fragile"
  } else {
    "not_estimable"
  }
}

ev_percentile_ci <- function(x, level) {
  a <- (1 - level) / 2
  as.numeric(stats::quantile(x, c(a, 1 - a), type = 7, names = FALSE))
}

# Point-failure status by metric family.
ev_failure_status <- function(metric) {
  if (metric %in% c("auc", "calibration_in_the_large", "calibration_slope", "calibration_intercept_joint")) {
    "not_estimable_no_class_variation"
  } else if (metric %in% c("brier_ipcw")) {
    "not_estimable_censoring_positivity"
  } else {
    "not_estimable_no_event_structure"
  }
}

# One machine-readable metric row with the pre-specified support classification.
ev_metric_row <- function(cfg, task, model, evaluation_set, n_total, n_evaluable, n_pos, n_neg, metric,
                          estimate, draws, ci_method, horizon = NA_real_, extra_limits = character(0),
                          status_override = NULL, point_fail_status = NULL, analytic_ci = NULL,
                          rows_hash = NA_character_, extra = list()) {
  B <- cfg$bootstrap$draws
  rules <- cfg$bootstrap$support_rules
  lim <- character(0)
  if (n_evaluable < n_total) lim <- c(lim, sprintf("complete_case_excluded_%d", n_total - n_evaluable))
  if (n_evaluable < 20) lim <- c(lim, "n_evaluable_lt_20")
  ci <- c(NA_real_, NA_real_); n_valid <- NA_integer_; support <- NA_character_
  if (!is.null(draws)) {
    valid <- draws[is.finite(draws)]
    n_valid <- length(valid)
    support <- ev_interval_support(n_valid, B, rules)
    if (n_valid < rules$supported_min_valid_draws) lim <- c(lim, "bootstrap_valid_draws_lt_1000")
    if (n_valid / B < rules$supported_min_valid_fraction) lim <- c(lim, "bootstrap_valid_fraction_lt_0.5")
    if (support != "not_estimable") ci <- ev_percentile_ci(valid, cfg$bootstrap$confidence_level)
    if (support == "not_estimable") lim <- c(lim, "bootstrap_valid_draws_lt_200_interval_not_reported")
  } else if (!is.null(analytic_ci)) {
    ci <- analytic_ci
    support <- if (all(is.finite(ci))) "analytic" else "not_estimable"
  }
  fail <- if (is.null(point_fail_status)) ev_failure_status(metric) else point_fail_status
  if (n_evaluable == 0) {
    status <- "not_evaluable_missing_predictor"; estimate <- NA_real_; ci <- c(NA_real_, NA_real_)
    lim <- c(lim, "no_evaluable_patients")
  } else if (!is.finite(estimate)) {
    status <- fail; estimate <- NA_real_; ci <- c(NA_real_, NA_real_)
    lim <- c(lim, "point_estimate_not_defined")
  } else if (identical(support, "supported") || identical(support, "analytic")) {
    status <- "supported"
  } else if (identical(support, "fragile")) {
    status <- "fragile_interval"
  } else if (identical(support, "not_estimable")) {
    status <- fail; lim <- c(lim, "point_estimate_retained")
  } else {
    status <- "supported"
  }
  if (!is.null(status_override) && is.finite(estimate)) status <- status_override
  lim <- c(lim, extra_limits)
  row <- data.frame(
    task = task, cohort = cfg$figures$cohort_display_name, model = model, evaluation_set = evaluation_set,
    horizon_months = horizon, n_total = n_total, n_evaluable = n_evaluable,
    n_events_or_positive = n_pos, n_nonevents_or_censored = n_neg, metric = metric,
    estimate = estimate, ci_lower_95 = ci[1], ci_upper_95 = ci[2], ci_method = ci_method,
    bootstrap_draws_planned = if (is.null(draws)) NA_integer_ else B,
    bootstrap_draws_valid = n_valid, interval_support = support,
    ci_displayable = identical(support, "supported") || identical(support, "analytic"),
    support_status = status,
    limitation_code = if (length(lim)) paste(unique(lim), collapse = ";") else "none",
    evaluation_rows_sha256 = rows_hash, stringsAsFactors = FALSE)
  for (k in names(extra)) row[[k]] <- extra[[k]]
  row$metric_key <- paste(task, model, evaluation_set, metric, ifelse(is.na(horizon), "na", horizon), sep = "|")
  row
}

# ==================================================================================
# 3. Frozen models: registry, newdata, predictions
# ==================================================================================

ev_model_specs <- function(cfg, paths) {
  out <- list()
  for (task in c("diagnostic", "prognostic")) {
    for (m in cfg$models[[task]]) {
      m$task <- task
      m$abs_path <- ev_repo(paths, m$path)
      if (!is.null(m$absolute_risk_object)) m$abs_risk_path <- ev_repo(paths, m$absolute_risk_object)
      out[[paste(task, m$key, sep = ":")]] <- m
    }
  }
  out
}

ev_model_predictors <- function(m) all.vars(stats::delete.response(stats::terms(m)))

ev_model_levels <- function(m, v) {
  if (inherits(m, "cph")) return(m$Design$parms[[v]])
  m$xlevels[[v]]
}

ev_coef_hash <- function(m) {
  b <- stats::coef(m)
  ev_sha256_text(sprintf("%s=%.17g", names(b), b))
}

ev_training_n <- function(m) {
  if (inherits(m, "glm")) return(c(n = length(m$y), events = sum(m$y == 1)))
  y <- m$y
  c(n = nrow(y), events = sum(y[, ncol(y)] == 1))
}

ev_model_contrasts <- function(m) {
  if (!is.null(m$contrasts)) return(lapply(m$contrasts, function(x) if (is.character(x)) x else "matrix"))
  if (inherits(m, "cph")) return(list(note = "rms::cph Design (treatment contrasts)"))
  list()
}

ev_registry_row <- function(spec, m, path, object_role) {
  tn <- ev_training_n(m)
  preds <- ev_model_predictors(m)
  xl <- stats::setNames(lapply(preds, function(v) ev_model_levels(m, v)), preds)
  xl <- xl[!vapply(xl, is.null, logical(1))]
  sha <- ev_sha256_file(path)
  exp_sha <- if (object_role == "absolute_risk_object") spec$absolute_risk_expected_sha256 else spec$expected_sha256
  data.frame(
    task = spec$task, model = spec$key, object_role = object_role, role = spec$role, origin = spec$origin,
    label = spec$label, path = sub("^.*?(analysis/)", "\\1", normalizePath(path, winslash = "/"), perl = TRUE),
    sha256 = sha,
    expected_sha256 = if (is.null(exp_sha)) NA_character_ else exp_sha,
    sha256_matches_expected = if (is.null(exp_sha)) NA else identical(sha, exp_sha),
    class = paste(class(m), collapse = ","),
    formula = paste(deparse(stats::formula(m), width.cutoff = 500L), collapse = " "),
    coefficient_sha256 = ev_coef_hash(m), n_coefficients = length(stats::coef(m)),
    training_n = unname(tn["n"]), training_events = unname(tn["events"]),
    required_predictors = paste(preds, collapse = ";"),
    declared_predictors_match = setequal(preds, ev_chr(spec$declared_predictors)),
    xlevels_json = as.character(jsonlite::toJSON(xl, auto_unbox = FALSE)),
    contrasts_json = as.character(jsonlite::toJSON(ev_model_contrasts(m), auto_unbox = TRUE)),
    stringsAsFactors = FALSE)
}

# Translate prepared 0/1/integer codes to the label vocabulary of the original setup scripts.
ev_encode_predictor <- function(v, x, task, cfg) {
  enc <- cfg$predictor_encoding
  if (v %in% ev_chr(enc$numeric)) return(as.numeric(x))
  if (!is.null(enc$binary_labels[[v]])) {
    if (!all(is.na(x) | x %in% c(0, 1))) stop("binary predictor ", v, " has values outside {0,1,NA}")
    lab <- ev_chr(enc$binary_labels[[v]])
    return(ifelse(is.na(x), NA_character_, lab[as.integer(x) + 1L]))
  }
  if (v == "grade") {
    mp <- ev_chr(enc$grade_labels[[task]]); names(mp) <- names(enc$grade_labels[[task]])
    out <- unname(mp[as.character(as.integer(x))])
    out[is.na(x)] <- NA_character_
    return(out)
  }
  if (v %in% ev_chr(enc$passthrough_factors)) return(as.character(x))
  stop("no encoding rule for predictor ", v)
}

# Build newdata with the MODEL's levels. Unknown levels are incompatibilities, never remapped.
ev_build_newdata <- function(m, df, task, cfg) {
  preds <- ev_model_predictors(m)
  n <- nrow(df)
  nd <- data.frame(.row = seq_len(n))
  miss <- rep(FALSE, n); incompat <- rep(FALSE, n); detail <- list()
  for (v in preds) {
    if (!v %in% names(df)) stop("prepared data lacks predictor ", v)
    val <- ev_encode_predictor(v, df[[v]], task, cfg)
    lv <- ev_model_levels(m, v)
    if (is.null(lv)) {
      nd[[v]] <- as.numeric(val); miss <- miss | is.na(val); next
    }
    bad <- !is.na(val) & !(val %in% lv)
    if (any(bad)) detail[[v]] <- list(n_patients = sum(bad), levels_not_in_model = sort(unique(val[bad])))
    incompat <- incompat | bad
    miss <- miss | is.na(val)
    val[bad] <- NA_character_
    nd[[v]] <- factor(val, levels = lv)
  }
  nd$.row <- NULL
  status <- ifelse(incompat, "incompatible_level", ifelse(miss, "missing_predictor", "ok"))
  list(newdata = nd, status = status, incompatibilities = detail, predictors = preds)
}

ev_predict_glm <- function(m, nb) {
  ok <- nb$status == "ok"
  p <- rep(NA_real_, length(ok)); lp <- p
  if (any(ok)) {
    nd <- nb$newdata[ok, , drop = FALSE]
    p[ok] <- as.numeric(stats::predict(m, newdata = nd, type = "response"))
    lp[ok] <- as.numeric(stats::predict(m, newdata = nd, type = "link"))
  }
  list(prob = p, lp = lp)
}

ev_predict_lp <- function(m, nb) {
  ok <- nb$status == "ok"
  lp <- rep(NA_real_, length(ok))
  if (any(ok)) lp[ok] <- as.numeric(stats::predict(m, newdata = nb$newdata[ok, , drop = FALSE], type = "lp"))
  lp
}

# Frozen cph: stored baseline survival (surv = TRUE), rms::survest per horizon.
ev_surv_cph <- function(cph, nb, times) {
  ok <- nb$status == "ok"
  S <- matrix(NA_real_, length(ok), length(times))
  if (any(ok)) {
    nd <- nb$newdata[ok, , drop = FALSE]
    for (k in seq_along(times)) {
      S[ok, k] <- as.numeric(rms::survest(cph, newdata = nd, times = times[k], se.fit = FALSE)$surv)
    }
  }
  S
}

# Frozen coxph: survfit uses the stored SEER training x/y (Breslow); step value at t as in 09_survival_dca.R.
ev_surv_coxph <- function(m, nb, times) {
  ok <- nb$status == "ok"
  S <- matrix(NA_real_, length(ok), length(times))
  if (any(ok)) {
    nd <- nb$newdata[ok, , drop = FALSE]
    sf <- survival::survfit(m, newdata = nd)
    if (sf$n != nrow(m$y)) stop("survfit baseline is not the stored training cohort")
    sm <- matrix(sf$surv, ncol = nrow(nd))
    for (k in seq_along(times)) {
      idx <- findInterval(times[k], sf$time)
      S[ok, k] <- if (idx == 0) 1 else as.numeric(sm[idx, ])
    }
  }
  S
}

# All patient-level predictions from the frozen objects (no fitting).
ev_make_predictions <- function(cfg, paths, diag, prog) {
  specs <- ev_model_specs(cfg, paths)
  hz <- as.numeric(ev_chr(cfg$evaluation$time_horizons_months))
  dpred <- list(); ppred <- list(); compat <- list()
  for (s in specs) {
    m <- readRDS(s$abs_path)
    df <- if (s$task == "diagnostic") diag else prog
    nb <- ev_build_newdata(m, df, s$task, cfg)
    compat[[paste(s$task, s$key)]] <- data.frame(
      task = s$task, model = s$key, n_cohort = nrow(df), n_ok = sum(nb$status == "ok"),
      n_missing_predictor = sum(nb$status == "missing_predictor"),
      n_incompatible_level = sum(nb$status == "incompatible_level"),
      incompatibilities_json = as.character(jsonlite::toJSON(nb$incompatibilities, auto_unbox = TRUE)),
      stringsAsFactors = FALSE)
    if (s$task == "diagnostic") {
      pr <- ev_predict_glm(m, nb)
      dpred[[s$key]] <- data.frame(patient_id = df$patient_id, row_index = seq_len(nrow(df)), model = s$key,
                                   status = nb$status, predicted_probability = pr$prob,
                                   linear_predictor = pr$lp, mets_liver = df$mets_liver, stringsAsFactors = FALSE)
    } else {
      lp <- ev_predict_lp(m, nb)
      S <- matrix(NA_real_, nrow(df), length(hz))
      if (!is.null(s$absolute_risk_object)) {
        if (s$key == "nomogram_figure") {
          cph <- readRDS(s$abs_risk_path)
          nb2 <- ev_build_newdata(cph, df, "prognostic", cfg)
          if (!identical(nb2$status, nb$status)) stop("cph and coxph eligibility differ")
          S <- ev_surv_cph(cph, nb2, hz)
        } else {
          S <- ev_surv_coxph(m, nb, hz)
        }
      }
      d <- data.frame(patient_id = df$patient_id, row_index = seq_len(nrow(df)), model = s$key,
                      status = nb$status, linear_predictor = lp, stringsAsFactors = FALSE)
      for (k in seq_along(hz)) d[[sprintf("surv_%d", hz[k])]] <- S[, k]
      d$os_months <- df$os_months; d$os_event <- df$os_event
      ppred[[s$key]] <- d
    }
  }
  list(diagnostic = do.call(rbind, dpred), prognostic = do.call(rbind, ppred),
       compatibility = do.call(rbind, compat))
}

# ==================================================================================
# 5. Bootstrap schedules (generated once, before any metric)
# ==================================================================================

ev_set_rng <- function(cfg) {
  k <- ev_chr(cfg$bootstrap$rng_kind)
  RNGkind(kind = k[1], normal.kind = k[2], sample.kind = k[3])
  set.seed(cfg$bootstrap$seed)
}

ev_schedule_diagnostic <- function(y, cfg) {
  ev_set_rng(cfg)
  pos <- which(y == 1); neg <- which(y == 0)
  B <- cfg$bootstrap$draws
  t(vapply(seq_len(B), function(b) {
    c(pos[sample.int(length(pos), length(pos), replace = TRUE)],
      neg[sample.int(length(neg), length(neg), replace = TRUE)])
  }, integer(length(y))))
}

ev_schedule_prognostic <- function(n, cfg) {
  ev_set_rng(cfg)
  B <- cfg$bootstrap$draws
  t(vapply(seq_len(B), function(b) sample.int(n, n, replace = TRUE), integer(n)))
}

ev_schedule_hash <- function(S) ev_sha256_text(apply(S, 1, paste, collapse = ","))

# ==================================================================================
# 6. Estimators
# ==================================================================================

# --- diagnostic -------------------------------------------------------------------
ev_auc_fast <- function(y, p) {
  n1 <- sum(y == 1); n0 <- sum(y == 0)
  if (n1 == 0 || n0 == 0) return(NA_real_)
  r <- rank(p)
  (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

ev_auc_proc <- function(y, p) {
  if (length(unique(y)) < 2) return(NA_real_)
  as.numeric(pROC::auc(pROC::roc(y, p, direction = "<", quiet = TRUE, levels = c(0, 1))))
}

ev_brier <- function(y, p) if (length(y)) mean((y - p)^2) else NA_real_

ev_logit_clip <- function(p, clip) stats::qlogis(pmin(pmax(p, clip), 1 - clip))

# Fast calibration (glm.fit); equality with glm() is checked on the point estimates.
ev_calib_fast <- function(y, p, clip) {
  out <- c(calibration_in_the_large = NA_real_, calibration_slope = NA_real_, calibration_intercept_joint = NA_real_)
  if (length(unique(y)) < 2) return(out)
  lp <- ev_logit_clip(p, clip)
  f1 <- tryCatch(suppressWarnings(stats::glm.fit(x = matrix(1, length(y), 1), y = y, offset = lp,
                                                 family = stats::binomial())), error = function(e) NULL)
  if (!is.null(f1) && isTRUE(f1$converged) && is.finite(f1$coefficients[1])) out[1] <- f1$coefficients[1]
  if (length(unique(lp)) > 1) {
    f2 <- tryCatch(suppressWarnings(stats::glm.fit(x = cbind(1, lp), y = y, family = stats::binomial())),
                   error = function(e) NULL)
    if (!is.null(f2) && isTRUE(f2$converged) && all(is.finite(f2$coefficients))) {
      out[2] <- f2$coefficients[2]; out[3] <- f2$coefficients[1]
    }
  }
  out
}

ev_calib_glm <- function(y, p, clip) {
  out <- c(calibration_in_the_large = NA_real_, calibration_slope = NA_real_, calibration_intercept_joint = NA_real_)
  if (length(unique(y)) < 2) return(out)
  lp <- ev_logit_clip(p, clip)
  f1 <- suppressWarnings(stats::glm(y ~ 1 + offset(lp), family = stats::binomial()))
  if (isTRUE(f1$converged)) out[1] <- stats::coef(f1)[1]
  if (length(unique(lp)) > 1) {
    f2 <- suppressWarnings(stats::glm(y ~ lp, family = stats::binomial()))
    if (isTRUE(f2$converged) && all(is.finite(stats::coef(f2)))) {
      out[2] <- stats::coef(f2)[2]; out[3] <- stats::coef(f2)[1]
    }
  }
  out
}

# Net benefit exactly as 08_dca_analysis.R: NB = TP/N - FP/N * t/(1-t).
ev_net_benefit <- function(y, p, thr) {
  n <- length(y)
  if (n == 0) return(rep(NA_real_, length(thr)))
  vapply(thr, function(t) {
    tp <- sum(p >= t & y == 1); fp <- sum(p >= t & y == 0)
    tp / n - fp / n * (t / (1 - t))
  }, numeric(1))
}

ev_treat_all <- function(y, thr) {
  if (!length(y)) return(rep(NA_real_, length(thr)))
  prev <- mean(y)
  prev - (1 - prev) * (thr / (1 - thr))
}

ev_wilson <- function(k, n, level = 0.95) {
  z <- stats::qnorm(1 - (1 - level) / 2); p <- k / n
  c <- p + z^2 / (2 * n); h <- z * sqrt(p * (1 - p) / n + z^2 / (4 * n^2)); d <- 1 + z^2 / n
  c((c - h) / d, (c + h) / d)
}

# Quantile grouping (pre-specified): try 3 groups, then 2; each group >= min_n.
ev_quantile_groups <- function(p, n_groups, min_n) {
  br <- stats::quantile(p, probs = seq(0, 1, length.out = n_groups + 1), type = 7, names = FALSE)
  if (any(duplicated(br))) return(NULL)
  g <- cut(p, breaks = br, include.lowest = TRUE, labels = FALSE)
  if (any(tabulate(g, n_groups) < min_n)) return(NULL)
  g
}

# --- prognostic -------------------------------------------------------------------
ev_cindex <- function(time, event, lp) {
  if (length(time) < 2 || all(event == 0)) return(c(c_index = NA_real_, comparable_pairs = 0))
  cf <- tryCatch(survival::concordance(survival::Surv(time, event) ~ lp, reverse = TRUE),
                 error = function(e) NULL)
  if (is.null(cf)) return(c(c_index = NA_real_, comparable_pairs = 0))
  pairs <- sum(cf$count[c("concordant", "discordant", "tied.x")])
  c(c_index = if (pairs > 0) unname(cf$concordance) else NA_real_, comparable_pairs = pairs)
}

ev_horizon_counts <- function(time, event, t) {
  c(n_cases = sum(time <= t & event == 1), n_controls = sum(time > t),
    n_censored_by_t = sum(time <= t & event == 0))
}

# timeROC cumulative/dynamic AUC with marginal (KM) IPCW, as 07_time_dependent_roc.R.
ev_time_auc <- function(time, event, marker, t, iid = FALSE) {
  hc <- ev_horizon_counts(time, event, t)
  res <- list(auc = NA_real_, se = NA_real_, TP = NULL, FP = NULL, counts = hc, error = NA_character_)
  if (hc["n_cases"] == 0 || hc["n_controls"] == 0) {
    res$error <- if (hc["n_cases"] == 0) "no_cases_by_horizon" else "no_controls_beyond_horizon"
    return(res)
  }
  out <- tryCatch(suppressWarnings(timeROC::timeROC(T = time, delta = event, marker = marker, cause = 1,
                                                    weighting = "marginal", times = t, iid = iid)),
                  error = function(e) e)
  if (inherits(out, "error")) { res$error <- paste0("timeROC_error: ", conditionMessage(out)); return(res) }
  k <- which(out$times == t)
  if (!length(k)) { res$error <- "timeROC_horizon_not_returned"; return(res) }
  res$auc <- unname(out$AUC[k])
  res$TP <- out$TP[, k]; res$FP <- out$FP[, k]
  if (iid) res$se <- tryCatch(unname(out$inference$vect_sd_1[k]), error = function(e) NA_real_)
  if (!is.finite(res$auc)) res$error <- "timeROC_non_finite_auc"
  res
}

# Reverse Kaplan-Meier of the censoring distribution with the prodlim tie convention
# (deaths leave the risk set before censorings at tied times), i.e. the censoring model
# used by riskRegression::Score(cens.model = "km").
ev_cens_km <- function(time, event) {
  u <- sort(unique(time))
  nr <- vapply(u, function(s) sum(time >= s) - sum(time == s & event == 1), numeric(1))
  ce <- vapply(u, function(s) sum(time == s & event == 0), numeric(1))
  list(u = u, G = cumprod(ifelse(nr > 0, 1 - ce / nr, 1)))
}

# IPCW Brier score at t (Graf et al.): cases weighted 1/G(T_i-), controls 1/G(t),
# censored-before-t weight 0, divided by n. Validated against riskRegression::Score.
ev_brier_ipcw <- function(time, event, S_t, t) {
  if (!length(time) || t > max(time)) return(NA_real_)
  g <- ev_cens_km(time, event)
  G_at <- function(x) { k <- findInterval(x, g$u); ifelse(k == 0, 1, g$G[pmax(k, 1)]) }
  G_prev <- function(x) { k <- match(x, g$u) - 1L; ifelse(k == 0, 1, g$G[pmax(k, 1)]) }
  case <- time <= t & event == 1; ctrl <- time > t
  w <- numeric(length(time))
  w[case] <- 1 / G_prev(time[case]); w[ctrl] <- 1 / G_at(t)
  if (any(!is.finite(w))) return(NA_real_)
  sum(w * (as.numeric(case) - (1 - S_t))^2) / length(time)
}

ev_brier_score_pkg <- function(time, event, S_t, t) {
  if (!length(time) || t > max(time)) return(NA_real_)
  sc <- tryCatch(suppressMessages(riskRegression::Score(
    list(model = matrix(1 - S_t, ncol = 1)), formula = Hist(time, status) ~ 1,
    data = data.frame(time = time, status = event), times = t, metrics = "brier", cens.model = "km",
    null.model = FALSE, conf.int = FALSE, summary = NULL)), error = function(e) NULL)
  if (is.null(sc) || !nrow(sc$Brier$score)) return(NA_real_)
  sc$Brier$score$Brier[1]
}

# Kaplan-Meier survival at t with Greenwood (log) CI; estimable only if follow-up reaches t.
ev_km_at <- function(time, event, t, conf_type = "log") {
  if (!length(time)) return(list(estimable = FALSE, est = NA_real_, lower = NA_real_, upper = NA_real_, n_risk = 0))
  km <- survival::survfit(survival::Surv(time, event) ~ 1, conf.type = conf_type)
  zero_before <- any(km$surv == 0 & km$time <= t)
  if (!(any(time >= t) || zero_before)) return(list(estimable = FALSE, est = NA_real_, lower = NA_real_,
                                                     upper = NA_real_, n_risk = 0))
  s <- summary(km, times = t, extend = TRUE)
  list(estimable = TRUE, est = s$surv, lower = s$lower, upper = s$upper, n_risk = s$n.risk)
}

# Survival-DCA eligibility exactly as 09_survival_dca.R.
ev_dca_eligibility <- function(time, event, t) {
  eligible <- (time >= t) | (event == 1 & time < t)
  y <- as.integer(event == 1 & time <= t)
  list(eligible = eligible, y = y)
}

# ==================================================================================
# 7. Figures (all plots are drawn ONLY from their figure-data frame)
# ==================================================================================

# Theme of the reference implementations: 10_combined_diagnostic_figures.R (diagnostic,
# base 11) and 07/09_*.R (prognostic panels, base 10).
ev_theme <- function(cfg, task = "diagnostic") {
  st <- cfg$figures$style[[task]]
  ggplot2::theme_bw(base_size = st$base_size) +
    ggplot2::theme(plot.tag = ggplot2::element_text(face = "bold", size = st$tag_size),
                   plot.title = ggplot2::element_text(face = "bold", size = st$title_size),
                   plot.subtitle = ggplot2::element_text(size = st$subtitle_size, color = "grey40"),
                   legend.position = "bottom",
                   legend.text = ggplot2::element_text(size = st$legend_text_size, lineheight = 0.95),
                   legend.key.size = ggplot2::unit(st$legend_key_cm, "cm"),
                   legend.key.width = ggplot2::unit(st$legend_key_width_cm, "cm"),
                   plot.margin = ggplot2::margin(5.5, 5.5, 5.5, 5.5))
}

ev_fig_data <- function(panel, layer, series, x, y, ymin = NA_real_, ymax = NA_real_, n = NA_real_,
                        label = NA_character_, status = NA_character_) {
  data.frame(panel = panel, layer = layer, series = series, x = x, y = y, ymin = ymin, ymax = ymax,
             n = n, label = label, status = status, stringsAsFactors = FALSE)
}

ev_not_estimable_panel <- function(title, reason, cfg, task = "diagnostic") {
  ggplot2::ggplot() +
    ggplot2::annotate("text", x = 0.5, y = 0.5, label = paste0("Not estimable:\n", reason), size = 3.6,
                      colour = "grey25") +
    ggplot2::scale_x_continuous(limits = c(0, 1)) + ggplot2::scale_y_continuous(limits = c(0, 1)) +
    ggplot2::labs(title = title, x = NULL, y = NULL) + ev_theme(cfg, task) +
    ggplot2::theme(axis.text = ggplot2::element_blank(), axis.ticks = ggplot2::element_blank(),
                   panel.grid = ggplot2::element_blank())
}

ev_scale_vals <- function(cfg, what, keys) {
  v <- ev_chr(cfg$figures[[what]][keys]); names(v) <- keys; v
}

ev_save_plot <- function(p, family, cfg, paths) {
  wh <- as.numeric(ev_chr(cfg$figures$sizes_in[[family]]))
  dir.create(file.path(paths$out, "figures"), showWarnings = FALSE, recursive = TRUE)
  pdf_path <- file.path(paths$out, "figures", paste0(family, ".pdf"))
  png_path <- file.path(paths$out, "figures", paste0(family, ".png"))
  dev <- if (isTRUE(capabilities("cairo"))) grDevices::cairo_pdf else grDevices::pdf
  tmp_pdf <- paste0(tools::file_path_sans_ext(pdf_path), ".tmp-", Sys.getpid(), ".pdf")
  tmp_png <- paste0(tools::file_path_sans_ext(png_path), ".tmp-", Sys.getpid(), ".png")
  ggplot2::ggsave(tmp_pdf, p, width = wh[1], height = wh[2], device = dev)
  ggplot2::ggsave(tmp_png, p, width = wh[1], height = wh[2], dpi = cfg$figures$dpi, device = ragg::agg_png)
  ev_replace(tmp_pdf, pdf_path); ev_replace(tmp_png, png_path)
  c(pdf = pdf_path, png = png_path)
}

# ==================================================================================
# 8. Diagnostic evaluation
# ==================================================================================

ev_diag_keys <- function(cfg) vapply(cfg$models$diagnostic, function(m) m$key, "")
ev_prog_keys <- function(cfg) vapply(cfg$models$prognostic, function(m) m$key, "")
ev_prog_figure_keys <- function(cfg) vapply(Filter(function(m) isTRUE(m$figure_level), cfg$models$prognostic),
                                            function(m) m$key, "")

ev_wide <- function(pred, keys, col) {
  sapply(keys, function(k) { d <- pred[pred$model == k, ]; d[[col]][order(d$row_index)] })
}

# Evaluation units: (model, evaluation_set, rows). Rows index the task cohort.
ev_units <- function(ok, keys, common_sets) {
  units <- list()
  for (k in keys) units[[length(units) + 1]] <- list(model = k, set = "model_specific", rows = which(ok[, k]))
  for (cs in names(common_sets)) {
    rows <- which(apply(ok[, common_sets[[cs]], drop = FALSE], 1, all))
    for (k in common_sets[[cs]]) units[[length(units) + 1]] <- list(model = k, set = cs, rows = rows)
  }
  units
}

ev_rows_hash <- function(ids, rows) ev_sha256_text(sort(ids[rows]))

ev_evaluate_diagnostic <- function(cfg, dp, sched) {
  keys <- ev_diag_keys(cfg)
  base <- dp[dp$model == keys[1], ]; base <- base[order(base$row_index), ]
  ids <- base$patient_id; y <- base$mets_liver
  P <- ev_wide(dp, keys, "predicted_probability")
  ok <- !is.na(P)
  units <- ev_units(ok, keys, list(common_all_diagnostic_models = keys))
  thr <- seq(cfg$evaluation$dca$thresholds$from, cfg$evaluation$dca$thresholds$to, by = cfg$evaluation$dca$thresholds$by)
  clip <- cfg$evaluation$probability_clip_for_logit
  B <- nrow(sched); n <- length(y)
  nm <- c("auc", "brier", "calibration_in_the_large", "calibration_slope", "calibration_intercept_joint")
  D <- array(NA_real_, c(B, length(units), length(nm)))
  NB <- array(NA_real_, c(B, length(units), length(thr))); TA <- NB
  in_set <- lapply(units, function(u) { v <- rep(FALSE, n); v[u$rows] <- TRUE; v })
  for (b in seq_len(B)) {
    idx <- sched[b, ]
    for (k in seq_along(units)) {
      r <- idx[in_set[[k]][idx]]
      yy <- y[r]; pp <- P[r, units[[k]]$model]
      D[b, k, 1] <- ev_auc_fast(yy, pp)
      D[b, k, 2] <- ev_brier(yy, pp)
      D[b, k, 3:5] <- ev_calib_fast(yy, pp, clip)
      NB[b, k, ] <- ev_net_benefit(yy, pp, thr)
      TA[b, k, ] <- ev_treat_all(yy, thr)
    }
  }
  rows <- list(); roc <- list(); cal <- list(); dca <- list(); checks <- list()
  for (k in seq_along(units)) {
    u <- units[[k]]; r <- u$rows; yy <- y[r]; pp <- P[r, u$model]
    rh <- ev_rows_hash(ids, r)
    auc_pkg <- ev_auc_proc(yy, pp); auc_fast <- ev_auc_fast(yy, pp)
    cal_pkg <- ev_calib_glm(yy, pp, clip); cal_fast <- ev_calib_fast(yy, pp, clip)
    checks[[k]] <- data.frame(model = u$model, evaluation_set = u$set,
                              max_abs_diff_fast_vs_package = max(abs(c(auc_pkg - auc_fast, cal_pkg - cal_fast)), na.rm = TRUE))
    point <- c(auc = auc_pkg, brier = ev_brier(yy, pp), cal_pkg)
    npos <- sum(yy == 1); nneg <- sum(yy == 0)
    for (j in seq_along(nm)) {
      method <- if (nm[j] == "auc") "percentile_bootstrap_stratified_by_mets_liver (pROC point estimate)" else
        "percentile_bootstrap_stratified_by_mets_liver"
      rows[[length(rows) + 1]] <- ev_metric_row(cfg, "diagnostic", u$model, u$set, n, length(r), npos, nneg,
                                                nm[j], unname(point[nm[j]]), D[, k, j], method, rows_hash = rh)
    }
    # ROC coordinates (threshold values are kept out of public tables)
    if (npos > 0 && nneg > 0) {
      ro <- pROC::roc(yy, pp, direction = "<", quiet = TRUE, levels = c(0, 1))
      co <- pROC::coords(ro, "all", ret = c("specificity", "sensitivity"), transpose = FALSE)
      co <- co[order(1 - co$specificity, co$sensitivity), ]
      roc[[k]] <- data.frame(model = u$model, evaluation_set = u$set, point = seq_len(nrow(co)),
                             false_positive_rate = 1 - co$specificity, true_positive_rate = co$sensitivity,
                             n_evaluable = length(r), stringsAsFactors = FALSE)
    }
    # Grouped calibration: <= 3 quantile groups, >= 5 patients each, both outcomes present
    g <- NULL; ng <- NA
    if (npos > 0 && nneg > 0) {
      for (cand in c(cfg$evaluation$diagnostic_calibration_groups$quantile_groups,
                     ev_chr(cfg$evaluation$diagnostic_calibration_groups$fallback_group_counts))) {
        g <- ev_quantile_groups(pp, cand, cfg$evaluation$diagnostic_calibration_groups$min_group_n)
        if (!is.null(g)) { ng <- cand; break }
      }
    }
    if (is.null(g)) {
      cal[[k]] <- data.frame(model = u$model, evaluation_set = u$set, n_groups = NA, group = NA, n = NA,
                             n_events = NA, mean_predicted = NA, observed_proportion = NA, observed_lower_95 = NA,
                             observed_upper_95 = NA, ci_method = "wilson", status = "unsupported_small_sample",
                             limitation_code = if (npos == 0 || nneg == 0) "no_class_variation" else
                               "quantile_groups_not_formable_with_min_5", stringsAsFactors = FALSE)
    } else {
      cal[[k]] <- do.call(rbind, lapply(seq_len(ng), function(gi) {
        s <- g == gi; w <- ev_wilson(sum(yy[s]), sum(s), cfg$bootstrap$confidence_level)
        data.frame(model = u$model, evaluation_set = u$set, n_groups = ng, group = gi, n = sum(s),
                   n_events = sum(yy[s]), mean_predicted = mean(pp[s]), observed_proportion = mean(yy[s]),
                   observed_lower_95 = w[1], observed_upper_95 = w[2], ci_method = "wilson",
                   status = "supported", limitation_code = sprintf("quantile_groups_%d_apparent", ng),
                   stringsAsFactors = FALSE)
      }))
    }
    # DCA (full 0.01-0.99 table) with pointwise percentile intervals
    nb_pt <- ev_net_benefit(yy, pp, thr); ta_pt <- ev_treat_all(yy, thr)
    mk <- function(strategy, est, draws) {
      nv <- colSums(is.finite(draws))
      sup <- vapply(nv, ev_interval_support, "", B = B, rules = cfg$bootstrap$support_rules)
      ci <- t(apply(draws, 2, function(x) {
        x <- x[is.finite(x)]
        if (length(x)) ev_percentile_ci(x, cfg$bootstrap$confidence_level) else c(NA_real_, NA_real_)
      }))
      ci[sup == "not_estimable", ] <- NA
      data.frame(model = u$model, evaluation_set = u$set, strategy = strategy, threshold = thr, net_benefit = est,
                 ci_lower_95 = ci[, 1], ci_upper_95 = ci[, 2], bootstrap_draws_valid = nv, interval_support = sup,
                 n_evaluable = length(r), n_positive = npos, stringsAsFactors = FALSE)
    }
    dca[[k]] <- rbind(mk(u$model, nb_pt, NB[, k, ]), mk("Treat All", ta_pt, TA[, k, ]),
                      data.frame(model = u$model, evaluation_set = u$set, strategy = "Treat None", threshold = thr,
                                 net_benefit = 0, ci_lower_95 = NA_real_, ci_upper_95 = NA_real_,
                                 bootstrap_draws_valid = NA_integer_, interval_support = "not_applicable",
                                 n_evaluable = length(r), n_positive = npos, stringsAsFactors = FALSE))
  }
  draws <- list(metrics = D, metric_names = nm, net_benefit = NB, treat_all = TA, thresholds = thr,
                units = lapply(units, function(u) u[c("model", "set")]))
  list(metrics = as.data.frame(dplyr::bind_rows(rows)), roc = do.call(rbind, roc), calibration = do.call(rbind, cal),
       dca = do.call(rbind, dca), draws = draws, units = units, y = y, P = P, ids = ids,
       fast_vs_package = do.call(rbind, checks))
}

# ==================================================================================
# 9. Prognostic evaluation
# ==================================================================================

ev_km_multi <- function(time, event, hz) vapply(hz, function(t) ev_km_at(time, event, t)$est, numeric(1))

ev_evaluate_prognostic <- function(cfg, pp, sched, cutpoint) {
  keys <- ev_prog_keys(cfg); fig <- ev_prog_figure_keys(cfg)
  hz <- as.numeric(ev_chr(cfg$evaluation$time_horizons_months))
  base <- pp[pp$model == keys[1], ]; base <- base[order(base$row_index), ]
  ids <- base$patient_id
  time <- ifelse(base$os_months == 0, cfg$mapping$zero_survival_time_replacement, base$os_months)
  event <- base$os_event
  n <- length(time)
  LP <- ev_wide(pp, keys, "linear_predictor"); ok <- !is.na(LP)
  S <- lapply(hz, function(t) ev_wide(pp, fig, sprintf("surv_%d", t))); names(S) <- as.character(hz)
  c_units <- ev_units(ok, keys, list(common_all_prognostic_models = keys, common_primary_comparison = fig))
  h_units <- Filter(function(u) u$model %in% fig && u$set %in% c("model_specific", "common_primary_comparison"), c_units)
  B <- nrow(sched)
  Dc <- matrix(NA_real_, B, length(c_units))
  Da <- array(NA_real_, c(B, length(h_units), length(hz))); Db <- Da; Dg <- Da
  in_c <- lapply(c_units, function(u) { v <- rep(FALSE, n); v[u$rows] <- TRUE; v })
  in_h <- lapply(h_units, function(u) { v <- rep(FALSE, n); v[u$rows] <- TRUE; v })
  for (b in seq_len(B)) {
    idx <- sched[b, ]
    for (k in seq_along(c_units)) {
      r <- idx[in_c[[k]][idx]]
      Dc[b, k] <- ev_cindex(time[r], event[r], LP[r, c_units[[k]]$model])["c_index"]
    }
    for (k in seq_along(h_units)) {
      r <- idx[in_h[[k]][idx]]; m <- h_units[[k]]$model
      tt <- time[r]; ee <- event[r]
      km <- ev_km_multi(tt, ee, hz)
      for (j in seq_along(hz)) {
        st <- S[[j]][r, m]
        Da[b, k, j] <- ev_time_auc(tt, ee, LP[r, m], hz[j])$auc
        Db[b, k, j] <- ev_brier_ipcw(tt, ee, st, hz[j])
        Dg[b, k, j] <- km[j] - mean(st)
      }
    }
  }
  rows <- list(); rocs <- list(); cal <- list(); grp <- list(); dca <- list(); chk <- list()
  thr <- seq(cfg$evaluation$dca$thresholds$from, cfg$evaluation$dca$thresholds$to, by = cfg$evaluation$dca$thresholds$by)
  for (k in seq_along(c_units)) {
    u <- c_units[[k]]; r <- u$rows
    ci <- ev_cindex(time[r], event[r], LP[r, u$model])
    rows[[length(rows) + 1]] <- ev_metric_row(cfg, "prognostic", u$model, u$set, n, length(r), sum(event[r]),
                                              sum(event[r] == 0), "c_index", unname(ci["c_index"]), Dc[, k],
                                              "percentile_bootstrap_patient_level (survival::concordance point estimate)",
                                              rows_hash = ev_rows_hash(ids, r),
                                              extra = list(comparable_pairs = unname(ci["comparable_pairs"])))
  }
  dgp <- cfg$evaluation$prognostic_calibration_groups
  for (k in seq_along(h_units)) {
    u <- h_units[[k]]; r <- u$rows; m <- u$model; tt <- time[r]; ee <- event[r]; rh <- ev_rows_hash(ids, r)
    for (j in seq_along(hz)) {
      t <- hz[j]; st <- S[[j]][r, m]
      hc <- ev_horizon_counts(tt, ee, t)
      ncase <- unname(hc["n_cases"])
      ex <- list(n_cases_by_t = ncase, n_controls_beyond_t = unname(hc["n_controls"]),
                 n_censored_by_t = unname(hc["n_censored_by_t"]))
      ta <- ev_time_auc(tt, ee, LP[r, m], t, iid = TRUE)
      rows[[length(rows) + 1]] <- ev_metric_row(
        cfg, "prognostic", m, u$set, n, length(r), ncase, length(r) - ncase,
        "time_auc", ta$auc, Da[, k, j], "percentile_bootstrap_patient_level (timeROC marginal IPCW point estimate)",
        horizon = t, extra_limits = if (!is.na(ta$error)) ta$error else character(0), rows_hash = rh,
        extra = c(ex, list(iid_se = ta$se, iid_ci_lower_95 = ta$auc - 1.96 * ta$se,
                           iid_ci_upper_95 = ta$auc + 1.96 * ta$se)))
      if (!is.null(ta$TP)) {
        o <- order(ta$FP, ta$TP)
        rocs[[length(rocs) + 1]] <- data.frame(model = m, evaluation_set = u$set, horizon_months = t,
                                               point = seq_along(o), false_positive_rate = ta$FP[o],
                                               true_positive_rate = ta$TP[o], n_evaluable = length(r),
                                               stringsAsFactors = FALSE)
      }
      b_pkg <- ev_brier_score_pkg(tt, ee, st, t); b_fast <- ev_brier_ipcw(tt, ee, st, t)
      chk[[length(chk) + 1]] <- data.frame(model = m, evaluation_set = u$set, horizon_months = t,
                                           brier_score_pkg = b_pkg, brier_engine = b_fast,
                                           abs_diff = abs(b_pkg - b_fast))
      rows[[length(rows) + 1]] <- ev_metric_row(
        cfg, "prognostic", m, u$set, n, length(r), ncase, length(r) - ncase,
        "brier_ipcw", b_pkg, Db[, k, j],
        "percentile_bootstrap_patient_level (riskRegression::Score cens.model=km point estimate)",
        horizon = t, rows_hash = rh, extra = ex)
      km <- ev_km_at(tt, ee, t, cfg$evaluation$km$conf_type)
      mp <- mean(st); gap <- if (km$estimable) km$est - mp else NA_real_
      rows[[length(rows) + 1]] <- ev_metric_row(
        cfg, "prognostic", m, u$set, n, length(r), ncase, length(r) - ncase,
        "km_observed_survival", if (km$estimable) km$est else NA_real_, NULL, "greenwood_log_kaplan_meier",
        horizon = t, analytic_ci = c(km$lower, km$upper), rows_hash = rh, extra = ex)
      rows[[length(rows) + 1]] <- ev_metric_row(
        cfg, "prognostic", m, u$set, n, length(r), ncase, length(r) - ncase,
        "mean_predicted_survival", mp, NULL, "descriptive_no_interval", horizon = t, rows_hash = rh, extra = ex)
      rows[[length(rows) + 1]] <- ev_metric_row(
        cfg, "prognostic", m, u$set, n, length(r), ncase, length(r) - ncase,
        "calibration_gap_km_minus_predicted", gap, Dg[, k, j], "percentile_bootstrap_patient_level", horizon = t,
        rows_hash = rh, extra = ex)
      cal[[length(cal) + 1]] <- data.frame(
        model = m, evaluation_set = u$set, horizon_months = t, n_evaluable = length(r), n_cases_by_t = ncase,
        n_censored_by_t = unname(hc["n_censored_by_t"]), km_estimable = km$estimable,
        observed_km_survival = if (km$estimable) km$est else NA_real_, observed_km_lower_95 = km$lower,
        observed_km_upper_95 = km$upper, km_n_at_risk = km$n_risk, mean_predicted_survival = mp,
        calibration_gap = gap, stringsAsFactors = FALSE)
      # exploratory 3-group calibration (>= 3 per group, KM estimable in every group)
      g <- ev_quantile_groups(st, dgp$quantile_groups, dgp$min_group_n)
      gr <- NULL; why <- "quantile_groups_not_formable_with_min_3"
      if (!is.null(g)) {
        gr <- do.call(rbind, lapply(seq_len(dgp$quantile_groups), function(gi) {
          s <- g == gi; kg <- ev_km_at(tt[s], ee[s], t, cfg$evaluation$km$conf_type)
          data.frame(model = m, evaluation_set = u$set, horizon_months = t, group = gi, n = sum(s),
                     mean_predicted_survival = mean(st[s]), observed_km_survival = kg$est,
                     observed_km_lower_95 = kg$lower, observed_km_upper_95 = kg$upper, km_estimable = kg$estimable,
                     stringsAsFactors = FALSE)
        }))
        if (!all(gr$km_estimable)) { why <- "km_not_estimable_in_a_group"; gr <- NULL }
      }
      grp[[length(grp) + 1]] <- if (is.null(gr)) {
        data.frame(model = m, evaluation_set = u$set, horizon_months = t, group = NA_integer_, n = NA_integer_,
                   mean_predicted_survival = NA_real_, observed_km_survival = NA_real_, observed_km_lower_95 = NA_real_,
                   observed_km_upper_95 = NA_real_, km_estimable = FALSE, status = "unsupported_small_sample",
                   limitation_code = why, stringsAsFactors = FALSE)
      } else {
        cbind(gr, status = "exploratory_small_sample", limitation_code = "grouped_km_three_groups_n_ge_3",
              stringsAsFactors = FALSE)
      }
      # survival DCA (original estimand); unsupported below 20 eligible / 5 events
      el <- ev_dca_eligibility(tt, ee, t)
      ye <- el$y[el$eligible]; risk <- 1 - st[el$eligible]
      n_el <- sum(el$eligible); n_ev <- sum(ye)
      supported <- n_el >= cfg$evaluation$dca$prognostic_min_eligible && n_ev >= cfg$evaluation$dca$prognostic_min_events
      st_lab <- if (supported) "supported" else "unsupported_small_sample"
      lim <- if (supported) "none" else sprintf("n_eligible_%d_lt_%d_or_events_%d_lt_%d", n_el,
                                                cfg$evaluation$dca$prognostic_min_eligible, n_ev,
                                                cfg$evaluation$dca$prognostic_min_events)
      mkd <- function(strategy, nb) data.frame(model = m, evaluation_set = u$set, horizon_months = t,
                                               strategy = strategy, threshold = thr, net_benefit = nb,
                                               n_eligible = n_el, n_events_by_t = n_ev, status = st_lab,
                                               limitation_code = lim, stringsAsFactors = FALSE)
      dca[[length(dca) + 1]] <- if (n_el > 0) {
        rbind(mkd(m, ev_net_benefit(ye, risk, thr)), mkd("Treat All", ev_treat_all(ye, thr)),
              mkd("Treat None", 0 * thr))
      } else {
        mkd(m, NA_real_ * thr)
      }
    }
  }
  # KM risk groups with the frozen SEER training cutpoint (primary model, model-specific sample)
  km_model <- cfg$evaluation$km$lp_model
  r <- which(ok[, km_model]); lp <- LP[r, km_model]
  grp_lab <- factor(ifelse(lp <= cutpoint, "Low-risk", "High-risk"), levels = c("Low-risk", "High-risk"))
  km <- ev_km_groups(time[r], event[r], grp_lab, cfg)
  km$assignment_margin <- min(abs(lp - cutpoint))
  km$rows_hash <- ev_rows_hash(ids, r)
  km$n <- length(r)
  draws <- list(c_index = Dc, time_auc = Da, brier_ipcw = Db, calibration_gap = Dg, horizons = hz,
                c_units = lapply(c_units, function(u) u[c("model", "set")]),
                h_units = lapply(h_units, function(u) u[c("model", "set")]))
  list(metrics = as.data.frame(dplyr::bind_rows(rows)), roc = do.call(rbind, rocs), calibration = do.call(rbind, cal),
       grouped_calibration = do.call(rbind, grp), dca = do.call(rbind, dca), km = km, draws = draws,
       c_units = c_units, h_units = h_units, time = time, event = event, LP = LP, S = S, ids = ids,
       brier_check = do.call(rbind, chk), n = n)
}

ev_km_groups <- function(time, event, group, cfg) {
  d <- data.frame(time = time, event = event, risk_group = group)
  n_by <- table(group)
  out <- list(n_by_group = n_by)
  if (nrow(d) < cfg$evaluation$km$min_patients_for_plot) {
    out$status <- "unsupported_small_sample"; return(out)
  }
  fit <- survival::survfit(survival::Surv(time, event) ~ risk_group, data = d, conf.type = cfg$evaluation$km$conf_type)
  strata <- sub("risk_group=", "", names(fit$strata))
  curve <- data.frame(risk_group = rep(strata, fit$strata), time = fit$time, surv = fit$surv, lower = fit$lower,
                      upper = fit$upper, n_risk = fit$n.risk, n_event = fit$n.event, n_censor = fit$n.censor,
                      stringsAsFactors = FALSE)
  curve <- rbind(data.frame(risk_group = strata, time = 0, surv = 1, lower = 1, upper = 1,
                            n_risk = as.integer(table(d$risk_group)[strata]), n_event = 0, n_censor = 0), curve)
  curve <- curve[order(match(curve$risk_group, c("Low-risk", "High-risk")), curve$time), ]
  brk <- cfg$evaluation$km$risk_table_break_months
  x_max <- ceiling(max(time) / brk) * brk
  breaks <- seq(0, x_max, by = brk)
  rs <- summary(fit, times = breaks, extend = TRUE)
  risk <- data.frame(risk_group = sub("risk_group=", "", as.character(rs$strata)), time = rs$time,
                     n_at_risk = rs$n.risk, stringsAsFactors = FALSE)
  tab <- summary(fit)$table
  if (is.null(dim(tab))) tab <- t(as.matrix(tab))
  summ <- data.frame(risk_group = sub("risk_group=", "", rownames(tab)), n_patients = as.integer(tab[, "records"]),
                     n_events = as.integer(tab[, "events"]), median_os_months = as.numeric(tab[, "median"]),
                     median_os_lower_95 = as.numeric(tab[, "0.95LCL"]), median_os_upper_95 = as.numeric(tab[, "0.95UCL"]),
                     stringsAsFactors = FALSE)
  summ$median_os_status <- ifelse(is.na(summ$median_os_months), "not_reached", "estimated")
  both <- length(n_by) == 2 && all(n_by > 0)
  lr <- if (both && sum(event) > 0) {
    tryCatch(survival::survdiff(survival::Surv(time, event) ~ risk_group, data = d), error = function(e) NULL)
  } else NULL
  out$logrank_p <- if (!is.null(lr)) stats::pchisq(lr$chisq, df = 1, lower.tail = FALSE) else NA_real_
  out$logrank_chisq <- if (!is.null(lr)) lr$chisq else NA_real_
  out$logrank_status <- if (!is.null(lr)) "computed" else if (!both) "not_defined_empty_group" else "not_defined_no_events"
  out$curve <- curve; out$risk_table <- risk; out$summary <- summ; out$x_max <- x_max; out$breaks <- breaks
  out$status <- "supported"
  out
}

# ==================================================================================
# 10. Descriptive and registry tables
# ==================================================================================

ev_baseline_characteristics <- function(diag, prog) {
  out <- list()
  fmt_pct <- function(k, n) sprintf("%d (%.1f%%)", k, 100 * k / n)
  cat_spec <- list(
    sex_male = c("0" = "Female", "1" = "Male"), race = NULL, married = c("0" = "Not married", "1" = "Married"),
    histology_paper = NULL, histology_prognostic_group = NULL, grade = c("1" = "I", "2" = "II", "3" = "III", "4" = "IV"),
    primary_site_group = NULL, ajcc_t = NULL, ajcc_n = NULL, surgery_primary = c("0" = "No", "1" = "Yes"),
    surgery_lymph_node = c("0" = "No", "1" = "Yes"), radiotherapy = c("0" = "No", "1" = "Yes"),
    chemotherapy = c("0" = "No", "1" = "Yes"), tumor_size_group = NULL, mets_bone = c("0" = "No", "1" = "Yes"),
    mets_brain = c("0" = "No", "1" = "Yes"), mets_lung = c("0" = "No", "1" = "Yes"),
    mets_liver = c("0" = "No", "1" = "Yes"), os_event = c("0" = "Alive (censored)", "1" = "Dead"))
  cont <- c("age_at_diagnosis", "tumor_size_mm", "os_months")
  for (cn in c("diagnostic_all_patients", "prognostic_liver_metastasis_subset")) {
    df <- if (cn == "diagnostic_all_patients") diag else prog
    n <- nrow(df)
    out[[length(out) + 1]] <- data.frame(cohort = cn, variable = "N", level = "", n = n, percent = 100,
                                         median = NA, q1 = NA, q3 = NA, n_nonmissing = n, display = as.character(n))
    for (v in cont) {
      if (!v %in% names(df)) next
      x <- df[[v]][!is.na(df[[v]])]
      q <- stats::quantile(x, c(0.25, 0.5, 0.75), type = 7, names = FALSE)
      lab <- if (v == "tumor_size_mm") "exact_mm_entries_only" else ""
      out[[length(out) + 1]] <- data.frame(cohort = cn, variable = v, level = lab, n = NA, percent = NA,
                                           median = q[2], q1 = q[1], q3 = q[3], n_nonmissing = length(x),
                                           display = sprintf("%.1f [%.1f, %.1f]", q[2], q[1], q[3]))
    }
    for (v in names(cat_spec)) {
      if (!v %in% names(df)) next
      x <- as.character(df[[v]])
      if (!is.null(cat_spec[[v]])) x <- unname(cat_spec[[v]][x])
      x[is.na(x)] <- "Missing/unknown"
      tb <- table(x)
      for (lv in names(tb)) {
        out[[length(out) + 1]] <- data.frame(cohort = cn, variable = v, level = lv, n = as.integer(tb[lv]),
                                             percent = round(100 * tb[[lv]] / n, 1), median = NA, q1 = NA, q3 = NA,
                                             n_nonmissing = sum(x != "Missing/unknown"),
                                             display = fmt_pct(as.integer(tb[lv]), n))
      }
    }
  }
  res <- do.call(rbind, out)
  res$privacy_rule <- "one-variable marginal counts per cohort; continuous variables as median [Q1, Q3] without extremes"
  res
}

ev_model_registry <- function(cfg, paths) {
  rows <- list()
  for (s in ev_model_specs(cfg, paths)) {
    m <- readRDS(s$abs_path)
    rows[[length(rows) + 1]] <- ev_registry_row(s, m, s$abs_path, "frozen_model")
    if (!is.null(s$absolute_risk_object) && s$abs_risk_path != s$abs_path) {
      rows[[length(rows) + 1]] <- ev_registry_row(s, readRDS(s$abs_risk_path), s$abs_risk_path, "absolute_risk_object")
    }
  }
  reg <- do.call(rbind, rows)
  reg$r_version <- R.version.string
  reg
}

# ==================================================================================
# 11. Supportedness ledger
# ==================================================================================

ev_expected_ledger <- function(cfg) {
  dk <- ev_diag_keys(cfg); pk <- ev_prog_keys(cfg); fk <- ev_prog_figure_keys(cfg)
  hz <- as.numeric(ev_chr(cfg$evaluation$time_horizons_months))
  key <- function(task, model, set, result, h = NA) paste(task, model, set, result, ifelse(is.na(h), "na", h), sep = "|")
  k <- character(0)
  for (m in dk) for (s in c("model_specific", "common_all_diagnostic_models"))
    for (r in c("auc", "brier", "calibration_in_the_large", "calibration_slope", "calibration_intercept_joint",
                "roc_curve", "grouped_calibration", "dca_net_benefit_curve")) k <- c(k, key("diagnostic", m, s, r))
  for (m in dk) k <- c(k, key("diagnostic", m, "all_patients", "prediction_coverage"))
  for (m in pk) for (s in c("model_specific", "common_all_prognostic_models")) k <- c(k, key("prognostic", m, s, "c_index"))
  for (m in pk) k <- c(k, key("prognostic", m, "all_patients", "prediction_coverage"))
  for (m in fk) k <- c(k, key("prognostic", m, "common_primary_comparison", "c_index"))
  for (m in fk) for (s in c("model_specific", "common_primary_comparison")) for (h in hz)
    for (r in c("time_auc", "brier_ipcw", "km_observed_survival", "mean_predicted_survival",
                "calibration_gap_km_minus_predicted", "time_roc_curve", "grouped_calibration_exploratory",
                "survival_dca_net_benefit_curve")) k <- c(k, key("prognostic", m, s, r, h))
  km <- cfg$evaluation$km$lp_model
  for (r in c("km_risk_group_curves", "logrank_test", "median_os_low_risk", "median_os_high_risk"))
    k <- c(k, key("prognostic", km, "model_specific", r))
  k
}

ev_ledger_row <- function(task, model, set, result, h, n_eval, status, limitation, estimate_available,
                          interval_support, source_table) {
  data.frame(ledger_key = paste(task, model, set, result, ifelse(is.na(h), "na", h), sep = "|"), task = task,
             model = model, evaluation_set = set, result = result, horizon_months = h, n_evaluable = n_eval,
             support_status = status, limitation_code = limitation, estimate_available = estimate_available,
             interval_support = interval_support, source_table = source_table, stringsAsFactors = FALSE)
}

ev_build_ledger <- function(cfg, de, pe, compat) {
  L <- list()
  for (i in seq_len(nrow(de$metrics))) {
    r <- de$metrics[i, ]
    L[[length(L) + 1]] <- ev_ledger_row("diagnostic", r$model, r$evaluation_set, r$metric, NA, r$n_evaluable,
                                        r$support_status, r$limitation_code, is.finite(r$estimate), r$interval_support,
                                        "tables/diagnostic_external_metrics.csv")
  }
  for (u in de$units) {
    auc <- de$metrics[de$metrics$model == u$model & de$metrics$evaluation_set == u$set & de$metrics$metric == "auc", ]
    has_roc <- any(de$roc$model == u$model & de$roc$evaluation_set == u$set)
    L[[length(L) + 1]] <- ev_ledger_row("diagnostic", u$model, u$set, "roc_curve", NA, length(u$rows),
                                        if (has_roc) "supported" else "not_estimable_no_class_variation",
                                        if (has_roc) "coordinates_without_threshold_values" else "no_class_variation",
                                        has_roc, NA, "tables/diagnostic_external_roc_curve.csv")
    cal <- de$calibration[de$calibration$model == u$model & de$calibration$evaluation_set == u$set, ]
    L[[length(L) + 1]] <- ev_ledger_row("diagnostic", u$model, u$set, "grouped_calibration", NA, length(u$rows),
                                        cal$status[1], cal$limitation_code[1], cal$status[1] == "supported", NA,
                                        "tables/diagnostic_external_calibration.csv")
    d <- de$dca[de$dca$model == u$model & de$dca$evaluation_set == u$set & de$dca$strategy == u$model, ]
    sup <- if (all(d$interval_support == "supported")) "supported" else "fragile_interval"
    L[[length(L) + 1]] <- ev_ledger_row("diagnostic", u$model, u$set, "dca_net_benefit_curve", NA, length(u$rows),
                                        sup, sprintf("pointwise_intervals_supported_%d_of_%d_thresholds",
                                                     sum(d$interval_support == "supported"), nrow(d)),
                                        TRUE, paste(unique(d$interval_support), collapse = ";"),
                                        "tables/diagnostic_external_dca.csv")
  }
  for (i in seq_len(nrow(compat))) {
    r <- compat[i, ]
    st <- if (r$n_ok == 0 && r$n_incompatible_level == 0) "not_evaluable_missing_predictor" else
      if (r$n_incompatible_level > 0) "not_estimable_model_incompatibility" else "supported"
    lim <- sprintf("ok_%d_missing_predictor_%d_incompatible_level_%d", r$n_ok, r$n_missing_predictor, r$n_incompatible_level)
    L[[length(L) + 1]] <- ev_ledger_row(r$task, r$model, "all_patients", "prediction_coverage", NA, r$n_ok, st, lim,
                                        r$n_ok > 0, NA, "tables/external_model_level_compatibility.csv")
  }
  for (i in seq_len(nrow(pe$metrics))) {
    r <- pe$metrics[i, ]
    L[[length(L) + 1]] <- ev_ledger_row("prognostic", r$model, r$evaluation_set, r$metric, r$horizon_months,
                                        r$n_evaluable, r$support_status, r$limitation_code, is.finite(r$estimate),
                                        r$interval_support,
                                        if (r$metric == "c_index") "tables/prognostic_external_discrimination.csv" else
                                          if (r$metric == "time_auc") "tables/prognostic_external_time_auc.csv" else
                                            "tables/prognostic_external_calibration.csv")
  }
  for (u in pe$h_units) for (h in pe$draws$horizons) {
    ta <- pe$metrics[pe$metrics$model == u$model & pe$metrics$evaluation_set == u$set & pe$metrics$metric == "time_auc" &
                       pe$metrics$horizon_months == h, ]
    has <- any(pe$roc$model == u$model & pe$roc$evaluation_set == u$set & pe$roc$horizon_months == h)
    L[[length(L) + 1]] <- ev_ledger_row("prognostic", u$model, u$set, "time_roc_curve", h, length(u$rows),
                                        if (has) ta$support_status else "not_estimable_no_event_structure",
                                        ta$limitation_code, has, ta$interval_support, "tables/prognostic_external_roc_curve.csv")
    g <- pe$grouped_calibration[pe$grouped_calibration$model == u$model & pe$grouped_calibration$evaluation_set == u$set &
                                  pe$grouped_calibration$horizon_months == h, ]
    L[[length(L) + 1]] <- ev_ledger_row("prognostic", u$model, u$set, "grouped_calibration_exploratory", h,
                                        length(u$rows), g$status[1], g$limitation_code[1],
                                        g$status[1] == "exploratory_small_sample", NA,
                                        "tables/prognostic_external_calibration.csv")
    d <- pe$dca[pe$dca$model == u$model & pe$dca$evaluation_set == u$set & pe$dca$horizon_months == h, ]
    L[[length(L) + 1]] <- ev_ledger_row("prognostic", u$model, u$set, "survival_dca_net_benefit_curve", h,
                                        d$n_eligible[1], d$status[1], d$limitation_code[1],
                                        any(is.finite(d$net_benefit[d$strategy == u$model])), "not_computed",
                                        "tables/prognostic_external_dca.csv")
  }
  km <- pe$km; m <- cfg$evaluation$km$lp_model
  kst <- km$status
  L[[length(L) + 1]] <- ev_ledger_row("prognostic", m, "model_specific", "km_risk_group_curves", NA, km$n, kst,
                                      "frozen_seer_training_cutpoint_applied_unchanged;n_evaluable_lt_20",
                                      kst == "supported", "greenwood_log_pointwise", "tables/prognostic_external_km_curve.csv")
  L[[length(L) + 1]] <- ev_ledger_row("prognostic", m, "model_specific", "logrank_test", NA, km$n,
                                      if (km$logrank_status == "computed") "supported" else "not_estimable_no_event_structure",
                                      paste0(km$logrank_status, ";n_evaluable_lt_20"), km$logrank_status == "computed", NA,
                                      "tables/prognostic_external_km_summary.csv")
  for (g in c("Low-risk", "High-risk")) {
    s <- km$summary[km$summary$risk_group == g, ]
    ok <- nrow(s) == 1 && is.finite(s$median_os_months)
    L[[length(L) + 1]] <- ev_ledger_row("prognostic", m, "model_specific",
                                        if (g == "Low-risk") "median_os_low_risk" else "median_os_high_risk", NA,
                                        if (nrow(s)) s$n_patients else 0,
                                        if (ok) "supported" else "not_estimable_no_event_structure",
                                        if (ok) "n_evaluable_lt_20" else "median_not_reached_or_empty_group", ok, NA,
                                        "tables/prognostic_external_km_summary.csv")
  }
  led <- do.call(rbind, L)
  led <- led[!duplicated(led$ledger_key), ]
  led
}

# ==================================================================================
# 12. Internal (SEER) context - copied, never re-estimated
# ==================================================================================

ev_internal_context <- function(cfg, paths, de, pe) {
  src <- cfg$internal_context_sources
  get_src <- function(file) {
    p <- ev_repo(paths, file)
    if (!file.exists(p)) return(NULL)
    list(df = utils::read.csv(p, stringsAsFactors = FALSE), sha = ev_sha256_file(p))
  }
  ext <- function(task, model, set, metric, h = NA) {
    M <- if (task == "diagnostic") de$metrics else pe$metrics
    r <- M[M$model == model & M$evaluation_set == set & M$metric == metric &
             (if (is.na(h)) is.na(M$horizon_months) else M$horizon_months %in% h), ]
    if (!nrow(r)) return(list(est = NA, lo = NA, hi = NA, n = NA, st = "not_computed_externally", set = set))
    list(est = r$estimate[1], lo = r$ci_lower_95[1], hi = r$ci_upper_95[1], n = r$n_evaluable[1],
         st = r$support_status[1], set = set)
  }
  row <- function(s, task, model, metric, h, ie, ilo, ihi, inn, sha, e, note) {
    data.frame(task = task, model = model, metric = metric, horizon_months = h,
               internal_cohort_label = s$cohort_label, internal_design = s$design, internal_estimate = ie,
               internal_ci_lower_95 = ilo, internal_ci_upper_95 = ihi, internal_n = inn,
               internal_source_file = s$file, internal_source_sha256 = sha, internal_source_status = "copied_unchanged",
               external_evaluation_set = e$set, external_estimate = e$est, external_ci_lower_95 = e$lo,
               external_ci_upper_95 = e$hi, external_n = e$n, external_support_status = e$st,
               comparability = "not_equivalent", comparability_note = note,
               difference_test = "none_performed", stringsAsFactors = FALSE)
  }
  note_d <- "different cohorts and designs: SEER apparent performance on the fitting data vs. frozen-model external performance; no difference test"
  note_p <- "different cohorts and designs: SEER training-cohort apparent estimate vs. frozen-model external estimate on a different, much smaller cohort; no difference test"
  out <- list()
  for (s in src) {
    g <- get_src(s$file)
    if (is.null(g)) {
      out[[length(out) + 1]] <- data.frame(task = s$task, model = NA, metric = NA, horizon_months = NA,
                                           internal_cohort_label = s$cohort_label, internal_design = s$design,
                                           internal_estimate = NA, internal_ci_lower_95 = NA, internal_ci_upper_95 = NA,
                                           internal_n = NA, internal_source_file = s$file, internal_source_sha256 = NA,
                                           internal_source_status = "source_file_absent_not_recomputed",
                                           external_evaluation_set = NA, external_estimate = NA,
                                           external_ci_lower_95 = NA, external_ci_upper_95 = NA, external_n = NA,
                                           external_support_status = NA, comparability = "not_equivalent",
                                           comparability_note = "internal source table not present in the repository; not recomputed",
                                           difference_test = "none_performed", stringsAsFactors = FALSE)
      next
    }
    d <- g$df; f <- basename(s$file)
    if (f == "roc_auc_diagnostic.csv") {
      for (i in seq_len(nrow(d))) out[[length(out) + 1]] <- row(s, "diagnostic", d$model[i], "auc", NA, d$auc[i],
                                                             d$ci_lower_95[i], d$ci_upper_95[i], d$n[i], g$sha,
                                                             ext("diagnostic", d$model[i], "common_all_diagnostic_models", "auc"), note_d)
    } else if (f == "calibration_metrics_diagnostic.csv") {
      mp <- c(brier_score = "brier", calibration_intercept = "calibration_intercept_joint", calibration_slope = "calibration_slope")
      for (i in seq_len(nrow(d))) out[[length(out) + 1]] <- row(s, "diagnostic", "nomogram_figure", mp[[d$metric[i]]], NA,
                                                             d$value[i], NA, NA, 2672, g$sha,
                                                             ext("diagnostic", "nomogram_figure", "common_all_diagnostic_models", mp[[d$metric[i]]]), note_d)
    } else if (f == "multivariable_cox_model_summary.csv") {
      for (i in seq_len(nrow(d))) out[[length(out) + 1]] <- row(s, "prognostic", d$model[i], "c_index", NA, d$concordance[i],
                                                             d$c_index_lower[i], d$c_index_upper[i], d$n_obs[i], g$sha,
                                                             ext("prognostic", d$model[i], "model_specific", "c_index"), note_p)
    } else if (f == "calibration_survival_metrics.csv") {
      for (i in seq_len(nrow(d))) for (mt in c("cal_slope", "cal_intercept")) {
        out[[length(out) + 1]] <- row(s, "prognostic", "nomogram_figure", paste0("grouped_", mt, "_", gsub(" ", "_", tolower(d$cohort[i]))),
                                      d$time_months[i], d[[mt]][i], NA, NA, NA, g$sha,
                                      list(est = NA, lo = NA, hi = NA, n = NA, st = "not_computed_externally_by_prespecification",
                                           set = NA),
                                      "external analysis pre-specifies no calibration slope from 11 patients; see calibration_gap_km_minus_predicted instead")
      }
    } else if (f == "km_risk_group_summary.csv") {
      for (i in seq_len(nrow(d))) {
        es <- pe$km$summary[pe$km$summary$risk_group == d$risk_group[i], ]
        for (mt in c("n_patients", "n_events", "median_os")) {
          ev <- if (!nrow(es)) NA else switch(mt, n_patients = es$n_patients, n_events = es$n_events, median_os = es$median_os_months)
          out[[length(out) + 1]] <- row(s, "prognostic", "nomogram_figure", paste0(mt, "_", gsub("-", "_", tolower(d$risk_group[i])), "_", gsub(" ", "_", tolower(d$cohort[i]))),
                                        NA, d[[mt]][i], NA, NA, NA, g$sha,
                                        list(est = ev, lo = NA, hi = NA, n = pe$km$n, st = pe$km$status, set = "model_specific"),
                                        "same frozen SEER training cutpoint; different cohorts")
        }
      }
    }
  }
  do.call(rbind, out)
}

# ==================================================================================
# 13. Report summaries (numeric fields and statuses only)
# ==================================================================================

ev_fmt <- function(x, d = 3) ifelse(is.finite(x), formatC(x, format = "f", digits = d), "NA")

ev_display <- function(r, d = 3) {
  if (!nrow(r) || !is.finite(r$estimate)) return("not estimable")
  if (isTRUE(r$ci_displayable) && r$ci_method != "descriptive_no_interval")
    return(sprintf("%s (%s to %s)", ev_fmt(r$estimate, d), ev_fmt(r$ci_lower_95, d), ev_fmt(r$ci_upper_95, d)))
  if (r$support_status == "fragile_interval") return(sprintf("%s (95%% CI fragile; not displayed)", ev_fmt(r$estimate, d)))
  if (r$ci_method == "descriptive_no_interval") return(ev_fmt(r$estimate, d))
  sprintf("%s (95%% CI not estimable)", ev_fmt(r$estimate, d))
}

ev_report_summary_diag <- function(cfg, de) {
  out <- list()
  for (m in cfg$models$diagnostic) {
    M <- de$metrics[de$metrics$model == m$key & de$metrics$evaluation_set == "common_all_diagnostic_models", ]
    row <- data.frame(model = m$key, model_label = m$label, role = m$role, origin = m$origin,
                      evaluation_set = "common_all_diagnostic_models", n_evaluable = M$n_evaluable[1],
                      n_positive = M$n_events_or_positive[1], n_negative = M$n_nonevents_or_censored[1],
                      stringsAsFactors = FALSE)
    for (mt in c("auc", "brier", "calibration_in_the_large", "calibration_slope")) {
      r <- M[M$metric == mt, ]
      row[[mt]] <- round(r$estimate, 4); row[[paste0(mt, "_ci_lower")]] <- round(r$ci_lower_95, 4)
      row[[paste0(mt, "_ci_upper")]] <- round(r$ci_upper_95, 4); row[[paste0(mt, "_status")]] <- r$support_status
      row[[paste0(mt, "_display")]] <- ev_display(r)
    }
    cal <- de$calibration[de$calibration$model == m$key & de$calibration$evaluation_set == "common_all_diagnostic_models", ]
    row$grouped_calibration_groups <- if (cal$status[1] == "supported") nrow(cal) else 0
    row$grouped_calibration_status <- cal$status[1]
    out[[length(out) + 1]] <- row
  }
  do.call(rbind, out)
}

ev_report_summary_prog <- function(cfg, pe) {
  out <- list()
  for (m in cfg$models$prognostic) {
    for (set in c("common_primary_comparison", "model_specific")) {
      if (set == "common_primary_comparison" && !isTRUE(m$figure_level)) next
      M <- pe$metrics[pe$metrics$model == m$key & pe$metrics$evaluation_set == set, ]
      ci <- M[M$metric == "c_index", ]
      hs <- if (isTRUE(m$figure_level)) pe$draws$horizons else NA
      for (h in hs) {
        row <- data.frame(model = m$key, model_label = m$label, role = m$role, evaluation_set = set,
                          n_evaluable = ci$n_evaluable, n_deaths = ci$n_events_or_positive,
                          c_index = round(ci$estimate, 4), c_index_ci_lower = round(ci$ci_lower_95, 4),
                          c_index_ci_upper = round(ci$ci_upper_95, 4), c_index_status = ci$support_status,
                          c_index_display = ev_display(ci), horizon_months = h, stringsAsFactors = FALSE)
        if (!is.na(h)) {
          for (mt in c("time_auc", "brier_ipcw", "km_observed_survival", "mean_predicted_survival",
                       "calibration_gap_km_minus_predicted")) {
            r <- M[M$metric == mt & M$horizon_months %in% h, ]
            row[[mt]] <- round(r$estimate, 4); row[[paste0(mt, "_ci_lower")]] <- round(r$ci_lower_95, 4)
            row[[paste0(mt, "_ci_upper")]] <- round(r$ci_upper_95, 4); row[[paste0(mt, "_status")]] <- r$support_status
            row[[paste0(mt, "_display")]] <- ev_display(r)
          }
          d <- pe$dca[pe$dca$model == m$key & pe$dca$evaluation_set == set & pe$dca$horizon_months == h, ]
          row$survival_dca_status <- d$status[1]; row$survival_dca_n_eligible <- d$n_eligible[1]
          row$survival_dca_events <- d$n_events_by_t[1]
        }
        out[[length(out) + 1]] <- row
      }
    }
  }
  res <- as.data.frame(dplyr::bind_rows(out))
  km <- pe$km$summary
  attr(res, "km") <- km
  res
}

# ==================================================================================
# 14. Figures - same format as the reference implementations (01_RefPaper_*):
#     10_combined_diagnostic_figures.R, 07_time_dependent_roc.R, 09_survival_dca.R,
#     10_kaplan_meier_risk_groups.R. Every plot is drawn ONLY from its figure data.
# ==================================================================================

ev_label <- function(cfg, task, key) {
  if (key %in% c("Treat All", "Treat None")) return(key)
  Filter(function(m) m$key == key, cfg$models[[task]])[[1]]$label
}

ev_short <- function(cfg, task, key) {
  if (key %in% c("Treat All", "Treat None")) return(key)
  Filter(function(m) m$key == key, cfg$models[[task]])[[1]]$short_label
}

ev_wrap <- function(x, width = 62) paste(strwrap(x, width = width), collapse = "\n")

ev_r3 <- function(x) as.character(round(x, 3))

# Interval text: shown only when the pre-specified support rule passes.
ev_ci_text <- function(r, style) {
  if (!nrow(r) || !is.finite(r$estimate)) return(" not estimable")
  if (isTRUE(r$ci_displayable)) {
    if (style == "diagnostic") return(sprintf(" (%s–%s)", ev_r3(r$ci_lower_95), ev_r3(r$ci_upper_95)))
    return(sprintf(" (95%% CI: %s–%s)", ev_r3(r$ci_lower_95), ev_r3(r$ci_upper_95)))
  }
  if (r$support_status == "fragile_interval") return(" (95% CI fragile; not shown)")
  " (95% CI not estimable)"
}

# Diagnostic legend as fig3: "<label> | AUC=0.903 (0.888-0.917)", wrapped at 62 characters.
ev_diag_legend <- function(cfg, key, r) {
  est <- if (nrow(r) && is.finite(r$estimate)) ev_r3(r$estimate) else "NA"
  ev_wrap(paste0(ev_label(cfg, "diagnostic", key), " | AUC=", est, ev_ci_text(r, "diagnostic")), 62)
}

# Prognostic legend as fig6: "<label>\n  AUC = 0.889 (95% CI: 0.842-0.936)".
ev_prog_legend <- function(cfg, key, r, what = "AUC") {
  est <- if (nrow(r) && is.finite(r$estimate)) ev_r3(r$estimate) else "NA"
  paste0(ev_label(cfg, "prognostic", key), "\n  ", what, " = ", est, ev_ci_text(r, "prognostic"))
}

ev_pal <- function(cfg, keys) ev_scale_vals(cfg, "palette", keys)
ev_lty <- function(cfg, keys) ev_scale_vals(cfg, "linetypes", keys)

# ---- diagnostic --------------------------------------------------------------------
ev_figdata_diag_roc <- function(cfg, de, models, set, panel = "ROC") {
  roc <- de$roc[de$roc$evaluation_set == set & de$roc$model %in% models, ]
  M <- de$metrics[de$metrics$evaluation_set == set & de$metrics$metric == "auc", ]
  lab <- vapply(models, function(m) ev_diag_legend(cfg, m, M[M$model == m, ]), "")
  rbind(ev_fig_data(panel, "curve", roc$model, roc$false_positive_rate, roc$true_positive_rate, n = roc$n_evaluable,
                    label = lab[roc$model], status = M$support_status[match(roc$model, M$model)]),
        ev_fig_data(panel, "reference", "diagonal", c(0, 1), c(0, 1)))
}

ev_plot_roc <- function(fd, cfg, task, title, subtitle, tag = NULL) {
  cur <- fd[fd$layer == "curve", ]; ref <- fd[fd$layer == "reference", ]
  if (!nrow(cur)) return(ev_not_estimable_panel(title, "no ROC curve estimable", cfg, task))
  ser <- unique(cur$series); lab <- cur$label[match(ser, cur$series)]
  ggplot2::ggplot() +
    ggplot2::geom_path(data = cur, ggplot2::aes(x = x, y = y, color = series),
                       linewidth = cfg$figures$style[[task]]$line_width) +
    ggplot2::geom_line(data = ref, ggplot2::aes(x = x, y = y), color = "grey60", linetype = "dashed", linewidth = 0.5) +
    ggplot2::scale_color_manual(values = ev_pal(cfg, ser), breaks = ser, labels = lab, name = NULL,
                                guide = ggplot2::guide_legend(ncol = 1, byrow = TRUE)) +
    ggplot2::coord_equal() +
    ggplot2::labs(tag = tag, title = title, subtitle = subtitle,
                  x = if (task == "diagnostic") "1 − Specificity (FPR)" else "1 - Specificity (FPR)",
                  y = "Sensitivity (TPR)") +
    ev_theme(cfg, task)
}

ev_figdata_diag_cal <- function(cfg, de, models, set, panel_by_model = TRUE) {
  cal <- de$calibration[de$calibration$evaluation_set == set & de$calibration$model %in% models, ]
  rows <- list()
  for (m in models) {
    c1 <- cal[cal$model == m, ]
    pn <- if (panel_by_model) m else "Calibration"
    if (c1$status[1] == "supported") {
      rows[[length(rows) + 1]] <- ev_fig_data(pn, "group_point", m, c1$mean_predicted, c1$observed_proportion,
                                              c1$observed_lower_95, c1$observed_upper_95, c1$n,
                                              sprintf("n=%d", c1$n), c1$status)
    } else {
      rows[[length(rows) + 1]] <- ev_fig_data(pn, "not_estimable", m, NA, NA, label = c1$limitation_code[1],
                                              status = c1$status[1])
    }
    rows[[length(rows) + 1]] <- ev_fig_data(pn, "reference", "Ideal", c(0, 1), c(0, 1))
  }
  do.call(rbind, rows)
}

# Calibration panel as fig3 B: Ideal (gray50, dashed) and Apparent (steelblue, points). External
# data give grouped apparent points only (no bias-corrected curve: nothing is refitted).
ev_plot_diag_cal <- function(fd, cfg, title, subtitle, tag = NULL) {
  pts <- fd[fd$layer == "group_point", ]; ref <- fd[fd$layer == "reference", ]
  if (!nrow(pts)) return(ev_not_estimable_panel(title, paste(unique(fd$label[fd$layer == "not_estimable"]), collapse = "; "), cfg))
  app_lab <- sprintf("Apparent (external, %d quantile groups; 95%% Wilson CI)", nrow(pts))
  long <- rbind(data.frame(x = ref$x[1:2], y = ref$y[1:2], curve = "Ideal"),
                data.frame(x = pts$x, y = pts$y, curve = app_lab))
  long$curve <- factor(long$curve, levels = c("Ideal", app_lab))
  cols <- stats::setNames(c("gray50", "steelblue"), c("Ideal", app_lab))
  ltys <- stats::setNames(c("dashed", "solid"), c("Ideal", app_lab))
  lwds <- stats::setNames(c(0.8, 1.2), c("Ideal", app_lab))
  ggplot2::ggplot(long, ggplot2::aes(x = x, y = y, color = curve, linetype = curve, linewidth = curve)) +
    ggplot2::geom_line() +
    ggplot2::geom_errorbar(data = pts, ggplot2::aes(x = x, ymin = ymin, ymax = ymax), inherit.aes = FALSE,
                           color = "steelblue", width = 0.015, linewidth = 0.4) +
    ggplot2::geom_point(data = long[long$curve != "Ideal", ], size = 1.8, show.legend = FALSE) +
    ggplot2::scale_color_manual(values = cols, name = NULL) +
    ggplot2::scale_linetype_manual(values = ltys, name = NULL) +
    ggplot2::scale_linewidth_manual(values = lwds, name = NULL) +
    ggplot2::coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
    ggplot2::labs(tag = tag, title = title, subtitle = subtitle, x = "Predicted Pr{liver=1}", y = "Actual Probability") +
    ev_theme(cfg, "diagnostic") +
    ggplot2::theme(legend.text = ggplot2::element_text(size = 9)) +
    ggplot2::guides(linewidth = "none")
}

ev_figdata_diag_dca <- function(cfg, de, models, set, panel = "DCA") {
  d <- de$dca[de$dca$evaluation_set == set & de$dca$model %in% models, ]
  ref <- d[d$model == models[1] & d$strategy %in% c("Treat All", "Treat None"), ]
  mod <- d[d$strategy %in% models & d$strategy == d$model, ]
  all <- rbind(mod, ref)
  ev_fig_data(panel, "net_benefit", all$strategy, all$threshold, all$net_benefit, all$ci_lower_95, all$ci_upper_95,
              all$n_evaluable, vapply(all$strategy, function(s) ev_wrap(ev_label(cfg, "diagnostic", s), 62), ""),
              all$interval_support)
}

# DCA panel as fig3 C / fig7: model curves solid, Treat All #333333 dashed, Treat None #999999 dotted,
# y range (-0.02, max(0.05, ceiling(max NB * 110) / 100)), full threshold range.
ev_plot_dca <- function(fd, cfg, task, title, subtitle, tag = NULL) {
  nb <- fd[fd$layer == "net_benefit" & is.finite(fd$y), ]
  if (!nrow(nb)) return(ev_not_estimable_panel(title, "no eligible patients", cfg, task))
  ser <- unique(nb$series); lab <- nb$label[match(ser, nb$series)]
  max_nb <- max(nb$y[nb$series != "Treat None"], na.rm = TRUE)
  y_upper <- max(0.05, ceiling(max_nb * 110) / 100)
  nb$series <- factor(nb$series, levels = ser)
  ggplot2::ggplot(nb, ggplot2::aes(x = x, y = y, color = series, linetype = series)) +
    ggplot2::geom_line(linewidth = cfg$figures$style[[task]]$dca_line_width) +
    ggplot2::scale_color_manual(values = ev_pal(cfg, ser), labels = lab, name = NULL,
                                guide = ggplot2::guide_legend(ncol = 1, byrow = TRUE)) +
    ggplot2::scale_linetype_manual(values = ev_lty(cfg, ser), labels = lab, name = NULL,
                                   guide = ggplot2::guide_legend(ncol = 1, byrow = TRUE)) +
    ggplot2::coord_cartesian(ylim = c(-0.02, y_upper)) +
    ggplot2::labs(tag = tag, title = title, subtitle = subtitle, x = "Threshold Probability", y = "Net Benefit") +
    ev_theme(cfg, task)
}

# ---- prognostic --------------------------------------------------------------------
ev_hz_label <- function(cfg, h) {
  hz <- as.numeric(ev_chr(cfg$evaluation$time_horizons_months))
  ev_chr(cfg$evaluation$time_horizon_labels)[match(h, hz)]
}

ev_figdata_prog_roc <- function(cfg, pe, set) {
  fk <- ev_prog_figure_keys(cfg); out <- list()
  for (h in pe$draws$horizons) {
    pn <- sprintf("%d months", h)
    M <- pe$metrics[pe$metrics$evaluation_set == set & pe$metrics$metric == "time_auc" & pe$metrics$horizon_months %in% h, ]
    for (m in fk) {
      r <- pe$roc[pe$roc$model == m & pe$roc$evaluation_set == set & pe$roc$horizon_months == h, ]
      lab <- ev_prog_legend(cfg, m, M[M$model == m, ], "AUC")
      if (nrow(r)) out[[length(out) + 1]] <- ev_fig_data(pn, "curve", m, r$false_positive_rate, r$true_positive_rate,
                                                         n = r$n_evaluable, label = lab, status = M$support_status[M$model == m])
      else out[[length(out) + 1]] <- ev_fig_data(pn, "not_estimable", m, NA, NA, label = M$limitation_code[M$model == m],
                                                 status = M$support_status[M$model == m])
    }
    out[[length(out) + 1]] <- ev_fig_data(pn, "reference", "diagonal", c(0, 1), c(0, 1))
    mm <- M[M$model == fk[1], ]
    out[[length(out) + 1]] <- ev_fig_data(pn, "counts", "horizon_counts", NA, NA, n = mm$n_evaluable,
                                          label = sprintf("N=%d | Deaths by %d mo=%d | Alive beyond=%d | Censored by %d mo=%d",
                                                          mm$n_evaluable, h, mm$n_cases_by_t, mm$n_controls_beyond_t, h, mm$n_censored_by_t))
  }
  do.call(rbind, out)
}

ev_figdata_prog_cal <- function(cfg, pe, set) {
  fk <- ev_prog_figure_keys(cfg); out <- list()
  for (h in pe$draws$horizons) {
    pn <- sprintf("%d months", h)
    for (m in fk) {
      c1 <- pe$calibration[pe$calibration$model == m & pe$calibration$evaluation_set == set & pe$calibration$horizon_months == h, ]
      gap <- pe$metrics[pe$metrics$model == m & pe$metrics$evaluation_set == set & pe$metrics$horizon_months %in% h &
                          pe$metrics$metric == "calibration_gap_km_minus_predicted", ]
      if (isTRUE(c1$km_estimable)) {
        out[[length(out) + 1]] <- ev_fig_data(pn, "overall_point", m, c1$mean_predicted_survival, c1$observed_km_survival,
                                              c1$observed_km_lower_95, c1$observed_km_upper_95, c1$n_evaluable,
                                              ev_prog_legend(cfg, m, gap, "Observed KM - mean predicted"), gap$support_status)
      } else {
        out[[length(out) + 1]] <- ev_fig_data(pn, "not_estimable", m, NA, NA, label = "km_not_estimable_at_horizon",
                                              status = "not_estimable_no_event_structure")
      }
      g <- pe$grouped_calibration[pe$grouped_calibration$model == m & pe$grouped_calibration$evaluation_set == set &
                                    pe$grouped_calibration$horizon_months == h, ]
      if (g$status[1] == "exploratory_small_sample") {
        out[[length(out) + 1]] <- ev_fig_data(pn, "group_point_exploratory", m, g$mean_predicted_survival,
                                              g$observed_km_survival, g$observed_km_lower_95, g$observed_km_upper_95,
                                              g$n, sprintf("n=%d", g$n), g$status)
      } else {
        out[[length(out) + 1]] <- ev_fig_data(pn, "group_not_estimable", m, NA, NA, label = g$limitation_code[1],
                                              status = g$status[1])
      }
    }
    out[[length(out) + 1]] <- ev_fig_data(pn, "reference", "Ideal", c(0, 1), c(0, 1))
  }
  do.call(rbind, out)
}

ev_plot_prog_cal_panel <- function(fd, cfg, title, subtitle) {
  ov <- fd[fd$layer == "overall_point", ]; gp <- fd[fd$layer == "group_point_exploratory", ]
  ref <- fd[fd$layer == "reference", ]
  if (!nrow(ov) && !nrow(gp)) return(ev_not_estimable_panel(title, "Kaplan-Meier survival not estimable at horizon", cfg, "prognostic"))
  ser <- unique(c(ov$series, gp$series))
  lab <- vapply(ser, function(s) { l <- ov$label[ov$series == s]; if (length(l)) l[1] else ev_label(cfg, "prognostic", s) }, "")
  ggplot2::ggplot() +
    ggplot2::geom_line(data = ref, ggplot2::aes(x = x, y = y), color = "gray50", linetype = "dashed", linewidth = 0.8) +
    ggplot2::geom_errorbar(data = ov, ggplot2::aes(x = x, ymin = ymin, ymax = ymax, color = series), width = 0.015,
                           linewidth = 0.5, show.legend = FALSE) +
    ggplot2::geom_point(data = gp, ggplot2::aes(x = x, y = y, color = series), shape = 1, size = 2.2, stroke = 0.8,
                        show.legend = FALSE) +
    ggplot2::geom_point(data = ov, ggplot2::aes(x = x, y = y, color = series), size = 2.6) +
    ggplot2::scale_color_manual(values = ev_pal(cfg, ser), breaks = ser, labels = lab, name = NULL,
                                guide = ggplot2::guide_legend(ncol = 1, byrow = TRUE)) +
    ggplot2::coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
    ggplot2::labs(title = title, subtitle = subtitle, x = "Predicted survival probability",
                  y = "Observed survival probability (KM, 95% CI)") +
    ev_theme(cfg, "prognostic")
}

ev_figdata_prog_dca <- function(cfg, pe, set) {
  fk <- ev_prog_figure_keys(cfg); out <- list()
  for (h in pe$draws$horizons) {
    pn <- sprintf("%d months", h)
    d <- pe$dca[pe$dca$evaluation_set == set & pe$dca$horizon_months == h & pe$dca$model %in% fk, ]
    mod <- d[d$strategy == d$model, ]
    ref <- d[d$model == fk[1] & d$strategy %in% c("Treat All", "Treat None"), ]
    all <- rbind(mod, ref)
    out[[length(out) + 1]] <- ev_fig_data(pn, "net_benefit", all$strategy, all$threshold, all$net_benefit,
                                          n = all$n_eligible, label = vapply(all$strategy, function(s) ev_label(cfg, "prognostic", s), ""),
                                          status = all$status)
    out[[length(out) + 1]] <- ev_fig_data(pn, "status_note", "status", NA, NA, n = d$n_eligible[1],
                                          label = sprintf("N=%d | Events=%d | %s (original rule: N>=%d and Events>=%d)",
                                                          d$n_eligible[1], d$n_events_by_t[1], d$status[1],
                                                          cfg$evaluation$dca$prognostic_min_eligible,
                                                          cfg$evaluation$dca$prognostic_min_events),
                                          status = d$status[1])
  }
  do.call(rbind, out)
}

ev_figdata_km <- function(cfg, km, cutpoint) {
  cv <- km$curve; rt <- km$risk_table
  rbind(ev_fig_data("Kaplan-Meier", "km_curve", cv$risk_group, cv$time, cv$surv, cv$lower, cv$upper, cv$n_risk,
                    status = km$status),
        ev_fig_data("Number at risk", "risk_table", rt$risk_group, rt$time, NA, n = rt$n_at_risk),
        ev_fig_data("Kaplan-Meier", "annotation", "cutpoint", cutpoint, NA,
                    label = sprintf("N=%d | SEER training cutpoint applied unchanged: LP <= %.4f -> Low-risk", km$n, cutpoint)),
        ev_fig_data("Kaplan-Meier", "annotation", "logrank", NA, km$logrank_p,
                    label = if (km$logrank_status == "computed") {
                      if (km$logrank_p < 0.001) "p < 0.001" else paste0("p = ", round(km$logrank_p, 3))
                    } else km$logrank_status))
}

# KM panel as fig8 (10_kaplan_meier_risk_groups.R): GROUP_COLORS, 95% CI ribbons, log-rank p,
# legend inside, number-at-risk table every 25 months. The ribbon is drawn as a step band from
# the same KM points (a straight-line ribbon between 11 patients' event times would mis-draw the CI).
ev_plot_km <- function(fd, cfg, title) {
  cv <- fd[fd$layer == "km_curve", ]; rt <- fd[fd$layer == "risk_table", ]
  ann <- fd[fd$layer == "annotation", ]
  cols <- ev_chr(cfg$evaluation$km$group_colors); names(cols) <- names(cfg$evaluation$km$group_colors)
  cv$series <- factor(cv$series, levels = c("Low-risk", "High-risk"))
  x_plot_max <- max(rt$x); time_breaks <- sort(unique(rt$x))
  band <- do.call(rbind, lapply(split(cv, cv$series), function(d) {
    if (!nrow(d)) return(NULL)
    d <- d[order(d$x), ]; k <- nrow(d)
    data.frame(series = d$series[1], x = c(rbind(d$x, c(d$x[-1], d$x[k])))[seq_len(2 * k - 1)],
               ymin = rep(d$ymin, each = 2)[seq_len(2 * k - 1)], ymax = rep(d$ymax, each = 2)[seq_len(2 * k - 1)])
  }))
  p_main <- ggplot2::ggplot(cv, ggplot2::aes(x = x, y = y, color = series, fill = series)) +
    ggplot2::geom_ribbon(data = band, ggplot2::aes(x = x, ymin = ymin, ymax = ymax, fill = series),
                         inherit.aes = FALSE, alpha = 0.15, color = NA, na.rm = TRUE) +
    ggplot2::geom_step(linewidth = 0.9) +
    ggplot2::scale_color_manual(values = cols, name = NULL) +
    ggplot2::scale_fill_manual(values = cols, name = NULL) +
    ggplot2::scale_y_continuous(limits = c(0, 1), labels = scales::percent_format(1),
                                expand = ggplot2::expansion(mult = c(0, 0.02))) +
    ggplot2::scale_x_continuous(limits = c(0, x_plot_max), breaks = time_breaks,
                                expand = ggplot2::expansion(mult = c(0.01, 0.02))) +
    ggplot2::annotate("text", x = x_plot_max * 0.62, y = 0.92, label = ann$label[ann$series == "logrank"],
                      size = 3.5, color = "black") +
    ggplot2::labs(title = title, subtitle = ann$label[ann$series == "cutpoint"], x = NULL, y = "Survival probability") +
    ggplot2::theme_bw(base_size = 11) +
    ggplot2::theme(plot.title = ggplot2::element_text(face = "bold"),
                   plot.subtitle = ggplot2::element_text(size = 8.5, color = "grey40"),
                   legend.position = c(0.83, 0.85),
                   legend.background = ggplot2::element_rect(fill = "white", color = NA),
                   legend.text = ggplot2::element_text(size = 9),
                   axis.text.x = ggplot2::element_blank(), axis.ticks.x = ggplot2::element_blank())
  rt$series <- factor(rt$series, levels = c("High-risk", "Low-risk"))
  p_risk <- ggplot2::ggplot(rt, ggplot2::aes(x = x, y = series, label = n)) +
    ggplot2::geom_text(size = 3.2, color = "black") +
    ggplot2::scale_x_continuous(limits = c(0, x_plot_max), breaks = time_breaks,
                                expand = ggplot2::expansion(mult = c(0.01, 0.02))) +
    ggplot2::scale_y_discrete(drop = FALSE) +
    ggplot2::labs(x = "Time (months)", y = NULL, title = "Number at risk") +
    ggplot2::theme_bw(base_size = 10) +
    ggplot2::theme(plot.title = ggplot2::element_text(size = 9, face = "bold", hjust = 0),
                   panel.grid = ggplot2::element_blank(), axis.ticks.y = ggplot2::element_blank(),
                   axis.text.y = ggplot2::element_text(size = 9), axis.text.x = ggplot2::element_text(size = 9))
  p_main / p_risk + patchwork::plot_layout(heights = c(4, 1.5))
}

# ==================================================================================
# 15. Figure families (figure data written first; plots drawn only from it)
# ==================================================================================

EV_FIGURE_FAMILIES <- c("diagnostic_external_roc", "diagnostic_external_calibration", "diagnostic_external_dca",
                        "diagnostic_external_evaluation_3panel", "prognostic_external_time_roc_3panel",
                        "prognostic_external_calibration_3panel", "prognostic_external_dca_3panel",
                        "prognostic_external_km_risk_groups")

ev_tags <- function(cfg, task) {
  patchwork::plot_annotation(tag_levels = "A", theme = ggplot2::theme(
    plot.tag = ggplot2::element_text(face = "bold", size = cfg$figures$style[[task]]$tag_size)))
}

ev_make_figures <- function(cfg, paths, de, pe, cutpoint) {
  dk <- ev_diag_keys(cfg); fk <- ev_prog_figure_keys(cfg)
  dset <- "common_all_diagnostic_models"; pset <- "common_primary_comparison"
  coh <- cfg$figures$cohort_display_name
  nd <- de$metrics$n_evaluable[de$metrics$evaluation_set == dset][1]
  npos <- de$metrics$n_events_or_positive[de$metrics$evaluation_set == dset][1]
  dsub <- sprintf("%s | Common evaluation set: N=%d (%d with liver metastasis)", coh, nd, npos)
  fdir <- file.path(paths$out, "figure_data")
  made <- list()
  emit <- function(family, fd, plot) {
    ev_write_csv(fd, file.path(fdir, paste0(family, ".csv")))
    f <- ev_save_plot(plot, family, cfg, paths)
    made[[family]] <<- data.frame(family = family, pdf = ev_rel(paths, f["pdf"]), png = ev_rel(paths, f["png"]),
                                  figure_data = ev_rel(paths, file.path(fdir, paste0(family, ".csv"))),
                                  n_figure_data_rows = nrow(fd), stringsAsFactors = FALSE)
  }
  cal_sub <- function(m, sub) {
    s <- Filter(function(x) x$key == m, cfg$models$diagnostic)[[1]]
    pts <- sub[sub$layer == "group_point", ]
    sprintf("Model: %s (%d vars) | N=%d | groups n=%s", m, length(s$declared_predictors), nd,
            if (nrow(pts)) paste(pts$n, collapse = "/") else "NA")
  }
  # 1 diagnostic ROC (all four models)
  fd <- ev_figdata_diag_roc(cfg, de, dk, dset)
  emit("diagnostic_external_roc", fd, ev_plot_roc(fd, cfg, "diagnostic", "ROC", dsub))
  # 2 diagnostic calibration (one panel per model, as fig3 B)
  fd <- ev_figdata_diag_cal(cfg, de, dk, dset, panel_by_model = TRUE)
  panels <- lapply(dk, function(m) { sub <- fd[fd$panel == m, ]; ev_plot_diag_cal(sub, cfg, "Calibration", cal_sub(m, sub)) })
  emit("diagnostic_external_calibration", fd, patchwork::wrap_plots(panels, ncol = 2) + ev_tags(cfg, "diagnostic"))
  # 3 diagnostic DCA (all four models)
  fd <- ev_figdata_diag_dca(cfg, de, dk, dset)
  emit("diagnostic_external_dca", fd, ev_plot_dca(fd, cfg, "diagnostic", "Decision Curve Analysis", dsub))
  # 4 diagnostic 3-panel as fig3: primary nomogram vs AJCC T/N baseline
  pm <- c("nomogram_figure", "tnm_baseline")
  f1 <- ev_figdata_diag_roc(cfg, de, pm, dset, panel = "A ROC")
  f2 <- ev_figdata_diag_cal(cfg, de, "nomogram_figure", dset, panel_by_model = FALSE); f2$panel <- "B Calibration"
  f3 <- ev_figdata_diag_dca(cfg, de, pm, dset, panel = "C Decision Curve Analysis")
  fd <- rbind(f1, f2, f3)
  p <- ev_plot_roc(f1, cfg, "diagnostic", "ROC", dsub, tag = "A") +
    ev_plot_diag_cal(f2, cfg, "Calibration", cal_sub("nomogram_figure", f2), tag = "B") +
    ev_plot_dca(f3, cfg, "diagnostic", "Decision Curve Analysis", dsub, tag = "C") +
    patchwork::plot_layout(ncol = 3)
  emit("diagnostic_external_evaluation_3panel", fd, p)
  # 5 prognostic time ROC (as fig6, one row: 1/2/3-year)
  fd <- ev_figdata_prog_roc(cfg, pe, pset)
  panels <- lapply(pe$draws$horizons, function(h) {
    pn <- sprintf("%d months", h); sub <- fd[fd$panel == pn, ]
    ttl <- paste0(ev_hz_label(cfg, h), " ROC - ", coh)
    if (!any(sub$layer == "curve")) return(ev_not_estimable_panel(ttl, paste(unique(sub$label[sub$layer == "not_estimable"]), collapse = "; "), cfg, "prognostic"))
    ev_plot_roc(sub, cfg, "prognostic", ttl, sub$label[sub$layer == "counts"])
  })
  emit("prognostic_external_time_roc_3panel", fd, patchwork::wrap_plots(panels, ncol = 3) + ev_tags(cfg, "prognostic"))
  # 6 prognostic calibration (no reference figure exists; same panel style)
  fd <- ev_figdata_prog_cal(cfg, pe, pset)
  panels <- lapply(pe$draws$horizons, function(h) {
    sub <- fd[fd$panel == sprintf("%d months", h), ]
    expl <- if (any(sub$layer == "group_point_exploratory")) "
Open: 3 quantile groups (exploratory_small_sample)" else ""
    ev_plot_prog_cal_panel(sub, cfg, paste0(ev_hz_label(cfg, h), " calibration - ", coh),
                           sprintf("N=%d | Filled: overall KM vs mean predicted survival%s", np_of(sub), expl))
  })
  emit("prognostic_external_calibration_3panel", fd, patchwork::wrap_plots(panels, ncol = 3) + ev_tags(cfg, "prognostic"))
  # 7 prognostic DCA (as fig7)
  fd <- ev_figdata_prog_dca(cfg, pe, pset)
  panels <- lapply(pe$draws$horizons, function(h) {
    sub <- fd[fd$panel == sprintf("%d months", h), ]
    ev_plot_dca(sub, cfg, "prognostic", paste0(ev_hz_label(cfg, h), " DCA - ", coh), sub$label[sub$layer == "status_note"])
  })
  emit("prognostic_external_dca_3panel", fd, patchwork::wrap_plots(panels, ncol = 3) + ev_tags(cfg, "prognostic"))
  # 8 KM risk groups (as fig8)
  fd <- ev_figdata_km(cfg, pe$km, cutpoint)
  emit("prognostic_external_km_risk_groups", fd, ev_plot_km(fd, cfg, coh))
  do.call(rbind, made)
}

np_of <- function(sub) { x <- sub$n[sub$layer == "overall_point"]; if (length(x)) as.integer(x[1]) else NA_integer_ }

# ==================================================================================
# 16. Identity, inventory and handoff
# ==================================================================================

EV_CODE_FILES <- c("scripts/01_prepare_external_cohort.py", "scripts/02_run_external_validation.R",
                   "src/__init__.py", "src/schema_contract.py", "src/mapping.py", "src/privacy.py",
                   "src/validation.py", "src/external_validation_helpers.R")

ev_identity <- function(cfg, paths) {
  specs <- ev_model_specs(cfg, paths)
  model_files <- unique(unlist(lapply(specs, function(s) c(s$path, s$absolute_risk_object))))
  prov <- unique(c(ev_chr(cfg$models$provenance_files), cfg$evaluation$km$cutpoint_source))
  h <- function(rels, root) stats::setNames(lapply(rels, function(r) ev_sha256_file(file.path(root, r))), rels)
  list(run_id = cfg$run_id,
       config_sha256 = ev_sha256_file(paths$config),
       workbook_sha256 = ev_sha256_file(ev_repo(paths, cfg$source$workbook_path)),
       preparation_manifest_sha256 = ev_sha256_file(file.path(paths$module, cfg$outputs$preparation_manifest)),
       prepared_sha256 = h(c(cfg$outputs$prepared_diagnostic, cfg$outputs$prepared_prognostic), paths$module),
       frozen_model_sha256 = h(model_files, paths$repo),
       provenance_sha256 = h(prov, paths$repo),
       code_sha256 = h(EV_CODE_FILES, paths$module),
       r_version = R.version.string, packages = ev_pkg_versions())
}

ev_identity_diff <- function(a, b) {
  fa <- unlist(a); fb <- unlist(b)
  keys <- union(names(fa), names(fb))
  keys[vapply(keys, function(k) !identical(unname(fa[k]), unname(fb[k])), logical(1))]
}

EV_MUTABLE_OUTPUTS <- c("outputs/run_manifest.json", "outputs/EXECUTION_HANDOFF.md", "outputs/manifests/report_handoff.json",
                        "outputs/validation/completeness_validation.json", "outputs/tables/result_inventory.csv")

ev_inventory <- function(paths) {
  files <- list.files(paths$out, recursive = TRUE, full.names = TRUE, all.files = FALSE)
  files <- files[!grepl("/local_only/", files) & !grepl("\\.tmp-", files)]
  rel <- ev_strip_prefix(paths$module, files)
  mut <- rel %in% EV_MUTABLE_OUTPUTS
  data.frame(path = rel, kind = sub("/.*$", "", sub("^outputs/", "", rel)),
             bytes = file.size(files),
             sha256 = ifelse(mut, "mutable_status_file", vapply(files, ev_sha256_file, "")),
             patient_level = FALSE, stringsAsFactors = FALSE)
}

ev_required_tables <- function() {
  c("external_cohort_flow.csv", "external_baseline_characteristics.csv", "external_model_registry.csv",
    "external_model_predictor_coverage.csv", "diagnostic_external_metrics.csv", "diagnostic_external_roc_curve.csv",
    "diagnostic_external_calibration.csv", "diagnostic_external_dca.csv", "prognostic_external_discrimination.csv",
    "prognostic_external_time_auc.csv", "prognostic_external_roc_curve.csv", "prognostic_external_calibration.csv",
    "prognostic_external_dca.csv", "prognostic_external_km_curve.csv", "prognostic_external_km_risk_table.csv",
    "prognostic_external_km_summary.csv", "external_supportedness_ledger.csv", "internal_external_context.csv",
    "report_summary_diagnostic.csv", "report_summary_prognostic.csv", "result_inventory.csv")
}

ev_commands <- function() {
  c(run = "python analysis/04_external_validation/scripts/01_prepare_external_cohort.py --config analysis/04_external_validation/config/external_validation_config.json",
    run2 = "Rscript analysis/04_external_validation/scripts/02_run_external_validation.R --config analysis/04_external_validation/config/external_validation_config.json",
    validate1 = "python analysis/04_external_validation/scripts/01_prepare_external_cohort.py --config analysis/04_external_validation/config/external_validation_config.json --validate-only",
    validate2 = "Rscript analysis/04_external_validation/scripts/02_run_external_validation.R --config analysis/04_external_validation/config/external_validation_config.json --validate-only",
    rebuild = "Rscript analysis/04_external_validation/scripts/02_run_external_validation.R --config analysis/04_external_validation/config/external_validation_config.json --rebuild-aggregates",
    dry1 = "python analysis/04_external_validation/scripts/01_prepare_external_cohort.py --config analysis/04_external_validation/config/external_validation_config.json --dry-run")
}

# ==================================================================================
# 17. Independent, read-only validator (run in a fresh process via --validate-only)
# ==================================================================================

ev_check <- function(name, passed, detail = list()) list(check = name, passed = isTRUE(passed), detail = detail)

ev_close <- function(a, b, tol = 1e-10) {
  if (length(a) != length(b)) return(FALSE)
  both_na <- is.na(a) & is.na(b)
  all(both_na | (!is.na(a) & !is.na(b) & abs(a - b) <= tol))
}

ev_validate <- function(cfg, paths) {
  C <- list(); add <- function(x) C[[length(C) + 1]] <<- x
  T <- function(f) ev_read_csv(file.path(paths$out, "tables", f))
  rm_path <- file.path(paths$out, "run_manifest.json")
  if (!file.exists(rm_path)) return(list(ev_check("run_manifest_present", FALSE)))
  rmf <- jsonlite::fromJSON(rm_path, simplifyVector = FALSE)
  # 1 hashes
  cur <- ev_identity(cfg, paths)
  d <- ev_identity_diff(rmf$identity, cur)
  add(ev_check("01_source_model_config_code_hashes_match_run_manifest", !length(d), list(differences = d)))
  reg <- T("external_model_registry.csv")
  add(ev_check("01b_registry_hashes_equal_current_files",
               all(vapply(seq_len(nrow(reg)), function(i) {
                 p <- if (grepl("^[A-Za-z]:/|^/", reg$path[i])) reg$path[i] else file.path(paths$repo, reg$path[i])
                 file.exists(p) && ev_sha256_file(p) == reg$sha256[i] }, logical(1))),
               list(n_objects = nrow(reg), expected_hash_matches = sum(reg$sha256_matches_expected %in% TRUE),
                    expected_hash_available = sum(!is.na(reg$expected_sha256)))))
  # 2 prepared cohort
  diag <- ev_read_csv(file.path(paths$module, cfg$outputs$prepared_diagnostic))
  prog <- ev_read_csv(file.path(paths$module, cfg$outputs$prepared_prognostic))
  seer_d <- strsplit(readLines(ev_repo(paths, cfg$reference_preprocessing$seer_diagnostic_input_path), n = 1), ",")[[1]]
  seer_p <- strsplit(readLines(ev_repo(paths, cfg$reference_preprocessing$seer_prognostic_input_path), n = 1), ",")[[1]]
  prep_man <- jsonlite::fromJSON(file.path(paths$module, cfg$outputs$preparation_manifest), simplifyVector = FALSE)
  prep_hash_ok <- identical(prep_man$outputs_sha256[[cfg$outputs$prepared_diagnostic]], ev_sha256_file(file.path(paths$module, cfg$outputs$prepared_diagnostic))) &&
    identical(prep_man$outputs_sha256[[cfg$outputs$prepared_prognostic]], ev_sha256_file(file.path(paths$module, cfg$outputs$prepared_prognostic)))
  add(ev_check("02_prepared_cohort_counts_and_schema",
               nrow(diag) == cfg$source$expected_n_rows && nrow(prog) == cfg$source$expected_prognostic_n_rows &&
                 identical(names(diag), seer_d) && identical(names(prog), seer_p) && prep_hash_ok &&
                 identical(prep_man$status, "passed") && all(prog$mets_liver == 1) &&
                 identical(prog$patient_id, diag$patient_id[diag$mets_liver == 1]),
               list(n_diagnostic = nrow(diag), n_prognostic = nrow(prog), preparation_status = prep_man$status)))
  # 3 prediction rows / patient sets
  dp <- ev_read_csv(file.path(paths$local, "predictions", "diagnostic_predictions.csv"))
  pp <- ev_read_csv(file.path(paths$local, "predictions", "prognostic_predictions.csv"))
  dk <- ev_diag_keys(cfg); pk <- ev_prog_keys(cfg)
  sets_ok <- all(vapply(dk, function(k) setequal(dp$patient_id[dp$model == k], diag$patient_id), logical(1))) &&
    all(vapply(pk, function(k) setequal(pp$patient_id[pp$model == k], prog$patient_id), logical(1)))
  pred_hash_ok <- identical(rmf$local_outputs_sha256$diagnostic_predictions, ev_sha256_file(file.path(paths$local, "predictions", "diagnostic_predictions.csv"))) &&
    identical(rmf$local_outputs_sha256$prognostic_predictions, ev_sha256_file(file.path(paths$local, "predictions", "prognostic_predictions.csv")))
  add(ev_check("03_prediction_row_counts_and_patient_sets",
               nrow(dp) == nrow(diag) * length(dk) && nrow(pp) == nrow(prog) * length(pk) && sets_ok && pred_hash_ok,
               list(diagnostic_rows = nrow(dp), prognostic_rows = nrow(pp),
                    diagnostic_patient_set_sha256 = ev_sha256_text(sort(diag$patient_id)),
                    prognostic_patient_set_sha256 = ev_sha256_text(sort(prog$patient_id)))))
  # 4 recompute predictions from the saved frozen objects (no fitting)
  re <- ev_make_predictions(cfg, paths, diag, prog)
  key <- function(d) paste(d$model, d$patient_id)
  a <- dp[match(key(re$diagnostic), key(dp)), ]; b <- pp[match(key(re$prognostic), key(pp)), ]
  hzc <- grep("^surv_", names(pp), value = TRUE)
  dmax <- max(abs(c(a$predicted_probability - re$diagnostic$predicted_probability)), na.rm = TRUE)
  pmax_ <- max(abs(c(as.matrix(b[, c("linear_predictor", hzc)]) - as.matrix(re$prognostic[, c("linear_predictor", hzc)]))), na.rm = TRUE)
  na_same <- all(is.na(a$predicted_probability) == is.na(re$diagnostic$predicted_probability)) &&
    all(unname(is.na(as.matrix(b[, c("linear_predictor", hzc)]))) == unname(is.na(as.matrix(re$prognostic[, c("linear_predictor", hzc)]))))
  add(ev_check("04_direct_recomputation_of_predictions", na_same && dmax <= 1e-12 && pmax_ <= 1e-12,
               list(max_abs_diff_diagnostic = dmax, max_abs_diff_prognostic = pmax_)))
  # 5 point metrics from saved predictions with direct package calls
  dm <- T("diagnostic_external_metrics.csv"); pm <- T("prognostic_external_discrimination.csv")
  ta <- T("prognostic_external_time_auc.csv"); pc <- T("prognostic_external_calibration.csv")
  clip <- cfg$evaluation$probability_clip_for_logit
  diffs <- c()
  for (i in seq_len(nrow(dm))) {
    r <- dm[i, ]; d1 <- dp[dp$model == r$model & dp$status == "ok", ]
    if (r$evaluation_set != "model_specific") {
      okids <- Reduce(intersect, lapply(dk, function(k) dp$patient_id[dp$model == k & dp$status == "ok"])); d1 <- d1[d1$patient_id %in% okids, ]
    }
    y <- d1$mets_liver; p <- d1$predicted_probability
    v <- switch(r$metric, auc = as.numeric(pROC::auc(pROC::roc(y, p, direction = "<", quiet = TRUE, levels = c(0, 1)))),
                brier = mean((y - p)^2), ev_calib_glm(y, p, clip)[[r$metric]])
    diffs <- c(diffs, if (is.na(v) && is.na(r$estimate)) 0 else abs(v - r$estimate))
  }
  base <- pp[pp$model == pk[1], ]; base <- base[order(base$row_index), ]
  tm <- ifelse(base$os_months == 0, cfg$mapping$zero_survival_time_replacement, base$os_months); ev <- base$os_event
  rows_for <- function(model, set) {
    okm <- sapply(pk, function(k) { d <- pp[pp$model == k, ]; d$status[order(d$row_index)] == "ok" })
    if (set == "model_specific") which(okm[, model]) else if (set == "common_all_prognostic_models") which(apply(okm, 1, all)) else
      which(apply(okm[, ev_prog_figure_keys(cfg), drop = FALSE], 1, all))
  }
  for (i in seq_len(nrow(pm))) {
    r <- pm[i, ]; rr <- rows_for(r$model, r$evaluation_set)
    lp <- pp$linear_predictor[pp$model == r$model][order(pp$row_index[pp$model == r$model])][rr]
    v <- unname(survival::concordance(survival::Surv(tm[rr], ev[rr]) ~ lp, reverse = TRUE)$concordance)
    diffs <- c(diffs, abs(v - r$estimate))
  }
  for (i in seq_len(nrow(ta))) {
    r <- ta[i, ]; rr <- rows_for(r$model, r$evaluation_set)
    lp <- pp$linear_predictor[pp$model == r$model][order(pp$row_index[pp$model == r$model])][rr]
    v <- tryCatch(timeROC::timeROC(T = tm[rr], delta = ev[rr], marker = lp, cause = 1, weighting = "marginal",
                                   times = r$horizon_months, iid = FALSE)$AUC, error = function(e) NA)
    v <- v[length(v)]
    diffs <- c(diffs, if (is.na(v) && is.na(r$estimate)) 0 else abs(v - r$estimate))
  }
  bs <- pc[pc$metric == "brier_ipcw", ]
  for (i in seq_len(nrow(bs))) {
    r <- bs[i, ]; rr <- rows_for(r$model, r$evaluation_set)
    s <- pp[[sprintf("surv_%d", r$horizon_months)]][pp$model == r$model][order(pp$row_index[pp$model == r$model])][rr]
    v <- ev_brier_score_pkg(tm[rr], ev[rr], s, r$horizon_months)
    diffs <- c(diffs, if (is.na(v) && is.na(r$estimate)) 0 else abs(v - r$estimate))
  }
  add(ev_check("05_direct_recomputation_of_point_metrics", all(is.finite(diffs)) && max(diffs) <= 1e-10,
               list(n_metrics_recomputed = length(diffs), max_abs_diff = max(diffs))))
  # 6 bootstrap schedules + sampled draw checks + interval recomputation
  sd_saved <- readRDS(file.path(paths$local, "bootstrap", "diagnostic_schedule.rds"))
  sp_saved <- readRDS(file.path(paths$local, "bootstrap", "prognostic_schedule.rds"))
  sd_new <- ev_schedule_diagnostic(diag$mets_liver, cfg); sp_new <- ev_schedule_prognostic(nrow(prog), cfg)
  sch_ok <- identical(sd_saved, sd_new) && identical(sp_saved, sp_new) &&
    identical(ev_schedule_hash(sd_new), rmf$bootstrap$diagnostic_schedule_sha256) &&
    identical(ev_schedule_hash(sp_new), rmf$bootstrap$prognostic_schedule_sha256)
  draws <- readRDS(file.path(paths$local, "bootstrap", "draw_level_results.rds"))
  nb <- cfg$bootstrap$validator_sampled_draws
  bsel <- unique(round(seq(1, cfg$bootstrap$draws, length.out = nb)))
  ddraw <- draws$diagnostic; maxd <- 0
  base_d <- dp[dp$model == dk[1], ]; base_d <- base_d[order(base_d$row_index), ]; y <- base_d$mets_liver
  P <- ev_wide(dp, dk, "predicted_probability"); okP <- !is.na(P)
  units_d <- ev_units(okP, dk, list(common_all_diagnostic_models = dk))
  for (b in bsel) for (k in seq_along(units_d)) {
    u <- units_d[[k]]; idx <- sd_saved[b, ]; r <- idx[idx %in% u$rows]
    v <- c(ev_auc_fast(y[r], P[r, u$model]), ev_brier(y[r], P[r, u$model]), ev_calib_fast(y[r], P[r, u$model], clip))
    s <- ddraw$metrics[b, k, ]
    maxd <- max(maxd, ifelse(is.na(v) & is.na(s), 0, abs(v - s)), na.rm = FALSE)
  }
  pdraw <- draws$prognostic; LP <- ev_wide(pp, pk, "linear_predictor"); okL <- !is.na(LP)
  cu <- ev_units(okL, pk, list(common_all_prognostic_models = pk, common_primary_comparison = ev_prog_figure_keys(cfg)))
  hu <- Filter(function(u) u$model %in% ev_prog_figure_keys(cfg) && u$set %in% c("model_specific", "common_primary_comparison"), cu)
  hz <- pdraw$horizons
  SS <- lapply(hz, function(h) ev_wide(pp, ev_prog_figure_keys(cfg), sprintf("surv_%d", h)))
  for (b in bsel[seq_len(min(15, length(bsel)))]) {
    idx <- sp_saved[b, ]
    for (k in seq_along(cu)) { r <- idx[idx %in% cu[[k]]$rows]; v <- ev_cindex(tm[r], ev[r], LP[r, cu[[k]]$model])[["c_index"]]
      maxd <- max(maxd, if (is.na(v) && is.na(pdraw$c_index[b, k])) 0 else abs(v - pdraw$c_index[b, k])) }
    for (k in seq_along(hu)) { r <- idx[idx %in% hu[[k]]$rows]; m <- hu[[k]]$model
      for (j in seq_along(hz)) {
        v <- c(ev_time_auc(tm[r], ev[r], LP[r, m], hz[j])$auc, ev_brier_ipcw(tm[r], ev[r], SS[[j]][r, m], hz[j]))
        s <- c(pdraw$time_auc[b, k, j], pdraw$brier_ipcw[b, k, j])
        maxd <- max(maxd, ifelse(is.na(v) & is.na(s), 0, abs(v - s)))
      } }
  }
  ci_diff <- 0
  for (i in seq_len(nrow(dm))) {
    r <- dm[i, ]; k <- which(vapply(units_d, function(u) u$model == r$model && u$set == r$evaluation_set, logical(1)))
    x <- ddraw$metrics[, k, match(r$metric, ddraw$metric_names)]; x <- x[is.finite(x)]
    if (length(x) != r$bootstrap_draws_valid) ci_diff <- Inf
    if (!is.na(r$ci_lower_95)) ci_diff <- max(ci_diff, abs(ev_percentile_ci(x, cfg$bootstrap$confidence_level) - c(r$ci_lower_95, r$ci_upper_95)))
  }
  for (i in seq_len(nrow(pm))) {
    r <- pm[i, ]; k <- which(vapply(cu, function(u) u$model == r$model && u$set == r$evaluation_set, logical(1)))
    x <- pdraw$c_index[, k]; x <- x[is.finite(x)]
    if (length(x) != r$bootstrap_draws_valid) ci_diff <- Inf
    if (!is.na(r$ci_lower_95)) ci_diff <- max(ci_diff, abs(ev_percentile_ci(x, cfg$bootstrap$confidence_level) - c(r$ci_lower_95, r$ci_upper_95)))
  }
  add(ev_check("06_bootstrap_schedule_and_sampled_intervals", sch_ok && maxd <= 1e-10 && ci_diff <= 1e-10,
               list(schedules_regenerated_identical = sch_ok, sampled_draws = length(bsel), max_abs_draw_diff = maxd,
                    max_abs_interval_diff = ci_diff, failed_draws_redrawn = FALSE)))
  # 7 common-sample membership
  cm_ok <- TRUE
  for (tb in list(dm, pm, ta)) for (s in setdiff(unique(tb$evaluation_set), "model_specific")) {
    h <- unique(tb$evaluation_rows_sha256[tb$evaluation_set == s]); cm_ok <- cm_ok && length(h) == 1
  }
  add(ev_check("07_common_sample_membership_consistent", cm_ok))
  # 8 calibration / DCA curve data consistency, figure data vs tables
  dca_t <- T("diagnostic_external_dca.csv"); dd <- dca_t[dca_t$evaluation_set == "common_all_diagnostic_models", ]
  thr <- seq(cfg$evaluation$dca$thresholds$from, cfg$evaluation$dca$thresholds$to, by = cfg$evaluation$dca$thresholds$by)
  rows_c <- which(apply(okP, 1, all)); dmx <- 0
  for (m in dk) {
    nbv <- ev_net_benefit(y[rows_c], P[rows_c, m], thr)
    dmx <- max(dmx, abs(nbv - dd$net_benefit[dd$model == m & dd$strategy == m]))
  }
  fdd <- ev_read_csv(file.path(paths$out, "figure_data", "diagnostic_external_dca.csv"))
  fig_ok <- TRUE
  for (m in dk) fig_ok <- fig_ok && ev_close(fdd$y[fdd$series == m], dd$net_benefit[dd$model == m & dd$strategy == m])
  rocd <- T("diagnostic_external_roc_curve.csv"); fr <- ev_read_csv(file.path(paths$out, "figure_data", "diagnostic_external_roc.csv"))
  for (m in dk) fig_ok <- fig_ok && ev_close(fr$y[fr$series == m & fr$layer == "curve"],
                                             rocd$true_positive_rate[rocd$model == m & rocd$evaluation_set == "common_all_diagnostic_models"])
  calt <- T("diagnostic_external_calibration.csv"); fc <- ev_read_csv(file.path(paths$out, "figure_data", "diagnostic_external_calibration.csv"))
  for (m in dk) fig_ok <- fig_ok && ev_close(fc$y[fc$series == m & fc$layer == "group_point"],
                                             calt$observed_proportion[calt$model == m & calt$evaluation_set == "common_all_diagnostic_models" & calt$status == "supported"])
  cal_ok <- TRUE
  for (m in dk) {
    c1 <- calt[calt$model == m & calt$evaluation_set == "common_all_diagnostic_models", ]
    if (c1$status[1] == "supported") {
      g <- ev_quantile_groups(P[rows_c, m], c1$n_groups[1], cfg$evaluation$diagnostic_calibration_groups$min_group_n)
      cal_ok <- cal_ok && !is.null(g) && identical(as.integer(tabulate(g, c1$n_groups[1])), as.integer(c1$n)) &&
        ev_close(vapply(seq_len(c1$n_groups[1]), function(i) mean(y[rows_c][g == i]), 1), c1$observed_proportion)
    }
  }
  pdt <- T("prognostic_external_dca.csv"); pfd <- ev_read_csv(file.path(paths$out, "figure_data", "prognostic_external_dca_3panel.csv"))
  for (h in hz) for (m in ev_prog_figure_keys(cfg)) {
    rr <- rows_for(m, "common_primary_comparison"); el <- ev_dca_eligibility(tm[rr], ev[rr], h)
    s <- pp[[sprintf("surv_%d", h)]][pp$model == m][order(pp$row_index[pp$model == m])][rr]
    nbv <- ev_net_benefit(el$y[el$eligible], 1 - s[el$eligible], thr)
    tab <- pdt$net_benefit[pdt$model == m & pdt$strategy == m & pdt$evaluation_set == "common_primary_comparison" & pdt$horizon_months == h]
    dmx <- max(dmx, abs(nbv - tab))
    fig_ok <- fig_ok && ev_close(pfd$y[pfd$panel == sprintf("%d months", h) & pfd$series == m & pfd$layer == "net_benefit"], tab)
  }
  prt <- T("prognostic_external_roc_curve.csv"); pfr <- ev_read_csv(file.path(paths$out, "figure_data", "prognostic_external_time_roc_3panel.csv"))
  for (h in hz) for (m in ev_prog_figure_keys(cfg)) fig_ok <- fig_ok && ev_close(
    pfr$y[pfr$panel == sprintf("%d months", h) & pfr$series == m & pfr$layer == "curve"],
    prt$true_positive_rate[prt$model == m & prt$horizon_months == h & prt$evaluation_set == "common_primary_comparison"])
  pcf <- ev_read_csv(file.path(paths$out, "figure_data", "prognostic_external_calibration_3panel.csv"))
  pco <- pc[pc$metric == "km_observed_survival" & pc$evaluation_set == "common_primary_comparison", ]
  for (i in seq_len(nrow(pco))) fig_ok <- fig_ok && ev_close(
    pcf$y[pcf$panel == sprintf("%d months", pco$horizon_months[i]) & pcf$series == pco$model[i] & pcf$layer == "overall_point"], pco$estimate[i])
  kmc <- T("prognostic_external_km_curve.csv"); kfd <- ev_read_csv(file.path(paths$out, "figure_data", "prognostic_external_km_risk_groups.csv"))
  fig_ok <- fig_ok && ev_close(kfd$y[kfd$layer == "km_curve"], kmc$surv)
  add(ev_check("08_calibration_dca_and_figure_data_consistency", dmx <= 1e-12 && cal_ok && fig_ok,
               list(max_abs_net_benefit_diff = dmx, grouped_calibration_recomputed = cal_ok, figure_data_equal_tables = fig_ok)))
  # 9 cutpoint identity
  cp_file <- ev_repo(paths, cfg$evaluation$km$cutpoint_source)
  cp <- utils::read.csv(cp_file)$cutpoint
  kms <- T("prognostic_external_km_summary.csv")
  lpn <- pp$linear_predictor[pp$model == cfg$evaluation$km$lp_model & pp$status == "ok"]
  grp_n <- c(`Low-risk` = sum(lpn <= cp), `High-risk` = sum(lpn > cp))
  add(ev_check("09_frozen_training_cutpoint_identity",
               identical(ev_sha256_file(cp_file), cur$provenance_sha256[[cfg$evaluation$km$cutpoint_source]]) &&
                 all(kms$cutpoint_applied == cp) && all(grp_n[kms$risk_group] == kms$n_patients),
               list(cutpoint = cp, group_counts = as.list(grp_n), cutpoint_estimated_from_external_data = FALSE)))
  # 10 figure coverage
  fams <- EV_FIGURE_FAMILIES
  cov <- vapply(fams, function(f) all(file.exists(file.path(paths$out, c(file.path("figures", paste0(f, c(".pdf", ".png"))),
                                                                            file.path("figure_data", paste0(f, ".csv")))))) &&
                  nrow(ev_read_csv(file.path(paths$out, "figure_data", paste0(f, ".csv")))) > 0, logical(1))
  add(ev_check("10_figure_pdf_png_and_figure_data_coverage", all(cov), list(missing = fams[!cov])))
  # 11 ledger completeness
  led <- T("external_supportedness_ledger.csv"); exp_k <- ev_expected_ledger(cfg)
  add(ev_check("11_supportedness_ledger_complete", all(exp_k %in% led$ledger_key) && all(led$support_status %in% EV_STATUS),
               list(expected = length(exp_k), present = sum(exp_k %in% led$ledger_key), missing = setdiff(exp_k, led$ledger_key),
                    unknown_status = unique(setdiff(led$support_status, EV_STATUS)))))
  # 12 identifiers
  cmap <- ev_read_csv(ev_repo(paths, file.path(cfg$privacy$local_only_root, cfg$privacy$code_map_relpath)))
  tokens <- c(cmap$code, cmap$patient_id)
  pub <- list.files(paths$module, recursive = TRUE, full.names = TRUE)
  pub <- pub[!grepl("/outputs/local_only/", pub) & !grepl("__pycache__", pub)]
  hits <- pub[vapply(pub, function(f) { b <- readBin(f, "raw", file.size(f)); any(vapply(tokens, function(t) length(grepRaw(t, b, fixed = TRUE)) > 0, logical(1))) }, logical(1))]
  csvs <- pub[grepl("\\.csv$", pub)]
  idcol <- csvs[vapply(csvs, function(f) "patient_id" %in% strsplit(readLines(f, n = 1, warn = FALSE), ",")[[1]], logical(1))]
  add(ev_check("12_no_identifiers_in_aggregate_outputs", !length(hits) && !length(idcol),
               list(files_scanned = length(pub), files_with_codes_or_pseudonyms = ev_rel(paths, hits),
                    public_csv_with_patient_id_column = ev_rel(paths, idcol))))
  # 13 pre-existing files byte identity (+ reports/, git-ignore)
  snap <- readLines(file.path(paths$module, cfg$outputs$pre_task_snapshot))
  sh <- sub(" .*$", "", snap); sp <- sub("^[0-9a-f]+ \\*?\\./", "", snap)
  skip <- grepl("(^|/)(__pycache__|\\.pytest_cache)/", sp)
  changed <- character(0); missing <- character(0)
  for (i in which(!skip)) {
    f <- file.path(paths$repo, sp[i])
    if (!file.exists(f)) { missing <- c(missing, sp[i]); next }
    if (sp[i] == ".gitignore") {
      b <- readBin(f, "raw", file.size(f)); txt <- rawToChar(b)
      beg <- regexpr(cfg$outputs$gitignore_block_begin, txt, fixed = TRUE)[1]
      end_tag <- cfg$outputs$gitignore_block_end
      en <- regexpr(end_tag, txt, fixed = TRUE)[1]
      stripped <- if (beg > 0 && en > 0) paste0(substr(txt, 1, beg - 2), substr(txt, en + nchar(end_tag) + 1, nchar(txt))) else txt
      if (digest::digest(stripped, algo = "sha256", serialize = FALSE) != sh[i]) changed <- c(changed, sp[i])
      next
    }
    if (ev_sha256_file(f) != sh[i]) changed <- c(changed, sp[i])
  }
  auth <- ev_chr(cfg$outputs$authorized_preexisting_changes$path_prefixes)
  is_auth <- function(x) if (!length(x)) logical(0) else vapply(x, function(f) any(startsWith(f, auth)), logical(1))
  authorized_changed <- changed[is_auth(changed)]; changed <- changed[!is_auth(changed)]
  authorized_new <- character(0)
  for (a in auth) {
    now <- paste0(a, list.files(file.path(paths$repo, a), recursive = TRUE, all.files = TRUE))
    authorized_new <- c(authorized_new, setdiff(now, sp))
  }
  rep_now <- list.files(file.path(paths$repo, "reports"), recursive = TRUE, all.files = TRUE)
  rep_new <- setdiff(paste0("reports/", rep_now), sp)
  ig <- function(p) suppressWarnings(system2("git", c("-C", shQuote(paths$repo, type = "cmd"), "check-ignore", "-q", shQuote(p, type = "cmd")),
                                              stdout = FALSE, stderr = FALSE)) == 0
  lo <- list.files(paths$local, recursive = TRUE, full.names = TRUE)
  lo_ok <- all(vapply(lo, function(f) ig(ev_rel(paths, f)), logical(1)))
  pub_out <- list.files(paths$out, recursive = TRUE, full.names = TRUE); pub_out <- pub_out[!grepl("/local_only/", pub_out)]
  pub_ok <- !any(vapply(pub_out, function(f) ig(ev_rel(paths, f)), logical(1)))
  add(ev_check("13_preexisting_files_byte_identical_and_git_ignore",
               !length(changed) && !length(missing) && !length(rep_new) && lo_ok && pub_ok,
               list(n_preexisting_files_checked = sum(!skip), n_cache_files_excluded = sum(skip), changed = changed,
                    authorized_reason = cfg$outputs$authorized_preexisting_changes$reason,
                    authorized_changed = authorized_changed, authorized_new_files = authorized_new,
                    missing = missing, new_files_under_reports = rep_new, gitignore_permitted_block_only = !(".gitignore" %in% changed),
                    local_only_files_git_ignored = lo_ok, n_local_only_files = length(lo), public_outputs_trackable = pub_ok)))
  # completion: required files and handoff paths
  req <- c(file.path("tables", ev_required_tables()), "manifests/report_handoff.json", "EXECUTION_HANDOFF.md", "run_manifest.json")
  miss_req <- req[!file.exists(file.path(paths$out, req))]
  ho_path <- file.path(paths$out, "manifests", "report_handoff.json")
  ho_ok <- FALSE; bad_paths <- character(0)
  if (file.exists(ho_path)) {
    ho <- jsonlite::fromJSON(ho_path, simplifyVector = FALSE)
    ps <- unique(c(unlist(lapply(ho$figures, function(f) c(f$pdf, f$png, f$figure_data))), unlist(ho$tables)))
    bad_paths <- ps[!file.exists(file.path(paths$repo, ps))]
    ho_ok <- !length(bad_paths)
  }
  add(ev_check("14_required_outputs_exist_and_handoff_paths_resolve", !length(miss_req) && ho_ok,
               list(missing_required = miss_req, unresolved_handoff_paths = bad_paths)))
  wb_ok <- identical(ev_sha256_file(ev_repo(paths, cfg$source$workbook_path)), cfg$source$expected_workbook_sha256)
  add(ev_check("15_source_workbook_byte_identical", wb_ok))
  C
}
