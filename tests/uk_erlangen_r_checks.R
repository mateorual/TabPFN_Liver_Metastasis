# tests/uk_erlangen_r_checks.R - component 13 R checks, run by tests/test_uk_erlangen_evaluation.py.
# Usage: Rscript tests/uk_erlangen_r_checks.R <config.json> <result.json>
# Read-only with respect to component outputs; temporary files go to local_only/tmp_tests and are removed.

argv <- commandArgs(trailingOnly = TRUE)
cfg_file <- argv[1]; out_file <- argv[2]
module <- dirname(dirname(normalizePath(cfg_file, winslash = "/")))
source(file.path(module, "src", "external_validation_helpers.R"))
ev_require()
paths <- ev_paths(cfg_file); cfg <- ev_load_cfg(paths)
R <- list()
run <- function(name, expr) {
  res <- tryCatch(expr, error = function(e) list(passed = FALSE, detail = conditionMessage(e)))
  if (is.logical(res)) res <- list(passed = isTRUE(res), detail = "")
  R[[name]] <<- res
}
tab <- function(f) ev_read_csv(file.path(paths$out, "tables", f))
have_outputs <- file.exists(file.path(paths$out, "run_manifest.json"))
prep <- list(diag = ev_read_csv(file.path(paths$module, cfg$outputs$prepared_diagnostic)),
             prog = ev_read_csv(file.path(paths$module, cfg$outputs$prepared_prognostic)))

run("rds_models_load_and_hashes_recorded", {
  reg <- tab("external_model_registry.csv")
  ok <- all(vapply(seq_len(nrow(reg)), function(i) {
    p <- file.path(paths$repo, reg$path[i]); m <- readRDS(p)
    ev_sha256_file(p) == reg$sha256[i] && ev_coef_hash(m) == reg$coefficient_sha256[i]
  }, logical(1)))
  list(passed = ok && nrow(reg) == 11, detail = sprintf("%d objects", nrow(reg)))
})

run("factor_levels_come_from_model_objects", {
  cox <- readRDS(ev_repo(paths, cfg$models$prognostic[[1]]$path))
  glm_m <- readRDS(ev_repo(paths, cfg$models$diagnostic[[1]]$path))
  one <- prep$prog[1, ]; one$grade <- 2L
  nb <- ev_build_newdata(cox, one, "prognostic", cfg)
  nd <- ev_build_newdata(glm_m, prep$diag[1:2, ], "diagnostic", cfg)
  identical(levels(nb$newdata$grade), cox$xlevels$grade) && identical(levels(nd$newdata$grade), glm_m$xlevels$grade) &&
    identical(levels(nd$newdata$grade), c("1", "2", "3", "4")) && identical(cox$xlevels$grade, c("I", "II", "III", "IV"))
})

run("novel_level_is_recorded_incompatibility_not_coerced", {
  rf <- readRDS(ev_repo(paths, cfg$models$prognostic[[5]]$path))
  x <- prep$prog; x$race[1] <- "Not a SEER category"
  nb <- ev_build_newdata(rf, x, "prognostic", cfg)
  list(passed = nb$status[1] == "incompatible_level" && is.na(nb$newdata$race[1]) &&
         identical(nb$incompatibilities$race$levels_not_in_model, "Not a SEER category"),
       detail = nb$status[1])
})

run("fresh_process_predictions_identical", {
  tmp <- file.path(paths$local, "tmp_tests"); dir.create(tmp, recursive = TRUE, showWarnings = FALSE)
  rs <- file.path(R.home("bin"), "Rscript"); q <- function(x) shQuote(x, type = "cmd")
  st <- system2(rs, c(q(paths$script), "--config", q(paths$config), "--internal-fresh-check", "predictions",
                      "--out", q(file.path(tmp, "p"))), stdout = FALSE, stderr = FALSE)
  a <- ev_read_csv(file.path(tmp, "p_diagnostic.csv")); b <- ev_read_csv(file.path(paths$local, "predictions", "diagnostic_predictions.csv"))
  c1 <- ev_read_csv(file.path(tmp, "p_prognostic.csv")); c2 <- ev_read_csv(file.path(paths$local, "predictions", "prognostic_predictions.csv"))
  unlink(tmp, recursive = TRUE)   # patient-level copies: removed explicitly (on.exit does not fire inside run())
  list(passed = st == 0 && identical(a, b) && identical(c1, c2) && !dir.exists(tmp),
       detail = "diagnostic probabilities and prognostic LP/S(12,24,36) recomputed in a fresh Rscript process")
})

run("no_external_row_used_in_any_fit", {
  reg <- tab("external_model_registry.csv")
  tnm <- readRDS(ev_repo(paths, cfg$models$diagnostic[[4]]$path))
  seer <- readRDS(ev_repo(paths, "analysis/03_reference_models/diagnostic/outputs/models/diagnostic_data_clean.rds"))
  ids <- c(prep$diag$patient_id)
  ok <- all(reg$training_n %in% c(2672, 323)) && length(tnm$y) == nrow(seer) &&
    !any(ids %in% as.character(unlist(tnm$data))) && all(grepl("^PNET_SEER_", tnm$data$patient_id))
  list(passed = ok, detail = "all frozen models trained on SEER only; T/N comparator data = SEER derivation rows")
})

run("tnm_baseline_is_reference_saved_object", {
  spec <- cfg$models$diagnostic[[4]]
  reg <- tab("external_model_registry.csv"); r <- reg[reg$task == "diagnostic" & reg$model == "tnm_baseline", ]
  saved <- readRDS(ev_repo(paths, spec$path))
  seer <- readRDS(ev_repo(paths, "analysis/03_reference_models/diagnostic/outputs/models/diagnostic_data_clean.rds"))
  refit <- stats::glm(mets_liver ~ ajcc_t + n_stage_positive, data = seer, family = stats::binomial(), na.action = stats::na.omit)
  rm <- jsonlite::fromJSON(file.path(paths$out, "run_manifest.json"))
  ok <- startsWith(spec$path, "analysis/03_reference_models/diagnostic/outputs/models/") && r$origin == "frozen_saved_rds" &&
    r$sha256 == ev_sha256_file(ev_repo(paths, spec$path)) && isTRUE(all.equal(coef(saved), coef(refit), tolerance = 1e-12)) &&
    length(saved$y) == nrow(seer) && isTRUE(rm$diagnostic_tnm_comparator$seer_internal_auc_check$passed)
  list(passed = ok, detail = "saved by 01_RefPaper_DiagnosticModel/scripts/04; equals the original formula on SEER data")
})

run("frozen_cutpoint_applied_unchanged", {
  cp <- utils::read.csv(ev_repo(paths, cfg$evaluation$km$cutpoint_source))$cutpoint
  ks <- tab("prognostic_external_km_summary.csv")
  pp <- ev_read_csv(file.path(paths$local, "predictions", "prognostic_predictions.csv"))
  lp <- pp$linear_predictor[pp$model == "nomogram_figure" & pp$status == "ok"]
  all(ks$cutpoint_applied == cp) && ks$n_patients[ks$risk_group == "Low-risk"] == sum(lp <= cp) &&
    ks$n_patients[ks$risk_group == "High-risk"] == sum(lp > cp) && all(!ks$cutpoint_estimated_from_external_data)
})

run("bootstrap_schedule_deterministic_and_shared", {
  s1 <- ev_schedule_prognostic(nrow(prep$prog), cfg); s2 <- ev_schedule_prognostic(nrow(prep$prog), cfg)
  d1 <- ev_schedule_diagnostic(prep$diag$mets_liver, cfg)
  rm <- jsonlite::fromJSON(file.path(paths$out, "run_manifest.json"))
  strat <- all(apply(d1, 1, function(r) sum(prep$diag$mets_liver[r]) == sum(prep$diag$mets_liver)))
  draws <- readRDS(file.path(paths$local, "bootstrap", "draw_level_results.rds"))
  shared <- dim(draws$diagnostic$metrics)[1] == nrow(d1) && dim(draws$prognostic$time_auc)[1] == nrow(s1)
  identical(s1, s2) && ev_schedule_hash(s1) == rm$bootstrap$prognostic_schedule_sha256 &&
    ev_schedule_hash(d1) == rm$bootstrap$diagnostic_schedule_sha256 && strat && shared
})

run("failed_bootstrap_draws_counted_not_replaced", {
  draws <- readRDS(file.path(paths$local, "bootstrap", "draw_level_results.rds"))
  ta <- tab("prognostic_external_time_auc.csv")
  k <- 1; j <- 1
  x <- draws$prognostic$time_auc[, k, j]
  row <- ta[ta$model == draws$prognostic$h_units[[k]]$model & ta$evaluation_set == draws$prognostic$h_units[[k]]$set &
              ta$horizon_months == draws$prognostic$horizons[j], ]
  fix <- ev_metric_row(cfg, "prognostic", "m", "s", 10, 10, 3, 7, "time_auc", 0.7, c(rep(NA, 4000), runif(1000)), "pb")
  list(passed = length(x) == cfg$bootstrap$draws && row$bootstrap_draws_valid == sum(is.finite(x)) &&
         row$bootstrap_draws_planned == cfg$bootstrap$draws && fix$bootstrap_draws_valid == 1000 &&
         fix$support_status == "fragile_interval",
       detail = sprintf("valid %d of %d planned for the first horizon unit", sum(is.finite(x)), length(x)))
})

run("estimators_agree_with_independent_packages_on_fixtures", {
  set.seed(11); n <- 40; y <- rbinom(n, 1, 0.4); p <- runif(n)
  a1 <- abs(ev_auc_fast(y, p) - as.numeric(pROC::auc(pROC::roc(y, p, direction = "<", quiet = TRUE))))
  a2 <- abs(ev_brier(y, p) - mean((y - p)^2))
  cg <- ev_calib_fast(y, p, 1e-6) - ev_calib_glm(y, p, 1e-6)
  tt <- sample(1:80, 30); ee <- rbinom(30, 1, 0.6); lp <- rnorm(30)
  c1 <- ev_cindex(tt, ee, lp)[["c_index"]]
  c2 <- unname(Hmisc::rcorr.cens(-lp, survival::Surv(tt, ee))["C Index"])
  t2 <- sample(c(6, 12, 18, 24, 30, 36, 48), 30, TRUE); S <- runif(30, 0.2, 0.95)
  b1 <- ev_brier_ipcw(t2, ee, S, 24); b2 <- ev_brier_score_pkg(t2, ee, S, 24)
  # pec evaluates at c(0, times), so the survival matrix carries S(0) = 1 as its first column
  pe <- pec::pec(object = list(m = cbind(1, S)), formula = Surv(time, status) ~ 1,
                 data = data.frame(time = t2, status = ee), times = 24, exact = FALSE, cens.model = "marginal",
                 splitMethod = "none", verbose = FALSE)
  b3 <- pe$AppErr$m[pe$time == 24]
  ta <- ev_time_auc(t2, ee, lp, 24)$auc
  ta2 <- timeROC::timeROC(T = t2, delta = ee, marker = lp, cause = 1, weighting = "marginal", times = 24, iid = FALSE)$AUC
  d <- c(a1, a2, max(abs(cg)), abs(c1 - c2), abs(b1 - b2), abs(b1 - b3), abs(ta - ta2[length(ta2)]))
  list(passed = all(d < 1e-10), detail = paste(signif(d, 3), collapse = ","))
})

run("saved_point_metrics_match_direct_package_calls", {
  rm <- jsonlite::fromJSON(file.path(paths$out, "validation", "completeness_validation.json"), simplifyVector = FALSE)
  ck <- Filter(function(x) x$check == "05_direct_recomputation_of_point_metrics", rm$checks)[[1]]
  ag <- ev_read_csv(file.path(paths$out, "validation", "estimator_agreement_diagnostic.csv"))
  bg <- ev_read_csv(file.path(paths$out, "validation", "estimator_agreement_brier_ipcw.csv"))
  list(passed = isTRUE(ck$passed) && max(ag$max_abs_diff_fast_vs_package) < 1e-10 && max(bg$abs_diff, na.rm = TRUE) < 1e-10,
       detail = sprintf("validator max diff %s", ck$detail$max_abs_diff))
})

run("dca_net_benefit_formula_spot_check", {
  y <- c(1, 1, 0, 0, 0, 1, 0, 0, 0, 0); p <- c(.9, .4, .35, .1, .5, .05, .2, .25, .3, .6)
  nb <- ev_net_benefit(y, p, c(0.2, 0.5))
  # t = 0.2: TP = 2 (p .9, .4), FP = 6 (p .35, .5, .2, .25, .3, .6); t = 0.5: TP = 1, FP = 2
  man <- c(2 / 10 - 6 / 10 * 0.2 / 0.8, 1 / 10 - 2 / 10 * 0.5 / 0.5)
  ta <- ev_treat_all(y, 0.2)
  isTRUE(all.equal(nb, man, tolerance = 1e-14)) && isTRUE(all.equal(ta, 0.3 - 0.7 * 0.25, tolerance = 1e-14))
})

run("survival_dca_eligibility_excludes_censoring_before_horizon", {
  time <- c(5, 10, 12, 12, 20, 30); event <- c(1, 0, 0, 1, 1, 0)
  el <- ev_dca_eligibility(time, event, 12)
  identical(el$eligible, c(TRUE, FALSE, TRUE, TRUE, TRUE, TRUE)) && identical(el$y[el$eligible], c(1L, 0L, 1L, 0L, 0L))
})

run("figure_data_exactly_match_plotted_data", {
  fd <- ev_read_csv(file.path(paths$out, "figure_data", "diagnostic_external_roc.csv"))
  p <- ev_plot_roc(fd, cfg, "diagnostic", "t", "s")
  b <- ggplot2::ggplot_build(p)$data[[1]]   # layer 1 = ROC path (reference-style layer order)
  cur <- fd[fd$layer == "curve", ]
  fk <- read.csv(file.path(paths$out, "figure_data", "prognostic_external_km_risk_groups.csv"))
  pk <- ev_plot_km(fk, cfg, "t")
  bk <- ggplot2::ggplot_build(pk[[1]])$data[[2]]
  ok_roc <- nrow(b) == nrow(cur) && isTRUE(all.equal(sort(b$y), sort(cur$y))) && isTRUE(all.equal(sort(b$x), sort(cur$x)))
  ok_km <- isTRUE(all.equal(sort(bk$y), sort(fk$y[fk$layer == "km_curve"])))
  list(passed = ok_roc && ok_km, detail = "ROC path layer and KM step layer equal their figure-data rows")
})

run("every_expected_result_has_ledger_row", {
  led <- tab("external_supportedness_ledger.csv"); k <- ev_expected_ledger(cfg)
  list(passed = all(k %in% led$ledger_key) && all(led$support_status %in% EV_STATUS),
       detail = sprintf("%d expected, %d ledger rows", length(k), nrow(led)))
})

jsonlite::write_json(R, out_file, auto_unbox = TRUE, pretty = TRUE, digits = NA)
