# scripts/02_run_external_validation.R
# Script 2 - apply the FROZEN reference-paper reproduction models to the prepared
# UK-Erlangen cohort and write validated external-validation outputs.
#
#   Rscript analysis/04_external_validation/scripts/02_run_external_validation.R \
#     --config analysis/04_external_validation/config/external_validation_config.json
#   ... --validate-only        fresh-process, read-only validator (writes completeness_validation.json only)
#   ... --rebuild-aggregates   recompute aggregates/figures from the saved local predictions
#
# Exit codes: 0 ok, 2 blocker, 3 resume refused (identity changed), 4 validation failed.
# Internal (used only by this script to obtain a fresh process):
#   --internal-fresh-check predictions --out <prefix>

ev_script_path <- function() {
  a <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", a[grepl("^--file=", a)])
  normalizePath(f[1], winslash = "/", mustWork = TRUE)
}

ev_parse_args <- function(argv) {
  out <- list(config = NULL, validate_only = FALSE, rebuild = FALSE, internal = NULL, out = NULL)
  i <- 1
  while (i <= length(argv)) {
    a <- argv[i]
    if (a == "--config") { out$config <- argv[i + 1]; i <- i + 2 }
    else if (a == "--validate-only") { out$validate_only <- TRUE; i <- i + 1 }
    else if (a == "--rebuild-aggregates") { out$rebuild <- TRUE; i <- i + 1 }
    else if (a == "--internal-fresh-check") { out$internal <- argv[i + 1]; i <- i + 2 }
    else if (a == "--out") { out$out <- argv[i + 1]; i <- i + 2 }
    else stop("unknown argument: ", a)
  }
  if (is.null(out$config)) stop("--config is required")
  if (out$validate_only && out$rebuild) stop("--validate-only and --rebuild-aggregates are mutually exclusive")
  out
}

ev_say <- function(...) cat(sprintf("[02 %s] ", format(Sys.time(), "%H:%M:%S")), ..., "\n", sep = "")

ev_fresh <- function(paths, args) {
  rs <- file.path(R.home("bin"), "Rscript")
  q <- function(x) shQuote(x, type = "cmd")
  out <- suppressWarnings(system2(rs, c(q(paths$script), "--config", q(paths$config), args), stdout = TRUE, stderr = TRUE))
  st <- attr(out, "status"); if (is.null(st)) st <- 0L
  list(status = st, output = out)
}

ev_load_prepared <- function(cfg, paths) {
  list(diag = ev_read_csv(file.path(paths$module, cfg$outputs$prepared_diagnostic)),
       prog = ev_read_csv(file.path(paths$module, cfg$outputs$prepared_prognostic)))
}

# ---- internal fresh-process modes ----------------------------------------------------
ev_internal <- function(args, cfg, paths) {
  if (args$internal == "predictions") {
    p <- ev_load_prepared(cfg, paths)
    pr <- ev_make_predictions(cfg, paths, p$diag, p$prog)
    ev_write_csv(pr$diagnostic, paste0(args$out, "_diagnostic.csv"))
    ev_write_csv(pr$prognostic, paste0(args$out, "_prognostic.csv"))
  } else stop("unknown internal check")
  invisible(0)
}

# ---- validator ------------------------------------------------------------------------
ev_run_validator <- function(cfg, paths) {
  checks <- ev_validate(cfg, paths)
  res <- list(run_id = cfg$run_id, generated = ev_now(),
              validator = "fresh Rscript process, --validate-only; read-only (no fitting, no output regeneration)",
              n_checks = length(checks), n_passed = sum(vapply(checks, function(x) x$passed, logical(1))),
              all_passed = all(vapply(checks, function(x) x$passed, logical(1))), checks = checks)
  ev_write_json(res, file.path(paths$out, "validation", "completeness_validation.json"))
  for (x in checks) ev_say(if (x$passed) "PASS " else "FAIL ", x$check)
  res
}

# ---- outputs ---------------------------------------------------------------------------
ev_write_tables <- function(cfg, paths, prep, de, pe, cutpoint, registry, compat) {
  td <- file.path(paths$out, "tables")
  w <- function(df, f) ev_write_csv(df, file.path(td, f))
  w(registry, "external_model_registry.csv")
  w(compat, "external_model_level_compatibility.csv")
  w(ev_baseline_characteristics(prep$diag, prep$prog), "external_baseline_characteristics.csv")
  w(de$metrics, "diagnostic_external_metrics.csv")
  w(de$roc, "diagnostic_external_roc_curve.csv")
  w(de$calibration, "diagnostic_external_calibration.csv")
  w(de$dca, "diagnostic_external_dca.csv")
  pm <- pe$metrics
  w(pm, "prognostic_external_metrics.csv")
  w(pm[pm$metric == "c_index", ], "prognostic_external_discrimination.csv")
  w(pm[pm$metric == "time_auc", ], "prognostic_external_time_auc.csv")
  w(pe$roc, "prognostic_external_roc_curve.csv")
  w(pm[pm$metric %in% c("brier_ipcw", "km_observed_survival", "mean_predicted_survival",
                        "calibration_gap_km_minus_predicted"), ], "prognostic_external_calibration.csv")
  w(pe$grouped_calibration, "prognostic_external_calibration_groups.csv")
  w(pe$dca, "prognostic_external_dca.csv")
  km <- pe$km
  w(km$curve, "prognostic_external_km_curve.csv")
  w(km$risk_table, "prognostic_external_km_risk_table.csv")
  ks <- km$summary
  ks$cutpoint_applied <- cutpoint
  ks$cutpoint_source <- cfg$evaluation$km$cutpoint_source
  ks$cutpoint_rule <- cfg$evaluation$km$cutpoint_rule
  ks$cutpoint_estimated_from_external_data <- FALSE
  ks$cutpoint_rounding_sensitive <- km$assignment_margin < 1e-6
  ks$logrank_status <- km$logrank_status; ks$logrank_chisq <- km$logrank_chisq; ks$logrank_p <- km$logrank_p
  ks$evaluation_set <- "model_specific"; ks$n_evaluable <- km$n; ks$status <- km$status
  w(ks, "prognostic_external_km_summary.csv")
  led <- ev_build_ledger(cfg, de, pe, compat)
  w(led, "external_supportedness_ledger.csv")
  w(ev_internal_context(cfg, paths, de, pe), "internal_external_context.csv")
  w(ev_report_summary_diag(cfg, de), "report_summary_diagnostic.csv")
  w(ev_report_summary_prog(cfg, pe), "report_summary_prognostic.csv")
  ev_write_csv(de$fast_vs_package, file.path(paths$out, "validation", "estimator_agreement_diagnostic.csv"))
  ev_write_csv(pe$brier_check, file.path(paths$out, "validation", "estimator_agreement_brier_ipcw.csv"))
  led
}

ev_handoff <- function(cfg, paths, prep, de, pe, figs, rmf, led, completion) {
  tbls <- file.path("analysis/04_external_validation/outputs/tables",
                    c(ev_required_tables(), "external_model_level_compatibility.csv", "prognostic_external_metrics.csv",
                      "prognostic_external_calibration_groups.csv"))
  mrow <- function(M, table) lapply(seq_len(nrow(M)), function(i) {
    r <- M[i, ]
    list(metric_key = r$metric_key, task = r$task, model = r$model, evaluation_set = r$evaluation_set,
         metric = r$metric, horizon_months = r$horizon_months, estimate = r$estimate, ci_lower_95 = r$ci_lower_95,
         ci_upper_95 = r$ci_upper_95, ci_method = r$ci_method, ci_displayable = r$ci_displayable,
         n_evaluable = r$n_evaluable, bootstrap_draws_valid = r$bootstrap_draws_valid, support_status = r$support_status,
         limitation_code = r$limitation_code, source_table = table, source_row = i)
  })
  pm <- pe$metrics
  metrics <- c(mrow(de$metrics, "tables/diagnostic_external_metrics.csv"), mrow(pm, "tables/prognostic_external_metrics.csv"))
  reg <- ev_read_csv(file.path(paths$out, "tables", "external_model_registry.csv"))
  list(
    run_id = cfg$run_id, generated = ev_now(), completion_status = completion$status,
    nature = "factual execution handoff; contains no interpretation or manuscript claims",
    source = list(workbook = cfg$source$workbook_path, sha256 = rmf$identity$workbook_sha256, worksheet = cfg$source$expected_sheet),
    cohorts = list(diagnostic = list(n = nrow(prep$diag), mets_liver_positive = sum(prep$diag$mets_liver == 1),
                                     mets_liver_negative = sum(prep$diag$mets_liver == 0)),
                   prognostic = list(n = nrow(prep$prog), deaths = sum(prep$prog$os_event == 1),
                                     censored = sum(prep$prog$os_event == 0))),
    model_registry = lapply(seq_len(nrow(reg)), function(i) as.list(reg[i, c("task", "model", "object_role", "role", "origin",
                                                                              "path", "sha256", "sha256_matches_expected", "class",
                                                                              "coefficient_sha256", "training_n")])),
    diagnostic_tnm_comparator = rmf$diagnostic_tnm_comparator,
    metrics = metrics,
    figures = lapply(seq_len(nrow(figs)), function(i) as.list(figs[i, ])),
    tables = as.list(tbls),
    status_codes = as.list(table(led$support_status)),
    limitation_codes = as.list(table(unlist(strsplit(c(de$metrics$limitation_code, pm$limitation_code, led$limitation_code), ";")))),
    bootstrap = rmf$bootstrap,
    commands = as.list(ev_commands()),
    session = list(r_version = R.version.string, platform = R.version$platform, packages = ev_pkg_versions(),
                   cpu_only = TRUE),
    completeness_validation = completion$validation_summary,
    rules = list("frozen models only; no refitting, recalibration, threshold or cutpoint selection on external data",
                 "calibration intercept/slope are evaluation metrics only",
                 "intervals: percentile bootstrap on one shared schedule per task; failed draws counted, not redrawn",
                 "ci_displayable is TRUE only for >= 1000 valid draws and >= 50% valid (or analytic Greenwood KM CI)",
                 "internal SEER metrics in internal_external_context.csv are copied, not re-estimated, and not equivalent"))
}

ev_execution_md <- function(cfg, paths, prep, de, pe, figs, rmf, led, completion, mode) {
  st <- table(led$support_status)
  nonsup <- led[led$support_status != "supported", c("task", "model", "evaluation_set", "result", "horizon_months", "support_status", "limitation_code")]
  agg <- if (nrow(nonsup)) stats::aggregate(ledger ~ task + result + support_status,
                                            data = transform(nonsup, ledger = 1), FUN = length) else NULL
  L <- c(
    sprintf("# Execution handoff - `%s` (UK-Erlangen external validation)", cfg$run_id), "",
    "Technical record only: no scientific interpretation, ranking or clinical recommendation. Paths are relative to",
    "`analysis/04_external_validation/`.", "",
    "## 1. What ran", "",
    sprintf("- Completion status: `%s`", completion$status),
    sprintf("- Mode of last aggregate build: `%s`; run manifest status: `%s`", mode, completion$manifest_status),
    sprintf("- Independent validation: %s (`outputs/validation/completeness_validation.json`)", completion$validation_line),
    "- Script 1 (Python) prepared the workbook; Script 2 (R, CPU only) applied the frozen models, bootstrapped, tabulated and plotted.",
    "", "Commands (from `GithubRepo/`):", "", "```", ev_commands()[c("run", "run2")], "```", "",
    "## 2. What was prepared", "",
    sprintf("- Source: `%s`, worksheet `%s`, SHA-256 `%s` (unchanged).", cfg$source$workbook_path, cfg$source$expected_sheet, rmf$identity$workbook_sha256),
    sprintf("- Diagnostic cohort: %d rows (%d with liver metastasis, %d without).", nrow(prep$diag), sum(prep$diag$mets_liver == 1), sum(prep$diag$mets_liver == 0)),
    sprintf("- Prognostic cohort (observed mets_liver == 1): %d rows (%d deaths, %d censored).", nrow(prep$prog), sum(prep$prog$os_event == 1), sum(prep$prog$os_event == 0)),
    "- Column schemas and order equal `seer_diagnostic_input.csv` / `seer_prognostic_input.csv`. Mapping rules: `outputs/tables/external_variable_mapping.csv`.",
    "- Genuine source missingness kept as missing: marital status `Unkown`/unknown -> `married` NA; AJCC N `Missing` -> NA; `<20`/`>40` tumour sizes -> group only, `tumor_size_mm` NA.",
    "", "## 3. Frozen models evaluated", "", "| task | model | role | origin | SHA-256 (first 12) | matches expected hash |", "|---|---|---|---|---|---|")
  reg <- ev_read_csv(file.path(paths$out, "tables", "external_model_registry.csv"))
  L <- c(L, sprintf("| %s | %s (%s) | %s | %s | `%s` | %s |", reg$task, reg$model, reg$object_role, reg$role, reg$origin,
                    substr(reg$sha256, 1, 12), ifelse(is.na(reg$sha256_matches_expected), "n/a (not listed)", as.character(reg$sha256_matches_expected))))
  rc <- rmf$diagnostic_tnm_comparator
  L <- c(L, "", sprintf("Diagnostic AJCC T/N comparator: `%s`, saved by `%s` (reference training code, SEER derivation data, n = %d). SEER AUC of the saved object reproduces `roc_auc_diagnostic.csv` (%.4f vs %.4f): %s. No model was fitted in this component.",
                        rc$path, rc$saved_by, rc$training_n, rc$seer_internal_auc_check$recomputed_rounded,
                        rc$seer_internal_auc_check$table_value, rc$seer_internal_auc_check$passed),
         "", "## 4. Output inventory", "",
         sprintf("- Tables: %d CSV files in `outputs/tables/` (see `result_inventory.csv`).", length(list.files(file.path(paths$out, "tables")))),
         sprintf("- Figures: %d families x (PDF + PNG) in `outputs/figures/`, each with `outputs/figure_data/<family>.csv`:", nrow(figs)))
  L <- c(L, sprintf("  - `%s`", figs$family))
  L <- c(L, "- Manifests: `outputs/run_manifest.json`, `outputs/manifests/report_handoff.json`, pre-task snapshot files.",
         sprintf("- Metric rows: %d diagnostic, %d prognostic; supportedness ledger rows: %d.", nrow(de$metrics), nrow(pe$metrics), nrow(led)),
         sprintf("- Bootstrap: seed %d, %d planned draws per task; schedule SHA-256 diagnostic `%s`, prognostic `%s`.",
                 cfg$bootstrap$seed, cfg$bootstrap$draws, substr(rmf$bootstrap$diagnostic_schedule_sha256, 1, 16),
                 substr(rmf$bootstrap$prognostic_schedule_sha256, 1, 16)),
         "", "## 5. Tests and independent validations", "",
         "- Python tests: `tests/test_uk_erlangen_preparation.py`; R/estimator tests: `tests/test_uk_erlangen_evaluation.py` (drives `tests/uk_erlangen_r_checks.R`).",
         sprintf("- Validator checks: %s", completion$validation_line),
         "- Fast bootstrap engines vs package point estimates: `outputs/validation/estimator_agreement_*.csv`.",
         sprintf("- Fresh-process prediction recomputation: %s.", rmf$fresh_process_prediction_check$identical),
         "", "## 6. Unsupported or non-estimable results", "",
         paste0("Status counts (ledger): ", paste(sprintf("`%s` %d", names(st), as.integer(st)), collapse = ", "), "."), "")
  if (!is.null(agg)) {
    L <- c(L, "| task | result | status | ledger rows |", "|---|---|---|---|",
           sprintf("| %s | %s | `%s` | %d |", agg$task, agg$result, agg$support_status, agg$ledger))
  }
  L <- c(L, "", "Reason codes are in `limitation_code` of `external_supportedness_ledger.csv`. Survival-DCA panels follow the original >= 20 eligible / >= 5 events rule and are `unsupported_small_sample`; their curves are exploratory data only.",
         "", "## 7. Privacy handling", "",
         "- Raw `Code` values exist only in the workbook and in `outputs/local_only/privacy/code_to_pseudonym_map.csv`; pseudonyms are HMAC-SHA256 with a local key in `outputs/local_only/secrets/`.",
         "- Patient-level prepared data, predictions, schedules and draw-level results are under `outputs/local_only/` (git-ignored by a narrow `.gitignore` block).",
         "- Public tables hold aggregates only; ROC tables omit threshold values; descriptive tables use one-variable counts and median [Q1, Q3] without extremes.",
         "", "## 8. Resume / rebuild commands", "", "```", ev_commands()[c("dry1", "validate1", "validate2", "rebuild")], "```", "",
         "Script 2 refuses to continue under the same run ID if any model, source, configuration, code or package version hash changed; a completed run manifest is immutable.",
         "", "## 9. Report files", "",
         "The evaluation writes only to this component's output directory; the final report and curated result snapshots are not modified.",
         "Authorised reference change (user follow-up): `01_RefPaper_DiagnosticModel` script 04 now saves `logistic_model_tnm_baseline.rds` and scripts 06/08/10 load it; the reference pipeline was re-run. Validator check 13 lists every changed or new file under that prefix.")
  L
}

ev_finalize_status <- function(led, val) {
  if (!isTRUE(val$all_passed)) return("failed_validation")
  if (all(led$support_status == "supported")) "complete" else "complete_with_documented_small_sample_limitations"
}

# ---- main ----------------------------------------------------------------------------------
main <- function() {
  args <- ev_parse_args(commandArgs(trailingOnly = TRUE))
  script <- ev_script_path()
  source(file.path(dirname(dirname(script)), "src", "external_validation_helpers.R"))
  paths <- ev_paths(args$config)
  cfg <- ev_load_cfg(paths)
  ev_require()
  options(warn = 1)
  if (!is.null(args$internal)) return(ev_internal(args, cfg, paths))
  if (args$validate_only) {
    res <- ev_run_validator(cfg, paths)
    return(if (res$all_passed) 0L else 4L)
  }
  mode <- if (args$rebuild) "rebuild_aggregates" else "run"
  rm_path <- file.path(paths$out, "run_manifest.json")
  identity <- ev_identity(cfg, paths)
  rmf <- if (file.exists(rm_path)) jsonlite::fromJSON(rm_path, simplifyVector = FALSE) else NULL
  if (!is.null(rmf)) {
    d <- ev_identity_diff(rmf$identity, identity)
    if (length(d)) { ev_say("RESUME REFUSED: identity changed under run ID ", cfg$run_id, ": ", paste(d, collapse = ", ")); return(3L) }
    if (grepl("^complete", rmf$status) && !args$rebuild) {
      ev_say("run already ", rmf$status, " (immutable manifest). Use --validate-only or --rebuild-aggregates."); return(0L)
    }
  } else if (args$rebuild) { ev_say("BLOCKER: no run manifest; run the full stage first."); return(2L) }

  # preflight on the preparation stage
  pm <- jsonlite::fromJSON(file.path(paths$module, cfg$outputs$preparation_manifest), simplifyVector = FALSE)
  pre_ok <- identical(pm$status, "passed") && identical(pm$identity$config_sha256, identity$config_sha256) &&
    identical(pm$identity$workbook_sha256, identity$workbook_sha256) &&
    identical(pm$outputs_sha256[[cfg$outputs$prepared_diagnostic]], identity$prepared_sha256[[cfg$outputs$prepared_diagnostic]]) &&
    identical(pm$outputs_sha256[[cfg$outputs$prepared_prognostic]], identity$prepared_sha256[[cfg$outputs$prepared_prognostic]])
  if (!pre_ok) {
    ev_write_json(list(run_id = cfg$run_id, status = "blocked", reason = "preparation manifest/hashes do not match current inputs"),
                  file.path(paths$out, "validation", "evaluation_blocker.json"))
    ev_say("BLOCKER: preparation manifest does not match current inputs"); return(2L)
  }
  prep <- ev_load_prepared(cfg, paths)
  frozen <- !is.null(rmf) && grepl("^complete", rmf$status)
  if (is.null(rmf)) {
    rmf <- list(run_id = cfg$run_id, status = "in_progress", started = ev_now(), immutable_after_completion = TRUE,
                identity = identity)
    ev_write_json(rmf, rm_path)
  }
  pdir <- file.path(paths$local, "predictions"); bdir <- file.path(paths$local, "bootstrap")

  if (!args$rebuild) {
    ev_say("model integrity: loading frozen models")
    for (s in ev_model_specs(cfg, paths)) invisible(readRDS(s$abs_path))
    tmpd <- file.path(paths$local, "tmp"); dir.create(tmpd, recursive = TRUE, showWarnings = FALSE)
    tspec <- Filter(function(m) m$key == "tnm_baseline", cfg$models$diagnostic)[[1]]
    tnm <- readRDS(ev_repo(paths, tspec$path))
    seer <- readRDS(ev_repo(paths, "analysis/03_reference_models/diagnostic/outputs/models/diagnostic_data_clean.rds"))
    auc_seer <- as.numeric(pROC::auc(pROC::roc(seer$mets_liver, stats::predict(tnm, newdata = seer, type = "response"),
                                               direction = "<", quiet = TRUE, levels = c(0, 1))))
    tv <- utils::read.csv(ev_repo(paths, "analysis/03_reference_models/diagnostic/outputs/tables/roc_auc_diagnostic.csv"))
    tv <- tv$auc[tv$model == "tnm_baseline"]
    rmf$diagnostic_tnm_comparator <- list(
      origin = "frozen_saved_rds", path = tspec$path, saved_by = tspec$saved_by,
      sha256 = ev_sha256_file(ev_repo(paths, tspec$path)), coefficient_sha256 = ev_coef_hash(tnm),
      formula = paste(deparse(stats::formula(tnm)), collapse = " "), training_n = length(tnm$y),
      seer_internal_auc_check = list(recomputed = auc_seer, recomputed_rounded = round(auc_seer, 4), table_value = tv,
                                     passed = isTRUE(round(auc_seer, 4) == tv)))
    if (!isTRUE(round(auc_seer, 4) == tv)) { ev_say("BLOCKER: saved T/N comparator does not reproduce the SEER AUC table"); return(2L) }

    ev_say("generating frozen-model predictions")
    pr <- ev_make_predictions(cfg, paths, prep$diag, prep$prog)
    ev_write_csv(pr$diagnostic, file.path(pdir, "diagnostic_predictions.csv"))
    ev_write_csv(pr$prognostic, file.path(pdir, "prognostic_predictions.csv"))
    ev_write_csv(pr$compatibility, file.path(paths$local, "predictions", "compatibility.csv"))
    ev_say("fresh-process recomputation of predictions")
    f3 <- ev_fresh(paths, c("--internal-fresh-check", "predictions",
                            "--out", shQuote(file.path(tmpd, "fresh_predictions"), type = "cmd")))
    fd <- ev_read_csv(file.path(tmpd, "fresh_predictions_diagnostic.csv")); fp <- ev_read_csv(file.path(tmpd, "fresh_predictions_prognostic.csv"))
    sd0 <- ev_read_csv(file.path(pdir, "diagnostic_predictions.csv")); sp0 <- ev_read_csv(file.path(pdir, "prognostic_predictions.csv"))
    rmf$fresh_process_prediction_check <- list(exit_status = f3$status, identical = identical(fd, sd0) && identical(fp, sp0))
    if (!rmf$fresh_process_prediction_check$identical) { ev_say("BLOCKER: fresh-process predictions differ"); return(2L) }
    ev_say("generating bootstrap schedules (before any metric)")
    sd <- ev_schedule_diagnostic(prep$diag$mets_liver, cfg); sp <- ev_schedule_prognostic(nrow(prep$prog), cfg)
    ev_save_rds(sd, file.path(bdir, "diagnostic_schedule.rds")); ev_save_rds(sp, file.path(bdir, "prognostic_schedule.rds"))
    rmf$bootstrap <- list(seed = cfg$bootstrap$seed, rng_kind = ev_chr(cfg$bootstrap$rng_kind), draws = cfg$bootstrap$draws,
                          diagnostic_sampling = cfg$bootstrap$diagnostic_sampling,
                          prognostic_sampling = cfg$bootstrap$prognostic_sampling,
                          diagnostic_schedule_sha256 = ev_schedule_hash(sd), prognostic_schedule_sha256 = ev_schedule_hash(sp),
                          failed_draws_redrawn = FALSE)
    unlink(tmpd, recursive = TRUE)
    compat <- pr$compatibility
    dp <- sd0; ppd <- sp0   # evaluate from the saved CSVs so --rebuild-aggregates reproduces byte-identically
  } else {
    ev_say("rebuild: loading saved local predictions and schedules (no refitting, no regeneration)")
    chk <- c(identical(rmf$local_outputs_sha256$diagnostic_predictions, ev_sha256_file(file.path(pdir, "diagnostic_predictions.csv"))),
             identical(rmf$local_outputs_sha256$prognostic_predictions, ev_sha256_file(file.path(pdir, "prognostic_predictions.csv"))))
    if (!all(chk)) { ev_say("RESUME REFUSED: local predictions differ from the run manifest"); return(3L) }
    dp <- ev_read_csv(file.path(pdir, "diagnostic_predictions.csv")); ppd <- ev_read_csv(file.path(pdir, "prognostic_predictions.csv"))
    compat <- ev_read_csv(file.path(pdir, "compatibility.csv"))
    sd <- readRDS(file.path(bdir, "diagnostic_schedule.rds")); sp <- readRDS(file.path(bdir, "prognostic_schedule.rds"))
    if (!identical(ev_schedule_hash(sd), rmf$bootstrap$diagnostic_schedule_sha256) ||
        !identical(ev_schedule_hash(sp), rmf$bootstrap$prognostic_schedule_sha256)) { ev_say("RESUME REFUSED: schedule hash mismatch"); return(3L) }
  }

  cutpoint <- utils::read.csv(ev_repo(paths, cfg$evaluation$km$cutpoint_source))$cutpoint
  ev_say("diagnostic evaluation (", cfg$bootstrap$draws, " stratified draws)")
  de <- ev_evaluate_diagnostic(cfg, dp, sd)
  ev_say("prognostic evaluation (", cfg$bootstrap$draws, " patient-level draws)")
  pe <- ev_evaluate_prognostic(cfg, ppd, sp, cutpoint)
  ev_save_rds(list(diagnostic = de$draws, prognostic = pe$draws), file.path(bdir, "draw_level_results.rds"))
  if (max(de$fast_vs_package$max_abs_diff_fast_vs_package) > 1e-10 || max(pe$brier_check$abs_diff, na.rm = TRUE) > 1e-10) {
    ev_say("BLOCKER: fast bootstrap engine disagrees with package point estimates"); return(2L)
  }
  registry <- ev_model_registry(cfg, paths)
  if (!all(registry$declared_predictors_match)) { ev_say("BLOCKER: declared predictors differ from model terms"); return(2L) }
  ev_say("writing tables and figures")
  led <- ev_write_tables(cfg, paths, prep, de, pe, cutpoint, registry, compat)
  figs <- ev_make_figures(cfg, paths, de, pe, cutpoint)
  rmf$bootstrap$valid_draws <- list(
    diagnostic = stats::setNames(as.list(tapply(de$metrics$bootstrap_draws_valid, de$metrics$metric, min)), unique(sort(de$metrics$metric))),
    prognostic_min_by_metric = as.list(tapply(pe$metrics$bootstrap_draws_valid, pe$metrics$metric, function(x) if (all(is.na(x))) NA else min(x, na.rm = TRUE))))
  if (!frozen) {
    rmf$local_outputs_sha256 <- list(
      diagnostic_predictions = ev_sha256_file(file.path(pdir, "diagnostic_predictions.csv")),
      prognostic_predictions = ev_sha256_file(file.path(pdir, "prognostic_predictions.csv")),
      draw_level_results = ev_sha256_file(file.path(bdir, "draw_level_results.rds")))
    rmf$commands <- as.list(ev_commands()[c("run", "run2")])
    ev_write_json(rmf, rm_path)
  }
  completion <- list(status = "pending_independent_validation", manifest_status = rmf$status,
                     validation_line = "pending", validation_summary = NULL)
  ev_write_json(ev_handoff(cfg, paths, prep, de, pe, figs, rmf, led, completion), file.path(paths$out, "manifests", "report_handoff.json"))
  ev_write_text(ev_execution_md(cfg, paths, prep, de, pe, figs, rmf, led, completion, mode), file.path(paths$out, "EXECUTION_HANDOFF.md"))
  ev_write_csv(ev_inventory(paths), file.path(paths$out, "tables", "result_inventory.csv"))

  ev_say("independent validation in a fresh process")
  vr <- ev_fresh(paths, "--validate-only")
  val <- jsonlite::fromJSON(file.path(paths$out, "validation", "completeness_validation.json"), simplifyVector = FALSE)
  status <- ev_finalize_status(led, val)
  if (!frozen) {
    rmf$status <- status; rmf$completed <- ev_now()
    rmf$completeness_validation <- list(all_passed = val$all_passed, n_passed = val$n_passed, n_checks = val$n_checks)
    ev_write_json(rmf, rm_path)
  }
  completion <- list(status = status, manifest_status = rmf$status,
                     validation_line = sprintf("%d/%d checks passed (all_passed = %s)", val$n_passed, val$n_checks, val$all_passed),
                     validation_summary = list(all_passed = val$all_passed, n_passed = val$n_passed, n_checks = val$n_checks,
                                               failed = Filter(Negate(is.null), lapply(val$checks, function(x) if (!isTRUE(x$passed)) x$check)),
                                               file = "analysis/04_external_validation/outputs/validation/completeness_validation.json"))
  ev_write_json(ev_handoff(cfg, paths, prep, de, pe, figs, rmf, led, completion), file.path(paths$out, "manifests", "report_handoff.json"))
  ev_write_text(ev_execution_md(cfg, paths, prep, de, pe, figs, rmf, led, completion, mode), file.path(paths$out, "EXECUTION_HANDOFF.md"))
  ev_say("status: ", status, " | validator ", completion$validation_line)
  if (isTRUE(val$all_passed)) 0L else 4L
}

if (sys.nframe() == 0L) {
  rc <- tryCatch(main(), error = function(e) { cat("[02 ERROR] ", conditionMessage(e), "\n", sep = ""); 2L })
  quit(save = "no", status = if (is.numeric(rc)) rc else 0L)
}
