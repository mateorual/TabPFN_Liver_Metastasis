# scripts/05_run_all_embedding_prognostic.R
# Run the complete embedding-based prognostic pipeline in order.
#   Rscript scripts/05_run_all_embedding_prognostic.R
#   or from R: source("scripts/05_run_all_embedding_prognostic.R")

cat(strrep("=", 65), "\n")
cat("  Prognostic Model — TabPFN Embedding Pipeline\n")
cat(strrep("=", 65), "\n\n")

pipeline <- c(
  "scripts/00_setup_embedding_prognostic.R",
  "scripts/01_embedding_cox_model.R",
  "scripts/02_time_dependent_roc.R",
  "scripts/03_dca.R",
  "scripts/04_km_risk_groups.R"
)

start_total <- proc.time()

for (script in pipeline) {
  cat(strrep("-", 65), "\n")
  cat("  Stage:", script, "\n")
  cat(strrep("-", 65), "\n")
  t_start <- proc.time()

  tryCatch(
    withCallingHandlers(
      source(script, echo = FALSE, local = FALSE),
      warning = function(w) {
        cat("  [WARNING]", conditionMessage(w), "\n")
        invokeRestart("muffleWarning")
      }
    ),
    error = function(e) {
      cat("\n[PIPELINE ERROR] Script failed:", script, "\n")
      cat("  Error message:", conditionMessage(e), "\n")
      stop("Pipeline aborted at: ", script)
    }
  )

  elapsed <- proc.time() - t_start
  cat("  Elapsed:", round(elapsed["elapsed"], 1), "s\n\n")
}

elapsed_total <- proc.time() - start_total
cat(strrep("=", 65), "\n")
cat("  Pipeline completed in", round(elapsed_total["elapsed"], 1), "seconds.\n")
cat(strrep("=", 65), "\n\n")

source("config/config.R", local = TRUE)

MAIN_FIGURES <- c(
  "figures/fig6_roc_training_internal_6panel.png",
  "figures/fig7_dca_training_internal_6panel.png",
  "figures/fig8_km_risk_groups.png"
)

check <- function(rel_path) {
  full <- file.path(PROJECT_ROOT, "outputs", rel_path)
  if (file.exists(full)) "[OK]     " else "[MISSING]"
}

cat("Figures:\n")
for (f in MAIN_FIGURES) cat(" ", check(f), f, "\n")

cat("\nTables:\n")
for (f in sort(list.files(file.path(PROJECT_ROOT, "outputs", "tables")))) {
  cat("  ", f, "\n")
}

cat("\nModels:\n")
for (f in sort(list.files(file.path(PROJECT_ROOT, "outputs", "models")))) {
  cat("  ", f, "\n")
}

cat("\nDone.\n")
