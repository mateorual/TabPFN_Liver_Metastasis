# scripts/03_run_all_embedding_diagnostic.R
# Run the complete embedding-based diagnostic pipeline in order.
#   Rscript scripts/03_run_all_embedding_diagnostic.R
#   or from R: source("scripts/03_run_all_embedding_diagnostic.R")

cat(strrep("=", 65), "\n")
cat("  Diagnostic Model — TabPFN Embedding Pipeline\n")
cat(strrep("=", 65), "\n\n")

pipeline <- c(
  "scripts/00_setup_embedding_diagnostic.R",
  "scripts/01_embedding_model.R",
  "scripts/02_combined_evaluation_figure.R"
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

check <- function(rel_path) {
  full <- file.path(PROJECT_ROOT, "outputs", rel_path)
  if (file.exists(full)) "[OK]     " else "[MISSING]"
}

cat("Figures:\n")
cat(" ", check("figures/fig_embedding_diagnostic_evaluation.png"),
    "figures/fig_embedding_diagnostic_evaluation.png\n")

cat("\nTables:\n")
for (f in sort(list.files(file.path(PROJECT_ROOT, "outputs", "tables")))) {
  cat("  ", f, "\n")
}

cat("\nModels:\n")
for (f in sort(list.files(file.path(PROJECT_ROOT, "outputs", "models")))) {
  cat("  ", f, "\n")
}

cat("\nDone.\n")
