# config/config.R — paths, seed, and modeling settings for the prognostic
# TabPFN-embedding pipeline.
# Assumes the working directory is this implementation's root when sourced.

PROJECT_ROOT <- getwd()
REPO_ROOT    <- dirname(dirname(dirname(dirname(PROJECT_ROOT))))

# TabPFN out-of-fold embedding CSV: patient_id, event_emb_0000..0511,
# time_emb_0000..0511, mets_liver, os_months, os_event
# Produced by ../../00_TabPFN_Transformations/01_Embedding_Transformation
EMBEDDING_INPUT_FILE <- file.path(REPO_ROOT, "data", "restricted", "tabpfn_embeddings",
                                  "seer_prognostic_tabpfn_embeddings.csv")

# Real clinical CSV — used only to pull ajcc_t / n_stage_positive for the AJCC
# T/N baseline comparator (embeddings carry no named clinical variables).
CLINICAL_INPUT_FILE <- file.path(REPO_ROOT, "data", "restricted", "seer_processed",
                                 "seer_prognostic_input.csv")

TABLES_DIR  <- file.path(PROJECT_ROOT, "outputs", "tables")
FIGURES_DIR <- file.path(PROJECT_ROOT, "outputs", "figures")
MODELS_DIR  <- file.path(PROJECT_ROOT, "outputs", "models")

SEED <- 20230529

TIME_VAR  <- "os_months"
EVENT_VAR <- "os_event"

EVENT_EMB_PREFIX <- "event_emb_"
TIME_EMB_PREFIX  <- "time_emb_"

FEATURES_TN_BASELINE <- c("ajcc_t", "n_stage_positive")

TIME_HORIZONS_MONTHS <- c(12, 24, 36)
TIME_HORIZONS_LABELS <- c("1-year", "2-year", "3-year")

# Cohort split (mirrors the clinical prognostic pipeline's default)
SPLIT_METHOD <- "simple_random"

# Ridge (alpha=0) penalized Cox: embeddings are dense, correlated, non-sparse
# features (unlike named clinical variables), and p=1024 > n_train=~323, so
# strong regularization is required just for the model to be identifiable.
GLMNET_ALPHA  <- 0
GLMNET_NFOLDS <- 10

# Appends one row to model_sample_sizes_embedding_prognostic.csv
log_sample_size <- function(script, model_name, n_total, n_used) {
  row <- data.frame(
    script     = script,
    model_name = model_name,
    n_total    = n_total,
    n_used     = n_used,
    n_dropped  = n_total - n_used,
    stringsAsFactors = FALSE
  )
  csv_path <- file.path(TABLES_DIR, "model_sample_sizes_embedding_prognostic.csv")
  if (file.exists(csv_path)) {
    prev <- tryCatch(read.csv(csv_path, stringsAsFactors = FALSE),
                     error = function(e) data.frame())
    write.csv(rbind(prev, row), csv_path, row.names = FALSE)
  } else {
    write.csv(row, csv_path, row.names = FALSE)
  }
}
