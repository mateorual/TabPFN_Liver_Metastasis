# config/config.R — paths, seed, and modeling settings for the diagnostic
# TabPFN-embedding pipeline.
# Assumes the working directory is this implementation's root when sourced.

PROJECT_ROOT <- getwd()
REPO_ROOT    <- dirname(dirname(dirname(dirname(PROJECT_ROOT))))

# TabPFN out-of-fold embedding CSV: patient_id, emb_0000..emb_0511, mets_liver
# Produced by ../../00_TabPFN_Transformations/01_Embedding_Transformation
EMBEDDING_INPUT_FILE <- file.path(REPO_ROOT, "data", "restricted", "tabpfn_embeddings",
                                  "seer_diagnostic_tabpfn_embeddings.csv")

# Real clinical CSV — used only to pull ajcc_t / n_stage_positive for the AJCC
# T/N baseline comparator (embeddings carry no named clinical variables).
CLINICAL_INPUT_FILE <- file.path(REPO_ROOT, "data", "restricted", "seer_processed",
                                 "seer_diagnostic_input.csv")

TABLES_DIR  <- file.path(PROJECT_ROOT, "outputs", "tables")
FIGURES_DIR <- file.path(PROJECT_ROOT, "outputs", "figures")
MODELS_DIR  <- file.path(PROJECT_ROOT, "outputs", "models")

SEED <- 20230529

OUTCOME <- "mets_liver"
EMBEDDING_PREFIX <- "emb_"

FEATURES_TNM_BASELINE <- c("ajcc_t", "n_stage_positive")

# Ridge (alpha=0) is used for the embedding model: embeddings are dense,
# correlated, non-sparse features, unlike the clinical pipeline's LASSO on
# named clinical variables, so an L1 sparsity assumption doesn't apply here.
GLMNET_ALPHA   <- 0
GLMNET_NFOLDS  <- 10

# Appends one row to model_sample_sizes_embedding_diagnostic.csv
log_sample_size <- function(script, model_name, n_total, n_used) {
  row <- data.frame(
    script     = script,
    model_name = model_name,
    n_total    = n_total,
    n_used     = n_used,
    n_dropped  = n_total - n_used,
    stringsAsFactors = FALSE
  )
  csv_path <- file.path(TABLES_DIR, "model_sample_sizes_embedding_diagnostic.csv")
  if (file.exists(csv_path)) {
    prev <- tryCatch(read.csv(csv_path, stringsAsFactors = FALSE),
                     error = function(e) data.frame())
    write.csv(rbind(prev, row), csv_path, row.names = FALSE)
  } else {
    write.csv(row, csv_path, row.names = FALSE)
  }
}
