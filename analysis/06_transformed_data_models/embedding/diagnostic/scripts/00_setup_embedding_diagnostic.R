# scripts/00_setup_embedding_diagnostic.R
# Load the TabPFN diagnostic embedding CSV, join ajcc_t/n_stage_positive from
# the real clinical CSV (by patient_id) for the AJCC T/N baseline comparator,
# validate, and save a clean RDS.

cat("=== 00_setup_embedding_diagnostic.R ===\n")

source("config/config.R")

for (pkg in c("readr", "dplyr")) {
  if (!requireNamespace(pkg, quietly = TRUE))
    stop("Install '", pkg, "' with: install.packages('", pkg, "')")
}
suppressPackageStartupMessages({ library(readr); library(dplyr) })

for (d in c(TABLES_DIR, FIGURES_DIR, MODELS_DIR)) {
  if (!dir.exists(d)) dir.create(d, recursive = TRUE)
}

cat("  Reading embeddings:", EMBEDDING_INPUT_FILE, "\n")
emb <- read_csv(EMBEDDING_INPUT_FILE, show_col_types = FALSE)
cat("  Embedding dimensions:", nrow(emb), "x", ncol(emb), "\n")

emb_cols <- grep(paste0("^", EMBEDDING_PREFIX), names(emb), value = TRUE)
if (length(emb_cols) == 0) stop("No embedding columns found with prefix '", EMBEDDING_PREFIX, "'")
cat("  Embedding columns:", length(emb_cols), "\n")

if (!OUTCOME %in% names(emb)) stop("Outcome '", OUTCOME, "' not found in embeddings CSV.")
if (any(is.na(emb[emb_cols]))) stop("Missing values found in embedding columns.")
stopifnot("Outcome must be binary 0/1" = all(emb[[OUTCOME]] %in% c(0L, 1L)))

cat("  Outcome distribution:\n")
print(table(emb[[OUTCOME]]))

cat("  Reading real clinical CSV for TNM baseline vars:", CLINICAL_INPUT_FILE, "\n")
clin <- read_csv(CLINICAL_INPUT_FILE, show_col_types = FALSE) |>
  select(patient_id, all_of(FEATURES_TNM_BASELINE))

df <- emb |>
  select(patient_id, all_of(emb_cols), all_of(OUTCOME)) |>
  inner_join(clin, by = "patient_id")

n_dropped <- nrow(emb) - nrow(df)
if (n_dropped > 0) {
  warning(n_dropped, " embedding rows had no matching patient_id in the clinical CSV.")
}
cat("  Joined dataset:", nrow(df), "rows (dropped", n_dropped, "unmatched)\n")

df <- df |>
  mutate(ajcc_t = factor(ajcc_t), n_stage_positive = factor(n_stage_positive))

saveRDS(df, file.path(MODELS_DIR, "embedding_diagnostic_data_clean.rds"))
saveRDS(emb_cols, file.path(MODELS_DIR, "embedding_columns.rds"))
cat("  Saved embedding_diagnostic_data_clean.rds (", length(emb_cols), "embedding dims )\n")

log_sample_size("00_setup", "join_clinical_baseline", nrow(emb), nrow(df))

cat("=== 00 complete ===\n\n")
