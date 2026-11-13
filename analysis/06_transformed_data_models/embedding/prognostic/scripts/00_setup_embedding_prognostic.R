# scripts/00_setup_embedding_prognostic.R
# Load the TabPFN prognostic embedding CSV (event_emb_* + time_emb_*), join
# ajcc_t/n_stage_positive from the real clinical CSV for the AJCC T/N
# baseline comparator, split 70/30 into Training / Internal validation
# (mirrors the clinical prognostic pipeline), and save a clean RDS.

cat("=== 00_setup_embedding_prognostic.R ===\n")

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

event_emb_cols <- grep(paste0("^", EVENT_EMB_PREFIX), names(emb), value = TRUE)
time_emb_cols  <- grep(paste0("^", TIME_EMB_PREFIX),  names(emb), value = TRUE)
emb_cols <- c(event_emb_cols, time_emb_cols)
cat("  event_emb columns:", length(event_emb_cols),
    "| time_emb columns:", length(time_emb_cols), "\n")

required_cols <- c("patient_id", TIME_VAR, EVENT_VAR, emb_cols)
missing_cols <- setdiff(required_cols, names(emb))
if (length(missing_cols) > 0) stop("Missing required columns: ", paste(missing_cols, collapse = ", "))
if (any(is.na(emb[emb_cols]))) stop("Missing values found in embedding columns.")
if (!all(emb[[EVENT_VAR]] %in% c(0, 1))) stop("os_event must be 0 or 1.")
if (any(emb[[TIME_VAR]] < 0)) stop("os_months must be >= 0.")

cat("  Reading real clinical CSV for TNM baseline vars:", CLINICAL_INPUT_FILE, "\n")
clin <- read_csv(CLINICAL_INPUT_FILE, show_col_types = FALSE) |>
  select(patient_id, all_of(FEATURES_TN_BASELINE))

df <- emb |>
  select(patient_id, all_of(emb_cols), all_of(TIME_VAR), all_of(EVENT_VAR)) |>
  inner_join(clin, by = "patient_id") |>
  mutate(ajcc_t = factor(ajcc_t), n_stage_positive = factor(n_stage_positive))

n_dropped <- nrow(emb) - nrow(df)
if (n_dropped > 0) warning(n_dropped, " embedding rows had no matching patient_id in the clinical CSV.")
cat("  Joined dataset:", nrow(df), "rows (dropped", n_dropped, "unmatched)\n")

# os_months == 0 breaks Cox partial likelihood; replace with 0.5 (matches
# the clinical prognostic pipeline's handling), preserving the original.
n_zero_time <- sum(df[[TIME_VAR]] == 0)
cat("  os_months == 0:", n_zero_time, "rows\n")
df <- df |>
  mutate(os_months_original = .data[[TIME_VAR]],
         !!TIME_VAR := if_else(.data[[TIME_VAR]] == 0, 0.5, as.numeric(.data[[TIME_VAR]])))

# ---- 70/30 cohort split (simple random, matches clinical pipeline default) --
cat("  SPLIT_METHOD =", SPLIT_METHOD, "\n")
set.seed(SEED)
train_idx <- sample(seq_len(nrow(df)), round(nrow(df) * 0.7), replace = FALSE)
df <- df |>
  mutate(cohort = if_else(row_number() %in% train_idx, "Training", "Internal validation"))
df$cohort <- factor(df$cohort, levels = c("Training", "Internal validation"))

cat("  Training N =", sum(df$cohort == "Training"),
    "| Validation N =", sum(df$cohort == "Internal validation"), "\n")

split_tbl <- as.data.frame(table(cohort = df$cohort, event = df[[EVENT_VAR]]))
split_tbl$cohort_total <- ave(split_tbl$Freq, split_tbl$cohort, FUN = sum)
write_csv(split_tbl, file.path(TABLES_DIR, "split_counts_embedding_prognostic.csv"))
print(split_tbl)

saveRDS(df, file.path(MODELS_DIR, "embedding_prognostic_data_clean.rds"))
saveRDS(event_emb_cols, file.path(MODELS_DIR, "event_emb_columns.rds"))
saveRDS(time_emb_cols,  file.path(MODELS_DIR, "time_emb_columns.rds"))
cat("  Saved embedding_prognostic_data_clean.rds\n")

log_sample_size("00_setup", "join_split", nrow(emb), nrow(df))

cat("=== 00 complete ===\n\n")
