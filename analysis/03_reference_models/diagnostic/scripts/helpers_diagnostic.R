# scripts/helpers_diagnostic.R — shared helpers for plot labels and variable display.

pretty_var_label <- function(x) {
  lookup <- c(
    age_at_diagnosis   = "Age",
    race               = "Race",
    sex_male           = "Sex",
    married            = "Marital status",
    histology_paper    = "Histology",
    grade              = "Grade",
    primary_site_group = "Primary site",
    ajcc_t             = "T stage",
    n_stage_positive   = "N stage",
    surgery_primary    = "Surgery",
    surgery_lymph_node = "LN surgery",
    radiotherapy       = "Radiotherapy",
    chemotherapy       = "Chemotherapy",
    tumor_size_group   = "Tumor size",
    mets_bone          = "Bone mets",
    mets_brain         = "Brain mets",
    mets_lung          = "Lung mets"
  )
  ifelse(x %in% names(lookup), lookup[x], x)
}

format_feature_list <- function(vars, max_chars = 110) {
  txt <- paste(pretty_var_label(vars), collapse = " + ")
  if (nchar(txt) > max_chars)
    txt <- paste0(substr(txt, 1, max_chars - 3), "...")
  txt
}

model_display_label <- function(model_name, vars, auc_text = NULL,
                                max_chars = 110) {
  feat_str <- format_feature_list(vars, max_chars = max_chars)
  base <- paste0(model_name, " (", length(vars), " vars: ", feat_str, ")")
  if (!is.null(auc_text)) base <- paste0(base, " | AUC=", auc_text)
  base
}
