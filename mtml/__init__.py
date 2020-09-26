__doc__ = "``__init__.py`` for ``mtml`` package."

# useful global constants
trauma_score_cols = (
    "ais head1", "ais face2", "ais chest3", "ais abdomen4", "ais extremity5",
    "ais skin6", "iss", "APACHE-2", "GCS (ED arrival)"
)
"Labels for trauma score columns to drop from data set."

disease_comorb_cols = (
    "hiv", "aids", "hepatic failure", "immuno-suppression", "asthma", "copd",
    "ild", "other chronic lung disease", "cad", "chf", "esrd dialysis",
    "cirrhosis", "diabetes", "malignancy"
)
"Pre-existing disease and comorbidity column labels."

vitals_cols = (
    "hr0_temperature", "hr0_heart rate", "hr0_respiratory rate", "hr0_SBP",
    "hr0_pH", "hr0_paCO2", "hr0_paO2", "hr0_HCO3", "hr0_serum CO2"
)
"Vitals column labels."

lab_panel_cols = (
    "hr0_lactate", "albumin", "day1_bilirubin", "day1_urine output total (ml)",
    "hr0_BUN", "hr0_creatinine", "hr0_wbc", "hr0_hgb", "hr0_hct", "hr0_plts",
    "hr0_pt", "hr0_pt_rlab", "hr0_ptt", "hr0_ptt_rlab", "hr0_inr",
    "hr0_fibrinogen", "hr0_factorii", "hr0_factorv", "hr0_factorvii",
    "hr0_factorix", "hr0_factorx", "hr0_tfpi", "hr0_factorviii", "hr0_atiii",
    "hr0_Protein C", "hr0_D-Dimer"
)
"Lab panel column labels."