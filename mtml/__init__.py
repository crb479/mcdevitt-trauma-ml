__doc__ = "``__init__.py`` for ``mtml`` package."

import os.path

# define version dunder
__version__ = "0.0.1dev0"

TOP_LEVEL_PACKAGE_PATH = os.path.dirname(os.path.abspath(__file__))
"Absolute path to the top-level directory of the package."

# useful global constants
DF_TRAUMA_SCORE_COLS = (
    "ais head1", "ais face2", "ais chest3", "ais abdomen4", "ais extremity5",
    "ais skin6", "iss", "APACHE-2", "GCS (ED arrival)"
)
"Labels for trauma score columns to drop from SF trauma data set."

SF_DISEASE_COMORB_COLS = (
    "hiv", "aids", "hepatic failure", "immuno-suppression", "asthma", "copd",
    "ild", "other chronic lung disease", "cad", "chf", "esrd dialysis",
    "cirrhosis", "diabetes", "malignancy"
)
"Pre-existing disease and comorbidity column labels in SF trauma data set."

SF_VITALS_COLS = (
    "hr0_temperature", "hr0_heart rate", "hr0_respiratory rate", "hr0_SBP",
    "hr0_pH", "hr0_paCO2", "hr0_paO2", "hr0_HCO3", "hr0_serum CO2"
)
"Vitals column labels in SF trauma data set."

SF_LAB_PANEL_COLS = (
    "hr0_lactate", "albumin", "day1_bilirubin", "day1_urine output total (ml)",
    "hr0_BUN", "hr0_creatinine", "hr0_wbc", "hr0_hgb", "hr0_hct", "hr0_plts",
    "hr0_pt", "hr0_pt_rlab", "hr0_ptt", "hr0_ptt_rlab", "hr0_inr",
    "hr0_fibrinogen", "hr0_factorii", "hr0_factorv", "hr0_factorvii",
    "hr0_factorix", "hr0_factorx", "hr0_tfpi", "hr0_factorviii", "hr0_atiii",
    "hr0_Protein C", "hr0_D-Dimer"
)
"Lab panel column labels in SF trauma data set."

SF_NUM_DATA_PREP_PATH = (
    "/".join(
        [TOP_LEVEL_PACKAGE_PATH, "data", "files", "prep",
         "sf_trauma_data_num.csv"]
    )
)
"""Path to the (minimally) preprocssed SF hospital trauma data.

See the ``README.rst`` in ``mtml/data/files`` for more details.
"""

VTE_INPUT_COLS = (
    "inr_result", "ldl_result", "ptt_result", "d_dimer_result",
    "glucose_result", "crp_result", "fib_result", "trig_result",
    "thrombin_result", "plt_result", "gender_male0_female1", "age",
    "anticoagulant_use_yes1_no0"
)
"""All input columns used for classification on the VTE data set.

Note that we do not include the bmi column, which has 6000+ missing values, and
patient_identifier, which is useless to us. We also drop ``pt_result`` which is
the prothrombin column since it relatively correlated with ``ptt_result``. We
drop ``tot_cholesterol_result`` and ``hdl_result``, which are typically replaced
with their ratio. See
:func:`mtml.demos.vte_models._replace_hdl_tot_chol_with_ratio` for details.
"""

VTE_CONT_INPUT_COLS = (
    "inr_result", "ldl_result", "ptt_result", "d_dimer_result",
    "glucose_result", "crp_result", "fib_result", "trig_result",
    "thrombin_result", "plt_result", "age"
)
"""Continuous input columns used for classification on the VTE data set.

A subset of the columns in :attr:`VTE_INPUT_COLS` where we also drop the binary
columns ``gender_male0_female1`` and ``anticoagulant_use_yes1_no0``. See
:attr:`VTE_INPUT_COLS` docstring for more details.
"""

VTE_OUTPUT_COLS = ["thrombosis_present_yes1_no0"]
"Target column for VTE data. 1 if venous thromboembolism is present, 0 if not."

VTE_DATA_PREP_PATH = (
    "/".join(
        [TOP_LEVEL_PACKAGE_PATH, "data", "files", "prep",
         "vte_onlydata_preprocessed.csv"]
    )
)
"""Path to the [originally] preprocessed VTE data.

See the ``README.rst`` in ``mtml/data/files`` for more details.
"""