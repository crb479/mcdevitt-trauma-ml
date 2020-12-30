__doc__ = "Runs :func:`mtml.modeling.vte.decomposition.whitened_kernel_pca`."

import os.path

# pylint: disable=import-error,relative-beyond-top-level
from mtml.modeling.vte.decomposition import whitened_kernel_pca
from mtml.utils.path import find_results_home_ascending
from mtml.utils.persist import (
    persist_csv, persist_json, persist_pickle, remove_all_persist
)

# attempt to find the results top-level directory
RESULTS_HOME = find_results_home_ascending(".")
# my directory for VTE results
VTE_RESULTS_DIR = RESULTS_HOME + "/phetdam/vte"

# persistence decorator args. need safe = True to make params JSON safe
persist_json_args = dict(
    target = VTE_RESULTS_DIR + "/vte_whitened_kernel_pca_params.json",
    out_transform = lambda x: (x["pcas"][0].get_params(safe = True),
                               x["pcas"][1].get_params(safe = True))
)
persist_pickle_args = dict(
    target = VTE_RESULTS_DIR + "/vte_whitened_kernel_pcas.pickle",
    out_transform = lambda x: x["pcas"]
)
# args_1 for full data, args_2 for reduced data
persist_csv_args_1 = dict(
    target = VTE_RESULTS_DIR + "/vte_whitened_kernel_pca_cv_full.csv",
    out_transform = lambda x: x["cv_results"][0]
)
persist_csv_args_2 = dict(
    target = VTE_RESULTS_DIR + "/vte_whitened_kernel_pca_cv_red.csv",
    out_transform = lambda x: x["cv_results"][1]
)
# add persistence decorators and run. removes effect of originally applied
# persistence decorators (those are for local-only runs)
task = persist_csv(**persist_csv_args_1)(
    persist_csv(**persist_csv_args_2)(
        persist_json(**persist_json_args)(
            persist_pickle(**persist_pickle_args)(
                remove_all_persist(whitened_kernel_pca)
            )
        )
    )
)
_ = task(report = True, random_seed = 7)