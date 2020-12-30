__doc__ = """Runs :func:`mtml.modeling.vte.decomposition.whitened_kernel_pca`.

Takes in optional command-line arguments for metric, cross-validation folds,
number of jobs (processes) to use, and the level of GridSearchCV verbosity. All
parameters are passed in externally from bash script submitted to Slurm.
"""

import argparse
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


if __name__ == "__main__":
    # argument parser
    arp = argparse.ArgumentParser(
        description = __doc__,
        formatter_class = argparse.RawDescriptionHelpFormatter
    )
    # add all optional arguments
    arp.add_argument(
        "-m", "--metric", default = "f1_score",
        help = "ScoringKernelPCA metric to use during grid search"
    )
    arp.add_argument(
        "-c", "--cv-folds", default = 3, type = int,
        help = "Number of cross-validation folds to use during grid search"
    )
    arp.add_argument(
        "-n", "--njobs", default = 1, type = int,
        help = "Number of processes for joblib to use during multiprocessing"
    )
    arp.add_argument(
        "-v", "--verbose", nargs = "?", default = 1, const = 1, type = int,
        help = "Level of GridSearchCV verbosity"
    )
    # parse arguments
    args = arp.parse_args()
    # persist ScoringKernelPCA hyperparameters. need safe = True to make params
    # JSON safe (embedded LogisticRegression model is in there).
    persist_json_args = dict(
        target = VTE_RESULTS_DIR + "/vte_whitened_kernel_pca_params.json",
        out_transform = lambda x: (x["pcas"][0].get_params(safe = True),
                                   x["pcas"][1].get_params(safe = True))
    )
    # persist ScoringKernelPCA models themselves
    persist_pickle_args = dict(
        target = VTE_RESULTS_DIR + "/vte_whitened_kernel_pcas.pickle",
        out_transform = lambda x: x["pcas"]
    )
    # args_1 for full data, args_2 for reduced data; truncated CV results
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
    _ = task(
        report = True, random_seed = 7, metric = args.metric,
        cv = args.cv_folds, n_jobs = args.njobs, verbosity = args.verbose
    )