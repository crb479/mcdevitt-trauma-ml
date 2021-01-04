__doc__ = """Runs :func:`mtml.modeling.vte.decomposition.whitened_kernel_pca`.

Takes in optional command-line arguments for metric, cross-validation folds,
number of jobs (processes) to use, and the level of GridSearchCV verbosity. All
parameters are passed in externally from bash script submitted to Slurm.
"""

# pylint: disable=import-error
import argparse
from dask.distributed import Client, LocalCluster
from dask_jobqueue import SLURMCluster
import json
import os.path

# pylint: disable=import-error,relative-beyond-top-level
from mtml.modeling.vte.decomposition import whitened_kernel_pca
from mtml.utils.path import find_results_home_ascending, get_scratch_dir
from mtml.utils.persist import persist_csv, persist_json, persist_pickle

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
        "-b", "--backend", default = "dask",
        help = ("Backend joblib should use to manage multiprocessing. Do NOT "
                "change this value (not using loky anymore)")
    )
    arp.add_argument(
        "-j", "--jobqueue-config",
        help = ("JSON configuration file containing a dict which will be "
                "unpacked into the constructor of a SLURMCluster set up by "
                "dask_jobqueue. --backend=dask required (default). all kwargs "
                "for the SLURMCluster can be specified except for the shebang "
                "and local_directory args, which are hardcoded.")
    )
    arp.add_argument(
        "-n", "--njobs", default = 1, type = int,
        help = ("Number of processes for joblib to use during multiprocessing "
                "if --jobqueue-config is not passed a JSON config file. if "
                "--jobqueue-config does get a JSON config file, then this is "
                "the number of workers that will be started by a dask "
                "LocalCluster passed to the dask Client.")
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
    # add persistence decorators
    task = persist_csv(**persist_csv_args_1)(
        persist_csv(**persist_csv_args_2)(
            persist_json(**persist_json_args)(
                persist_pickle(**persist_pickle_args)(
                    whitened_kernel_pca
                )
            )
        )
    )
    # backend setup. if backend == "dask"
    if args.backend == "dask":
        # if jobqueue_config is not None, make SLURMCluster cluster
        if args.jobqueue_config is not None:
            # attempt to open file (with context manager) to get config dict
            with open(args.jobqueue_config) as f:
                config = json.load(f)
            # local_directory is user's scratch directory and use bash for
            # shebang. read other options from JSON config dict.
            cluster = SLURMCluster(
                local_directory = get_scratch_dir(),
                **config
            )
            # don't call scale here; scale called 
        # else set cluster to LocalCluster; scale is called inside task
        else:
            cluster = LocalCluster()
    # else raise error; not supported
    else:
        raise NotImplementedError("only supported backend is \"dask\"")
    # run task using the initialized cluster
    _ = task(
        report = True, random_seed = 7, metric = args.metric,
        cv = args.cv_folds, backend = args.backend, cluster = cluster,
        n_jobs = args.njobs, verbosity = args.verbose
    )