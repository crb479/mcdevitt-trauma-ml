__doc__ = "Runs :func:`mtml.modeling.vte.decomposition.whitened_pca`."

import os
import os.path
import platform
import resource

# pylint: disable=import-error,relative-beyond-top-level
from mtml.modeling.vte.decomposition import whitened_pca
from mtml.utils.path import find_results_home_ascending
from mtml.utils.persist import persist_json, persist_pickle

# attempt to find the results top-level directory
RESULTS_HOME = find_results_home_ascending(".")
# my directory for VTE results
VTE_RESULTS_DIR = RESULTS_HOME + "/phetdam/vte"


if __name__ == "__main__":
    # persistence decorator args
    persist_json_args = dict(
        target = VTE_RESULTS_DIR + "/vte_whitened_pca_params.json",
        out_transform = lambda x: (x["pcas"][0].get_params(),
                                   x["pcas"][1].get_params())
    )
    persist_pickle_args = dict(
        target = VTE_RESULTS_DIR + "/vte_whitened_pcas.pickle",
        out_transform = lambda x: x["pcas"]
    )
    # add persistence decorators and run
    task = persist_json(**persist_json_args)(
        persist_pickle(**persist_pickle_args)(
            whitened_pca
        )
    )
    _ = task(report = True, random_seed = 7)
    # get and print node-qualified PID + max RSS
    node, pid = platform.node(), os.getpid()
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"host:PID,max_rss = {node}:{pid},{max_rss}K")
