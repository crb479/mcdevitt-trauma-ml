__doc__ = "Runs :func:`mtml.modeling.vte.decomposition.whitened_pca`."

import os.path

# pylint: disable=import-error,relative-beyond-top-level
from mtml.modeling.vte.decomposition import whitened_pca
from mtml.utils.path import find_results_home_ascending
from mtml.utils.persist import persist_json, persist_pickle, remove_all_persist


# attempt to find the results top-level directory
results_home = find_results_home_ascending(os.path.abspath("."))

# persistence decorator args
persist_json_args = dict(
    target = results_home + "/djh458/vte_whitened_pca_params.json",
    out_transform = lambda x: (x["pcas"][0].get_params(),
                               x["pcas"][1].get_params())
)
persist_pickle_args = dict(
    target = results_home + "/djh458/vte_whitened_pcas.pickle",
    out_transform = lambda x: x["pcas"]
)
# add persistence decorators and run
task = persist_json(**persist_json_args)(
    persist_pickle(**persist_pickle_args)(
        remove_all_persist(whitened_pca)
    )
)
_ = task(report = True, random_seed = 7)