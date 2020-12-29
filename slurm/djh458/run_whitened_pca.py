__doc__ = "Runs :func:`mtml.modeling.vte.decomposition.whitened_pca`."

import os.path

# pylint: disable=import-error,relative-beyond-top-level
from mtml.modeling.vte.decomposition import whitened_pca
from mtml.utils.persist import persist_json, persist_pickle

from .. import RESULTS_HOME


task = persist_json(
    target = RESULTS_HOME + "/djh458/vte_whitened_pca_params.json",
    out_transform = lambda x: (x["pcas"][0].get_params(),
                               x["pcas"][1].get_params())
)(
    persist_pickle(
        target = RESULTS_HOME + "/djh458/vte_whitened_pcas.pickle",
        out_transform = lambda x: x["pcas"]
    )(whitened_pca)
)


if __name__ == "__main__":
    task(report = True, random_seed = 7)