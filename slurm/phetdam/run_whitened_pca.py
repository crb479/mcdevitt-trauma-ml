__doc__ = "Runs :func:`mtml.modeling.vte.decomposition.whitened_pca`."

# pylint: disable=import-error
from dask.distributed import Client, LocalCluster
from dask_jobqueue import SLURMCluster
import io
import os
import os.path
import platform
import resource

# pylint: disable=import-error,relative-beyond-top-level
from mtml.modeling.vte.decomposition import whitened_pca
from mtml.utils.functools import return_eval_record
from mtml.utils.path import find_results_home_ascending, get_scratch_dir
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
    # SLURMCluster started with a single job
    cluster = SLURMCluster(
        cores = 1,
        memory = "200M",
        processes = 1,
        local_directory = get_scratch_dir(),
        shebang = "#!/usr/bin/bash",
        walltime = "00:00:10"
    )
    cluster.scale(jobs = 1)
    # initialize client with SLURMCluster
    client = Client(cluster)
    # submit decorated task to client and receive future. return_eval_record
    # return an EvaluationRecord containing args, kwargs, and result.
    fut = client.submit(
        return_eval_record(task), report = True, stream = io.StringIO(),
        random_seed = 7
    )
    # once finished, print report from stream and get and print the
    # # node-qualified PID + max RSS of this process (master)
    while not fut.done():
        pass
    print(fut.result().kwargs["stream"].getvalue())
    node, pid = platform.node(), os.getpid()
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"master host:PID,max_rss = {node}:{pid},{max_rss}K")