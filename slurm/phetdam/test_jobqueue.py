__doc__ = """dask_jobqueue test script to see what's happening on Slurm.

Runs scalar square root on an array, where the square root function used has an
adjustable delay to simulate an expensive computation. Uses the dask_jobqueue
SLURMCluster to start dask workers by submitting with sbatch or LocalCluster.
"""

# pylint: disable=import-error
import argparse
from dask.distributed import Client, LocalCluster
from dask_jobqueue import SLURMCluster
from joblib import delayed, parallel_backend, Parallel
import json
import math
import numpy as np
import os
import platform
import pwd
import resource
import time


def slow_sqrt(x, delay = 1):
    """Square root function delayed by ``delay`` seconds.

    Delay used for simulating expensive computation.

    :returns: Tuple of result, hostname-qualified PID of process executing
        the function, and max resident set size in K
    :rtype: tuple
    """
    time.sleep(delay)
    # return result, hostname-qualified PID of process executing function, and
    return (
        math.sqrt(x), f"{platform.node()}:{os.getpid()}",
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    )


if __name__ == "__main__":
    # instantiate argument parser and add arguments
    arp = argparse.ArgumentParser(
        description = __doc__,
        formatter_class = argparse.RawDescriptionHelpFormatter
    )
    arp.add_argument(
        "-b", "--backend", default = "dask",
        help = ("Backend joblib should use to manage multiprocessing. Do NOT "
                "change this value (not using loky anymore)")
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
        "-j", "--jobqueue-config",
        help = ("JSON configuration file containing a dict which will be "
                "unpacked into the constructor of a SLURMCluster set up by "
                "dask_jobqueue. --backend=dask required (default). all kwargs "
                "for the SLURMCluster can be specified except for the shebang "
                "and local_directory args, which are hardcoded.")
    )
    arp.add_argument(
        "-v", "--verbose", nargs = "?", default = 1, const = 1, type = int,
        help = "Level of joblib verbosity"
    )
    # parse arguments
    args = arp.parse_args()
    # array with 100 elements
    ar = np.arange(100)
    # print number of CPUs on the node
    print(f"total node cores: {os.cpu_count()}")
    # print main process's hostname:PID
    print(f"parent PID: {platform.node()}:{os.getpid()}")
    # if backend == "dask"
    if args.backend == "dask":
        # if jobqueue_config is not None, make SLURMCluster cluster
        if args.jobqueue_config is not None:
            # attempt to open file (with context manager) to get config dict
            with open(args.jobqueue_config) as f:
                config = json.load(f)
            # local_directory is user's scratch directory and use bash for
            # shebang. read other options from JSON config dict.
            cluster = SLURMCluster(
                local_directory = f"/scratch/{pwd.getpwuid(os.getuid())[0]}",
                shebang = "#!/usr/bin/bash",
                **config
            )
            # create args.njobs workers, each with specified number of processes
            cluster.scale(jobs = args.njobs)
            # print the job script that will be generated for each worker
            print(cluster.job_script())
        # else set cluster to LocalCluster with args.njobs workers
        else:
            cluster = LocalCluster(n_workers = args.njobs)
        # setup dask Client
        client = Client(cluster)
    # else raise error; not supported
    else:
        raise NotImplementedError("only supported backend is \"dask\"")
    # with the given backend and a particular parallel instance. we could submit
    # directly to the Client but this is what is used internally in scikit-learn
    # GridSearchCV, so we want to verify that this works
    with parallel_backend(args.backend):
        res = Parallel(verbose = args.verbose)(
            delayed(slow_sqrt)(x ** 2) for x in ar
        )
    # collect PIDs + max RSS values
    res_pairs = np.array([(pid, max_rss) for _, pid, max_rss in res])
    # get unique PIDS and update max RSS values in dict if necessary
    res_dict = {}
    for pid, max_rss in res_pairs:
        if pid not in res_dict:
            res_dict[pid] = max_rss
        else:
            res_dict[pid] = max(res_dict[pid], max_rss)
    # set res_pairs to pid, max_rss in K pairs
    res_pairs = np.array(
        [f"{pid},{max_rss}K" for pid, max_rss in res_dict.items()]
    )
    # print unique PIDs (array makes the lines wrap at 80 columns)
    print(f"unique PIDs + max memory usage (K):\n{res_pairs}\n")
    # number of processes spawned
    print(f"worker processes: {len(res_pairs)}")