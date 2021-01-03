__doc__ = """Test script for debugging what's happening on Slurm.

Runs scalar square root on an array, where the square root function used has an
adjustable delay to simulate an expensive computation.
"""

# pylint: disable=import-error
import argparse
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from joblib import delayed, parallel_backend, Parallel
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
        "-b", "--backend", default = "loky",
        help = "Backend joblib should use to manage multiprocessing"
    )
    arp.add_argument(
        "-n", "--njobs", default = 1, type = int,
        help = "Number of processes for joblib to use during multiprocessing"
    )
    arp.add_argument(
        "-j", "--jobqueue-cluster", action = "store_true",
        help = ("Pass to use a SLURMCluster set up by dask_jobqueue. "
                "Parameters are hardcoded. --backend=dask required else this "
                "argument will simply be ignored.")
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
    # print number of CPUS this process was given
    print(f"total cores granted: {len(os.sched_getaffinity(0))}")
    # print main process's hostname:PID
    print(f"parent PID: {platform.node()}:{os.getpid()}")
    # if backend == "dask"
    if args.backend == "dask":
        # if jobqueue_cluster is True, make SLURMCluster cluster
        if args.jobqueue_cluster:
            # single job, 10 processes, 10 cores (hardcoded). use infiniband
            # for faster IPC. local_directory is user's scratch directory
            cluster = SLURMCluster(
                cores = 10,
                memory = "500M",
                processes = 10,
                interface = "ib0",
                local_directory = f"/scratch/{pwd.getpwuid(os.getuid())[0]}",
                shebang = "#!/usr/bin/bash",
                walltime = "00:00:30"
            )
            # only one job on one node
            cluster.scale(jobs = 1)
            # print the job script that will be generated
            print(cluster.job_script())
        # else set cluster to None
        else:
            cluster = None
        # setup dask Client
        client = Client(cluster)
    # with the given backend and a particular parallel instance
    with parallel_backend(args.backend, n_jobs = args.njobs):
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
    res_pairs = [f"{pid},{max_rss}K" for pid, max_rss in res_dict.items()]
    # for each unique PID, get max max_rss (these can differ based on the time
    # of reporting, so we need to take their max to get unique max_rss)
    # print unique PIDs
    print(f"unique PIDs + max memory usage (K):\n{res_pairs}")