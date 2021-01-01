__doc__ = """Test script for debugging what's happening on Slurm.

Runs scalar square root on an array, where the square root function used has an
adjustable delay to simulate an expensive computation.
"""

# pylint: disable=import-error
import argparse
from distributed import Client
from joblib import delayed, parallel_backend, Parallel
import math
import numpy as np
import os
import platform
import time


def slow_sqrt(x, delay = 1):
    """Square root function delayed by ``delay`` seconds.

    Delay used for simulating expensive computation.

    :returns: Tuple of result and hostname-qualified PID of process executing
        the function (hostname:PID)
    :rtype: tuple
    """
    time.sleep(delay)
    # return result and hostname-qualified PID of process executing function
    return math.sqrt(x), f"{platform.node()}:{os.getpid()}"


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
        "-j", "--jobqueue-cluster", default = None,
        help = ("Valid name of a dask_jobqueue cluster class for scheduling on "
                "particular distributed HPC systems. --backend=dask required "
                "else this argument will simply be ignored.")
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
        # if jobqueue_cluster is None, make local cluster
        if args.jobqueue_cluster is None:
            client = Client()
        # else make distributed cluster using jobqueue_cluster name
        else:
            raise NotImplementedError("no distributed implementation")
    # with the given backend and a particular parallel instance
    with parallel_backend(args.backend, n_jobs = args.njobs):
        res = Parallel(verbose = args.verbose)(
            delayed(slow_sqrt)(x ** 2) for x in ar
        )
    # collect PIDs and save unique ones
    pids = np.unique([pid for _, pid in res])
    # print unique PIDs
    print(f"unique PIDs:\n{pids}")