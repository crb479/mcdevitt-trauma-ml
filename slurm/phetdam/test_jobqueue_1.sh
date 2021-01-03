#!/bin/bash
# test slurm batch job using dask_jobqueue SLURMCluster, single worker only.
#
# note that the test_jobqueue_1.json file holds the configuration for the number
# of processes and memory that are needed for the single worker; this is
# submitted with sbatch. we only need a minimal amount of memory for this job
# that acts as the master for the worker job.

#SBATCH --job-name=djh458:test_jobqueue_1
#SBATCH --output=test_jobqueue_1.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100M
#SBATCH --time=00:00:30

# joblib Parallel verbosity
VERBOSITY=1

# activate venv, run, deactivate. will use dask_jobqueue SLURMCluster with
# specified number of processes and cores given in test_jobqueue_1.json while
# we pass --njobs=1 in order to spawn only a single dask worker.
source ~/djh458/bin/activate
python3 ~/mcdevitt-trauma-ml/slurm/phetdam/test_jobqueue.py --njobs=1 \
    --jobqueue-config=./test_jobqueue_1.json --verbose=$VERBOSITY
deactivate