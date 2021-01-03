#!/bin/bash
# test slurm batch job using dask_jobqueue SLURMCluster, four workers.
#
# note that the test_jobqueue_4.json file holds the configuration for the number
# of processes and memory that are needed for each worker; this is submitted
# with sbatch for each worker. this script only needs minimal memory.

#SBATCH --job-name=djh458:test_jobqueue_4
#SBATCH --output=test_jobqueue_4.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100M
#SBATCH --time=00:00:30

# joblib Parallel verbosity
VERBOSITY=1

# activate venv, run, deactivate. will use dask_jobqueue SLURMCluster with
# specified number of processes and cores given in test_jobqueue_4.json while
# we pass --njobs=4 to spawn 4 dask workers
source ~/djh458/bin/activate
python3 ~/mcdevitt-trauma-ml/slurm/phetdam/test_job.py --njobs=4 \
    --jobqueue-config=./test_jobqueue_4.json --verbose=$VERBOSITY
deactivate