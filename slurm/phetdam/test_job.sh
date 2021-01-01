#!/bin/bash
# test slurm batch job to debug what's going on.
#
# the nyu configuration for slurm is very confusing

#SBATCH --job-name=djh458:test_job
#SBATCH --output=test_job.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100M
#SBATCH --time=00:00:30

# joblib verbosity
VERBOSITY=1

# activate venv, run, deactivate. pass $SLURM_CPUS_PER_TASK and $VERBOSITY to
# control number of forked processes and joblib verbosity. use dask backend
source ~/djh458/bin/activate
python3 ~/mcdevitt-trauma-ml/slurm/phetdam/test_job.py \
    --backend=dask --njobs=$SLURM_CPUS_PER_TASK --verbose=$VERBOSITY
deactivate