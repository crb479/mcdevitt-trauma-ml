#!/usr/bin/bash
# a script to monitor the memory usage of an sbatch job.

USAGE="usage: source $0 logfile [resolution] &

Reports memory use of job running on Slurm submitted using sbatch from within
the sbatch script every resolution seconds. This script should be source'd
inside an sbatch script and run in the background. If resolution is not
provided, then the default resolution used is 60 seconds.

If file logfile doesn't exist, it will be created, but if it does exist, it will
be overwritten. You must have write permissions for logfile.

optional arguments:
 -h, --help  show this usage
"

# monitor memory usage of running script
monitor() {
    # directory containing memory information
    SLURM_MEMINFO_DIR=/sys/fs/cgroup/memory/slurm/uid_$UID/job_$SLURM_JOB_ID
    # write periodically to file
    {
        while true
        do
            # get memory usage in bytes and convert to G
            MEM_USAGE=$(cat $SLURM_MEMINFO_DIR/memory.usage_in_bytes)
            MEM_USAGE=$(echo "scale=2; $MEM_USAGE / (1024 ^ 3)" | bc -l)
            # print date and memory usage in G
            echo $(date -- mem="$MEM_USAGE"G)
            # sleep for specified resolution
            sleep $2
        done
    } > $1 2>&1
}

# if no args or too many, print error
if [[ $# == 0 ]]
then
    echo "$0: error: missing required file path" 1>&2
# use default resolution of 60 seconds
elif [[ $# == 1 ]]
then
    if [ $1 = "-h"] || [ $1 = "--help" ]
    then
        # double quote needed to preserve spacing
        echo "$USAGE"
    else
        monitor $1 60
    fi
# use specified resolution
elif [[ $# == 2 ]]
then
    monitor $1 $2
else
    echo "$0: too many arguments. try $0 --help for usage" 1>&2
fi

# clean up if source'd
unset USAGE