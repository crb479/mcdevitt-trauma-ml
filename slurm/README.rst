.. README.rst for NYU HPC Slurm batch scripts. by Derek Huang

Slurm batch scripts for NYU HPC
===============================

   Important:

   Read this file in its entirety!

On NYU HPC clusters, one may only access the computing resources there by
submitting jobs to `Slurm`__ through ``sbatch`` or ``srun`` [#]_. Please place
any ``bash`` scripts to be submitted to Slurm along with supporting code here in
a directory named after your GitHub username. For example, if your username is
``phetdam``, you should put your ``bash`` scripts and other code for Slurm in
the directory ``phetdam`` in this directory.

Read on for quickstart instructions on using the package code with HPC. The
guide asssumes that you already know how to ``ssh`` into your HPC account
through VPN or ``gw.hpc.nyu.edu`` and into a valid cluster (Greene).

Some info about the Greene cluster can be found `here`__.

.. __: https://slurm.schedmd.com/documentation.html

.. __: https://sites.google.com/a/nyu.edu/nyu-hpc/systems/greene-cluster

.. [#] Unless you need an interactive session, you should always use ``sbatch``.


Python quickstart
-----------------

   Note:

   I will *not* be using ``conda`` for simplicity reasons. ``venv`` is enough.

One-time virtual environment setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first thing to do is load your desired Python 3 environment, which must have
a version that is >=3.6. At the time of this writing (2020-12-29), the latest
version [#]_ of Python available on Greene is 3.8.6, and one can load it with

.. code:: bash

   module load python3/intel/3.8.6

To prevent dependency conflicts, create a `virtual environment`__ [#]_; it's
enough to just use the built-in ``venv`` module unless you have specific needs.
To create a virtual environment named ``venv`` using ``venv`` in your current
directory [#]_ and then activate it, you can do something like

.. code:: bash

   python3 -m venv venv && source venv/bin/activate

Here ``venv`` is the directory where the virtual environment will be created and
where ``pip`` will install dependencies after activation. For subsequent logins,
don't use ``module`` and just ``source <your_venv_dir>/bin/activate`` to start
using your virtual environment.

.. __: https://docs.python.org/3/tutorial/venv.html

.. [#] Rather, the *only* version of Python 3 available on Greene.

.. [#] The official guide says to do ``pip install`` with the ``--user`` flag,
   but that still doesn't prevent the issue of having dependency conflicts. Just
   use a virtual environment to isolate your dependencies.

.. [#] When you first login, you are typically dumped in
   ``/home/[your_username]``.

Developing with ``mtml``
~~~~~~~~~~~~~~~~~~~~~~~~

To use the code from the `GitHub repo`__, just use ``git clone``, and assuming
we are downloading using HTTPS, you can do

.. code:: bash

   git clone https://github.com/crb479/mcdevitt-trauma-ml.git

To install the package, ``cd`` into the repository directory and simply run

.. code:: bash

   pip3 install .

``pip`` will read from the ``setup.cfg`` file and install the package and all
stated dependencies into the virtual environment directory, i.e. ``venv``. If
the repo is already in your home directory, no need to ``git clone``. Just pull
the latest code from the repo, ``cd`` into it, and then run ``pip3 install .``
to get the latest version of the code installed into the virtual environment.

Pull *frequently* and then ``cd`` back to the base directory to re-install your
local copy of ``mtml`` if your are using any of that code.

.. __: https://github.com/crb479/mcdevitt-trauma-ml

Submitting to Slurm
~~~~~~~~~~~~~~~~~~~

The NYU HPC guide for submitting jobs to Slurm on Greene, which is re-linked
`here`__, walks you through basic use of ``sbatch`` and ``srun`` with Slurm.
The `Slurm cheatsheet`__ [#]_ may also be helpful while the
`online documentation`__ provides more details. One can also do ``man sbatch``
and so on for plaintext help on a command. Also read the
`Slurm best practices`__ since resources are shared and it is in everyone's
interest to make use of compute time and memory as efficiently as possible.

There are many Slurm commands, but the three essentials that you must know how
to use are ``sbatch``, ``sacct``, and ``seff``. ``sbatch`` is for submitting
your batch scripts to the scheduler, ``sacct`` gives you details on your
submitted/completed jobs, and ``seff``, when given a job ID, yields time, CPU,
and memory usage statistics which are helpful for paring down the amount of
resources requested (avoid oversubscription).

.. __: https://sites.google.com/a/nyu.edu/nyu-hpc/documentation/greene

.. __: https://slurm.schedmd.com/pdfs/summary.pdf

.. __: https://slurm.schedmd.com/documentation.html

.. __: https://sites.google.com/a/nyu.edu/nyu-hpc/documentation/prince/batch/
   slurm-best-practices

.. [#] This might be a little out of date. For example, there are many
   environment variables listed in the `Slurm documentation for sbatch`__ that
   are not mentioned in the sheet.

.. __: https://slurm.schedmd.com/sbatch.html

Writing to ``../results``
~~~~~~~~~~~~~~~~~~~~~~~~~

Assuming that your directory for your Slurm ``bash`` scripts is located in the
format described at the beginning of this ``README.rst``, a way to easily get
access in Python to the ``results`` directory located next to the ``slurm``
directory is to use the ``mtml.utils.path`` function
``find_results_home_ascending``, which crawls up the directory tree from a given
directory to find the absolute path to that ``results`` directory. Sample usage
is show below, where it is assumed that the code is in a standalone Python
script located in the directory ``../slurm/phetdam`` along with a Slurm ``bash``
script will invoke the script with the Python interpreter.

.. code:: python3

   import json
   import numpy as np
   # mtml needs to be installed
   from mtml.utils.path import find_results_home_ascending

   # starts ascending from directory that the file is located in
   RESULTS_HOME = find_results_home_ascending(".")
   # convert to list since JSON encoder doesn't understand what ndarray is
   vals = list(np.log([i + 1 for i in range(20)]))
   # dump in ../results/phetdam/np_log_vals.json
   with open(RESULTS_HOME + "/phetdam/np_log_vals.json", "w") as f:
      json.dump(vals, f)

Distributing computation with ``dask_jobqueue`` and ``joblib``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `dask_jobqueue`__ package greatly simplifies the task of distributing
computations. It is part of the dependencies listeed in the ``setup.cfg`` file
so it should be automatically installed when ``pip3 install .`` is run in the
top-level repository directory. ``dask_jobqueue`` offers subclasses of the
`distributed.deploy.spec.SpecCluster`__, specifically the
`dask.jobqueue.slurm.SLURMCluster`__, that facilitates the interaction between
your user code, the ``distributed`` scheduler, and NYU HPC's Slurm resource
manager.

Although typical usage is to initialize a `distributed.client.Client`__ and then
use its ``map`` and ``submit`` methods to send tasks to be scheduled, it's also
possible to use the ``distributed`` scheduler with ``joblib``, which is what
typically backs scikit-learn estimators that accept the ``n_jobs`` parameter.
The ``distributed`` scheduler and ``Client`` can also be used with the
``SLURMCluster`` offered by ``dask_jobqueue`` and together these tools make it
easier to multiprocess on a single node or distribute computation across several
compute nodes.

Below is an example of computing square roots in a distributed fashion using
the `joblib.parallel.parallel_backend`__ context manager to pass control to the
``distributed`` ``Client``, which uses the clsuter started by the
``dask_jobqueue`` ``SLURMCluster``.

.. code::  python3

   from dask.distributed import Client
   from dask_jobqueue import SLURMCluster
   from joblib import delayed, parallel_backend, Parallel
   import math

   # initialize SLURMCluster
   cluster = SLURMCluster(
       local_directory = "/scratch/djh458", # replace with your own scratch dir
       shebang = "#!/usr/bin/bash",
       cores = 3,                           # each worker gets 3 CPU cores
       memory = "300M",                     # each worker gets 300M total memory
       processes = 3,                       # each worker starts 3 processes
       interface = "ib0",                   # infiniband gives faster IPC
       walltime = "00:00:30"                # worker walltime before death
   )
   # start 4 workers (submits 4 dask-worker jobs to Slurm)
   cluster.scale(jobs = 4)
   # connect to distributed Client
   client = Client(cluster)
   # use context manager with "dask" argument to use distributed backend
   with parallel_backend("dask"):
       # typical joblib mapping using generator expression, Parallel, delayed
       res = Parallel(verbose = 1)(
           delayed(math.sqrt)(x) for x in [i for i in range(1000)]
       )

The ``parallel_backend`` context manager can also be used to change the backend
used by ``joblib`` internally withing scikit-learn code. Below we show an
example of using the `sklearn.model_selection._search.GridSearchCV`__ estimator
together with the kernel SVM model on a hyperparameter grid, with computation
done in a distributed fashion using the ``SLURMCluster``, ``Client``, and
``parallel_backend`` context manager [#]_.

.. code:: python3

   from dask.distributed import Client
   from dask_jobqueue import SLURMCluster
   from joblib import parallel_backend
   from sklearn.datasets import load_digits
   from sklearn.svm import SVC
   from sklearn.model_selection import GridSearchCV, train_test_split
   from sklearn.preprocessing import StandardScaler

   # initialize SLURMCluster
   cluster = SLURMCluster(
       local_directory = "/scratch/djh458",
       shebang = "#!/usr/bin/bash",
       cores = 4,
       memory = "10G",
       processes = 4,
       interface = "ib0",
       walltime = "00:45:00"
   )
   # start 6 workers + connect to distributed Client
   cluster.scale(jobs = 6)
   client = Client(cluster)
   # get digits data and standardize (all features are 0-16 so not necessary)
   X, y = load_digits(return_X_y = True)
   X = StandardScaler().fit_transform(X)
   # train, test split and grid of parameters
   X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 7)
   grid = dict(
       C = [0.05, 0.1, 1, 5],
       kernel = ["linear", "rbf", "poly"],
       random_state = [7]
   )
   # initialize GridSearchCV
   search = GridSearchCV(SVC(), grid, scoring = "f1", cv = 5, verbose = 1)
   # use distributed backend with cluster started by SLURMCluster
   with parallel_backend("dask"):
       search.fit(X_train, y_train)

.. [#] Warning: this code is for illustrative purposes only and has not been
   tested on Greene.

.. __: https://jobqueue.dask.org/en/latest/

.. __: https://distributed.dask.org/en/latest/api.html#distributed.SpecCluster

.. __: https://jobqueue.dask.org/en/latest/generated/dask_jobqueue.SLURMCluster.
   html#dask_jobqueue.SLURMCluster

.. __: https://distributed.dask.org/en/latest/api.html#distributed.Client

.. __: https://joblib.readthedocs.io/en/latest/parallel.html#joblib.
   parallel_backend

.. __: https://scikit-learn.org/stable/modules/generated/sklearn.
   model_selection.GridSearchCV.html

Configuring ``dask`` and ``dask_jobqueue``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is possible to modify the default ``distributed`` scheduler behavior with
YAML configuration files. This is useful for modifying the default
`worker memory management behavior`__ for ``distributed`` or for overriding
default parameters for the special cluster classes in ``dask_jobqueue``. It is
preferable for configuration files to be located at ``~/.config/dask``. This
directory also contains two sample YAML configurations, ``jobqueue.yaml`` for
``dask_jobqueue`` and ``distributed.yaml`` for ``distributed``, which modify the
default worker memory management policies and override defaults for a couple of
named parameters to pass to the ``SLURMCluster``.\

.. __: https://distributed.dask.org/en/latest/worker.html