.. README.rst for NYU HPC Slurm batch scripts. by Derek Huang

Slurm batch scripts for NYU HPC
===============================

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

It's important to pull frequently and then ``cd`` back to the base directory to
re-install your local copy of ``mtml`` in case you are using any of that code.

Submitting to Slurm
~~~~~~~~~~~~~~~~~~~

The NYU HPC guide for submitting jobs to Slurm on Greene, which is re-linked
`here`__, walks you through basic use of ``sbatch`` and ``srun`` with Slurm.
The `Slurm cheatsheet`__ may also be helpful while the `online documentation`__
provides more details. One can also do ``man sbatch`` and so on for plaintext
help on a command.

Writing to ``../results``
~~~~~~~~~~~~~~~~~~~~~~~~~

Assuming that your directory for your Slurm ``bash`` scripts is located in the
format described at the beginning of this ``README.rst``, a way to easily get
access in Python to the ``results`` directory located next to the ``slurm``
directory is to use ``mtml.utils.path.find_results_home_ascending``, which
crawls up the directory tree from a given directory to find the absolute path
to that ``results`` directory. Sample usage is show below, where it is assumed
that the code is in a standalone Python script located in the directory
``../slurm/phetdam`` along with a Slurm ``bash`` script will invoke the script
with the Python interpreter.

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

.. __: https://docs.python.org/3/tutorial/venv.html

.. __: https://github.com/crb479/mcdevitt-trauma-ml

.. __: https://sites.google.com/a/nyu.edu/nyu-hpc/documentation/greene

.. __: https://slurm.schedmd.com/pdfs/summary.pdf

.. __: https://slurm.schedmd.com/documentation.html

.. [#] Rather, the *only* version of Python 3 available on Greene.

.. [#] The official guide says to do ``pip install`` with the ``--user`` flag,
   but that still doesn't prevent the issue of having dependency conflicts. Just
   use a virtual environment to isolate your dependencies.

.. [#] When you first login, you are typically dumped in
   ``/home/[your_usename]``.