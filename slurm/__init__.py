__doc__ = """``__init__.py`` to allow relative imports from this file.

Having ``slurm`` act like its own package (which will not be installed by
setuptools) allows constants to be defined here and be accessed by scripts in
each contributor's directory with the ``slurm`` directory.
"""

import os
import os.path

PLAT_DELIM = "\\" if os.name == "nt" else "/"
"Directory path delimiter. Platform dependent."

RESULTS_HOME = PLAT_DELIM.join(
    os.path.dirname(os.path.abspath(__file__)).split(PLAT_DELIM)[:-1]
) + f"{PLAT_DELIM}results"
"""Home directory where all results should be written.

If current directory is ``.``, the results home directory is ``../results``,
i.e. the directory right next to ``slurm``.
"""