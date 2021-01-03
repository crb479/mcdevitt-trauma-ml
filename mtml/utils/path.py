__doc__ = "Path-related utilities."

import os
import os.path
import pwd


def find_results_home_ascending(cur_dir = ".", dirname = "results"):
    """From ``startpath``, ascend directory tree to find results home directory.

    .. warning::

       Only tested to work correctly on POSIX-like systems. Not sure what will
       happen on Windows, where the path separator is reversed.

    Typical use case is when executing a script on NYU HPC from the ``slurm``
    directory. There's no way to start up a package and somehow precompute the
    location of the ``results`` directory, but we can search up the directory
    tree until we find it given a starting path. If we fail,
    :class:`FileNotFoundError` is raised.

    :param cur_dir: Starting path. If not given, then the current directory
        where the function is executed is used (default).
    :type cur_dir: str, optional
    :param dirname: Name of the results home directory. This should not need to
        be changed at all.
    :type dirname: str, optional
    :raises FileNotFoundError: If the results home directory can't be found by
        traversing up the directory tree.
    :returns: Absolute path to the results home directory.
    :rtype: str
    """
    # change cur_dir to absolute path
    cur_dir = os.path.abspath(".")
    # if cur_dir doesn't exist, raise error
    if not os.path.exists(cur_dir):
        raise FileNotFoundError("cur_dir does not exist")
    # must be a directory
    if not os.path.isdir(cur_dir):
        raise ValueError("cur_dir must be a directory path")
    # while we haven't reached the root directory yet (path is empty)
    while cur_dir != "":
        # get list of files in this directory
        files = os.listdir(cur_dir)
        # if our desired directory name is in files and it is a directory, then
        # add "/" + dirname to cur_dir and break
        if dirname in files and os.path.isdir(f"{cur_dir}/{dirname}"):
            cur_dir = f"{cur_dir}/{dirname}"
            break
        # else, we drop one more directory level
        cur_dir = "/".join(cur_dir.split("/")[:-1])
    # if cur_dir is empty, we failed, so raise error
    if cur_dir == "":
        raise FileNotFoundError(
            f"couldn't find dir {dirname} by walking up the directory tree"
        )
    # else return, we are done
    return cur_dir


def get_scratch_dir():
    """Returns the path to your ``/scratch`` directory on Greene.

    More precisely, it uses ``pwd.getpwuid(os.getuid())[0]`` to retrieve your
    login name given your user ID and then prepends ``/scratch/`` to the value.
    This method of login name retrieval is available only on Unix-like systems.

    :rtype: str
    """
    return f"/scratch/{pwd.getpwuid(os.getuid())[0]}"