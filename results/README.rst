.. README.rst for results directory

Storing modeling results
========================

If you have modeling results you would like to share, please make a directory
whose name is your GitHub username and put all your results in that directory.
For example, if your username is ``phetdam``, store your results in a directory
``phetdam`` in this directory.

In general, try to version control only plain text files. CSV files are an
exception and should not be version controlled unless they are relatively small,
as one shouldn't be storing data files here. Don't version control binary files,
although reasonably-sized images are ok to show as results.

Note that ``git`` does not track empty directories by default so you may have to
use ``touch`` to create an empty file there, for example a ``.gitignore`` file.
A blank ``.gitignore`` file won't do anything and will just sit there.