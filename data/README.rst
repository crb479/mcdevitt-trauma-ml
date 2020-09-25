.. README for temp data folder

data
====

Contains CSV data for patient model development. None of these files are too
big (all less than 50 MB) and so do not need to be uploaded via Git LFS.


Raw data
--------

Raw data from the `Google Drive for SF hospital trauma data`__, in the ``raw``
directory. Unless there is some compelling reason, use the preprocessed data
instead for programmatic applications.

AUC_withNA.csv
   Shape ``(65, 5)``. Gives univariate AUC for SF hospital trauma data for
   mortality, multiple organ failure, as well as corresponding :math:`R^2`
   values for each. Columns are ``biomarker``, ``mortality_auc``, ``mof_auc``,
   ``mortality_r2``, ``mof_r2``.

AUC_withoutNA.csv
   Shape ``(64, 4)``. Gives univariate AUC for SF hospital trauma data for
   mortality and multiple organ failure. AUCs are sorted in descending order, as
   there are two different biomarker columns for each AUC type. Columns are
   ``biomarker``, ``mortality_auc``, ``biomarker`` [#]_, and ``mof_auc``.

sf_trauma_data_num_raw.csv
   Shape ``(1494, 90)``. The raw numerical SF hospital patient trauma data,
   corresponding to the ``numerical variables- T-0`` sheet in the ``SF Hospital
   trauma data-v2.xlsx`` located on the Google Drive, with the top row removed.
   Columns include ``male``, ``age``, ``hr0_D-Dimer``, etc. Note that
   pre-existing disease/morbidity columns are actually repeated [#]__, so for
   example, the columns ``immuno-suppression`` and ``immuno-suppression.1`` will
   show up if the file is loaded with ``pandas.read_csv`` [#]__. Note that the
   ``esrddialysis`` column has been renamed ``esrd dialysis`` to ensure
   consistency with the rest of the duplicated pre-existing disease/morbidity
   columns.

.. [#] ``pandas`` will change this column to ``biomarker.1`` so the columns
   are all uniquely named.

.. [#] These columns are actually equal. Checked using ``equals`` method.

.. [#] Except for the ``hepatic failure`` and ``immuno-suppression`` columns.
   The ``hepatic failure`` column has indices ``477, 478`` with value ``-1``
   while in the ``hepatic failure.1`` column ``np.nan`` is the value. The
   ``immuno-suppression`` column has indices ``465, 467, 469, 477, 478`` with
   value ``-1`` while in the ``immuno-suppression.1`` column ``np.nan`` is the
   value.

.. __ : https://drive.google.com/drive/folders/1VyFHmTdDq-yMMvj_CPfEcV60Jvb70-
   RL?usp=sharing


Preprocessed data
-----------------

The raw data above, but preprocessed to make programmatic work easier, in the
``prep`` directory. These may not be directly usable in any models but should
relieve users of some annoying data cleaning.

mof_AUC_withNA.csv
   Shape ``(65, 2)``. The multiple organ failure univariate AUCs from
   ``AUC_withNA.csv``, sorted by AUC in descending order.

mort_AUC_withNA.csv
   Shape ``(65, 2)``. The mortality univariate AUCs from ``AUC_withoutNA.csv``,
   sorted by AUC in descending order.

mof_AUC_withoutNA.csv
   Shape ``(65, 2)``. The multiple organ failure univariate AUCs from
   ``AUC_withoutNA.csv``, sorted by AUC in descending order.

mort_AUC_withoutNA.csv
   Shape ``(65, 2)``. The mortality univariate AUCs from ``AUC_withoutNA.csv``,
   sorted by AUC in descending order.

sf_trauma_data_num.csv
   Shape ``(1494, 76)``. The numerical SF hospital patient trauma data in
   ``data/sf_trauma_data_num_raw.csv`` with the last 14 duplicate columns
   and trauma scorer columns (``iss``, ``APACHE-2``, etc.) removed, ``"Yes"``
   and ``"No"`` changed to ``1`` and ``0`` respectively. The minimal
   preprocessing is intentional.