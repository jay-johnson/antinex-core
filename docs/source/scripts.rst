=======
Scripts
=======

Standalone Processing Examples
==============================

Using Scaler Train and Test Helper
----------------------------------

Train a DNN using the Scaler-Normalized AntiNex Django Dataset. This builds the train and test datasets using the `antinex_utils.build_scaler_train_and_test_datasets.py`_ method from the internal modules.

.. _antinex_utils.build_scaler_train_and_test_datasets.py: https://github.com/jay-johnson/antinex-utils/blob/master/antinex_utils/build_scaler_train_and_test_datasets.py

.. automodule:: antinex_core.scripts.antinex_scaler_django
   :members: build_model,run_antinex_scaler_normalization_on_django_dataset

Using Manual Scaler Objects
---------------------------

Train a DNN using the Scaler-Normalized AntiNex Django Dataset. This builds the train and test datasets manually to verify the process before editing the `antinex_utils.build_scaler_dataset_from_records.py`_ method.

.. _antinex_utils.build_scaler_dataset_from_records.py: https://github.com/jay-johnson/antinex-utils/blob/master/antinex_utils/build_scaler_dataset_from_records.py

.. automodule:: antinex_core.scripts.standalone_scaler_django
   :members: build_model,build_a_scaler_dataset_and_train_a_dnn

Convert Bottom Rows from a CSV File into JSON
=============================================

When testing live DNN predictions you can use this utility script to print a few JSON-ready dictionaries out to ``stdout``. 

Usage:

::

    convert_bottom_rows_to_json.py -f <CSV File> -b <Optional - number of rows from the bottom>

.. automodule:: antinex_core.scripts.convert_bottom_rows_to_json
   :members: convert_bottom_rows_to_json
