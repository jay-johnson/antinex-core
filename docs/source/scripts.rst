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

S3 Testing
----------

Run this script to verify S3 is working.

Set Environment Variables
=========================

Set these as needed for your S3 deployment

::

    export S3_ACCESS_KEY=<access key>
    export S3_SECRET_KEY=<secret key>
    export S3_REGION_NAME=<region name: us-east-1>
    export S3_ADDRESS=<S3 endpoint address host:port like: minio-service:9000>
    export S3_UPLOAD_FILE=<path to file to upload>
    export S3_BUCKET=<bucket name - s3-verification-tests default>
    export S3_BUCKET_KEY=<bucket key name - s3-worked-on-%Y-%m-%d-%H-%M-%S default>
    export S3_SECURE=<use ssl '1', disable with '0' which is the default>

Run S3 Verification Test
------------------------

Run the included S3 verification script:

::

    run-s3-test.py

.. automodule:: antinex_core.scripts.run_s3_test
   :members: run_s3_test
