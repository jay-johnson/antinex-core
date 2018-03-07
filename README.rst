AntiNex Core
============

A Celery worker pool for processing training and prediction requests for deep neural networks to detect network exploits using Keras and Tensorflow in near real-time. Internally each worker manages a buffer of pre-trained models identified by the ``label`` from the initial training request. Once trained, a model can be used for rapid prediction testing provided the same ``label`` name is used on the prediction request. Models can also be re-trained by using the training api with the same ``label``.

This repository is a standalone training and prediction worker pool that is decoupled from the AntiNex REST API:

https://github.com/jay-johnson/train-ai-with-django-swagger-jwt

Install
-------

pip install antinex-core

Run
---

Please make sure redis is running and accessible before starting the core:

::

    redis-cli 
    127.0.0.1:6379>

With redis running and the antinex-core pip installed in the python 3 runtime, use this command to start the core:

::

    ./run-antinex-core.sh

Or with celery:

::

    celery worker -A antinex_core.antinex_worker -l DEBUG

Publish a Predict Request
-------------------------

::

    ./publish_predict_request.py

Publish a Train Request
-----------------------

::

    ./publish_train_request.py

Publish a Regression Prediction Request
---------------------------------------

::

    ./publish_regression_predict.py

JSON API
--------

The AntiNex core manages a pool of workers that are subscribed to process tasks found in two queues (``webapp.train.requests`` and ``webapp.predict.requests``). Tasks are defined as JSON dictionaries and must have the following structure:

::

    {
        "label": "Django-AntiNex-Simple-DNN",
        "dataset": "./tests/datasets/classification/cleaned_attack_scans.csv",
        "ml_type": "classification",
        "predict_feature": "label_value",
        "features_to_process": [
            "eth_type",
            "idx",
            "ip_ihl",
            "ip_len",
            "ip_tos",
            "ip_version",
            "tcp_dport",
            "tcp_fields_options.MSS",
            "tcp_fields_options.Timestamp",
            "tcp_fields_options.WScale",
            "tcp_seq",
            "tcp_sport"
        ],
        "ignore_features": [
        ],
        "sort_values": [
        ],
        "seed": 42,
        "test_size": 0.2,
        "batch_size": 32,
        "epochs": 5,
        "num_splits": 2,
        "loss": "binary_crossentropy",
        "optimizer": "adam",
        "metrics": [
            "accuracy"
        ],
        "histories": [
            "val_loss",
            "val_acc",
            "loss",
            "acc"
        ],
        "model_desc": {
            "layers": [
                {
                    "num_neurons": 250,
                    "init": "uniform",
                    "activation": "relu"
                },
                {
                    "num_neurons": 200,
                    "init": "uniform",
                    "activation": "relu"
                },
                {
                    "num_neurons": 150,
                    "init": "uniform",
                    "activation": "relu"
                },
                {
                    "num_neurons": 100,
                    "init": "uniform",
                    "activation": "relu"
                },
                {
                    "num_neurons": 50,
                    "init": "uniform",
                    "activation": "relu"
                },
                {
                    "num_neurons": 1,
                    "init": "uniform",
                    "activation": "sigmoid"
                }
            ]
        },
        "label_rules": {
            "labels": [
                "not_attack",
                "attack"
            ],
            "label_values": [
                0,
                1
            ]
        },
        "version": 1
    }


Regression prediction tasks are also supported, and here is an example from an included dataset with mock stock prices:

::

    {
        "label": "Close-Regression",
        "dataset": "./tests/datasets/regression/stock.csv",
        "ml_type": "regression",
        "predict_feature": "close",
        "features_to_process": [
            "high",
            "low",
            "open",
            "volume"
        ],
        "ignore_features": [
        ],
        "sort_values": [
        ],
        "seed": 7,
        "test_size": 0.2,
        "batch_size": 32,
        "epochs": 5,
        "num_splits": 2,
        "loss": "mse",
        "optimizer": "adam",
        "metrics": [
            "mse",
            "mae",
            "mape",
            "cosine"
        ],
        "model_desc": {
            "layers": [
                {
                    "activation": "relu",
                    "init": "uniform",
                    "num_neurons": 12
                },
                {
                    "activation": "relu",
                    "init": "uniform",
                    "num_neurons": 6
                },
                {
                    "activation": null,
                    "init": "uniform",
                    "num_neurons": 1
                }
            ]
        }
    }
   


Development
-----------
::

    virtualenv -p python3 ~/.venvs/antinexcore && source ~/.venvs/antinexcore/bin/activate && pip install -e .

Testing
-------

Run all

::

    python setup.py test

Run a test case

::

    python -m unittest tests.test_train.TestTrain.test_train_antinex_simple_success_retrain

Linting
-------

flake8 .

pycodestyle --exclude=.tox,.eggs

License
-------

Apache 2.0 - Please refer to the LICENSE_ for more details

.. _License: https://github.com/jay-johnson/antinex-core/blob/master/LICENSE
