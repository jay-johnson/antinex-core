AntiNex Core
============

A Celery worker pool for processing training and prediction requests for deep neural networks to detect network exploits using Keras and Tensorflow in near real-time.

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

Development
-----------
::

    virtualenv -p python3 ~/.venvs/antinexcore && source ~/.venvs/antinexcore/bin/activate && pip install -e .

Linting
-------

flake8 .

pycodestyle --exclude=.tox,.eggs

License
-------

Apache 2.0 - Please refer to the LICENSE_ for more details

.. _License: https://github.com/jay-johnson/antinex-core/blob/master/LICENSE
