AntiNex Core Worker - API Reference
-----------------------------------

Splunk Environment Variables
----------------------------

This repository uses the `Spylunking <https://github.com/jay-johnson/spylunking>`__ logger that supports publishing logs to Splunk over the authenticated HEC REST API. You can set these environment variables to publish to Splunk:

::

    export SPLUNK_ADDRESS="<splunk address host:port>"
    export SPLUNK_API_ADDRESS="<splunk api address host:port>"
    export SPLUNK_USER="<splunk username for login>"
    export SPLUNK_PASSWORD="<splunk password for login>"
    export SPLUNK_TOKEN="<Optional - username and password will login or you can use a pre-existing splunk token>"
    export SPLUNK_INDEX="<splunk index>"
    export SPLUNK_QUEUE_SIZE="<num msgs allowed in queue - 0=infinite>"
    export SPLUNK_RETRY_COUNT="<attempts per log to retry publishing>"
    export SPLUNK_RETRY_BACKOFF="<cooldown in seconds per failed POST>"
    export SPLUNK_SLEEP_INTERVAL="<sleep in seconds per batch>"
    export SPLUNK_SOURCE="<splunk source>"
    export SPLUNK_SOURCETYPE="<splunk sourcetype>"
    export SPLUNK_TIMEOUT="<timeout in seconds>"
    export SPLUNK_DEBUG="<1 enable debug|0 off - very verbose logging in the Splunk Publishers>"

Celery Worker
-------------

Here is the Celery Worker's source code.

.. automodule:: antinex_core.antinex_worker
   :members: setup_celery_logging,AntiNexCore,start_antinex_core_worker

Process Consumed Messages From the Queues
=========================================

The processor class processes any messages the worker consumes from the queue.

.. automodule:: antinex_core.antinex_processor
   :members: AntiNexProcessor

Send Results to the Broker
==========================

This method is responsible for publishing what the core's results were from the processed job.

.. note:: The results must be sent back as a JSON dictionary for the REST API's Celery Workers to handle.

.. automodule:: antinex_core.send_results_to_broker
   :members: send_results_to_broker
