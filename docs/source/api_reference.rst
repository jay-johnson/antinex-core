===================================
AntiNex Core Worker - API Reference
===================================

Celery Worker
=============

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
