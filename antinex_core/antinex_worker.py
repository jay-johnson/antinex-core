from celery import Celery
from celery import signals
from antinex_core.log.setup_logging import build_colorized_logger
from antinex_utils.utils import ev
from celery_connectors.subscriber import Subscriber
from antinex_core.antinex_processor import AntiNexProcessor


# Disable celery log hijacking
# https://github.com/celery/celery/issues/2509
@signals.setup_logging.connect
def setup_celery_logging(**kwargs):
    pass


name = "antinex"
log = build_colorized_logger(name=name)


class AntiNexCore:

    def __init__(
            self,
            name="",
            broker_url=ev(
                "BROKER_URL",
                "redis://localhost:6379/6"),
            train_queue_name=ev(
                "TRAIN_QUEUE",
                "webapp.train.requests"),
            predict_queue_name=ev(
                "PREDICT_QUEUE",
                "webapp.predict.requests"),
            max_msgs=100,
            max_models=100):
        """__init__

        :param name: worker name
        :param broker_url: connection string to broker
        :param train_queue_name: queue name for training requests
        :param predict_queue_name: queue name for predict requests
        :param max_msgs: num msgs to save for replay debugging (FIFO)
        :param max_models: num pre-trained models to keep in memory (FIFO)
        """

        self.name = name
        log.info(("{} - INIT")
                 .format(
                     self.name))

        self.state = "INIT"
        self.broker_url = broker_url

        # Setup queues:
        self.train_queue_name = train_queue_name
        self.predict_queue_name = predict_queue_name

        self.queues = [
            self.train_queue_name,
            self.predict_queue_name
        ]

        # Subscribers
        self.all_queues_sub = None

        # SSL Celery options dict
        self.ssl_options = {}

        # http://docs.celeryproject.org/en/latest/userguide/calling.html#calling-retry  # noqa
        # allow publishes to retry for a time
        self.task_publish_retry_policy = {
            "interval_max": 1,
            "max_retries": 120,     # None - forever
            "interval_start": 0.1,
            "interval_step": 0.2}

        # Confirm publishes with Celery
        # https://github.com/celery/kombu/issues/572
        self.transport_options = {
            "confirm_publish": True}

        self.conn_attrs = {
            "task_default_queue": "antinex.worker.control",
            "task_default_exchange": "antinex.worker.control",
            # noqa http://docs.celeryproject.org/en/latest/userguide/configuration.html#std:setting-worker_prefetch_multiplier
            "worker_prefetch_multiplier": 1,  # consume 1 message at a time
            # noqa http://docs.celeryproject.org/en/latest/userguide/configuration.html#std:setting-worker_prefetch_multiplie
            "prefetch_count": 3,  # noqa consume 1 message at a time per worker (3 workers)
            # noqa http://docs.celeryproject.org/en/latest/userguide/configuration.html#std:setting-broker_heartbeat
            "broker_heartbeat": 240,  # in seconds
            # noqa http://docs.celeryproject.org/en/latest/userguide/configuration.html#std:setting-broker_connection_max_retries
            "broker_connection_max_retries": None,  # None is forever
            # noqa http://docs.celeryproject.org/en/latest/userguide/configuration.html#std:setting-task_acks_late
            "task_acks_late": True,  # noqa on consume do not send an immediate ack back
            "task_publish_retry_policy": self.task_publish_retry_policy}

        self.processor = AntiNexProcessor(
            name="{}-prc".format(
                    self.name),
            max_msgs=max_msgs,
            max_models=max_models)
    # end of __init__

    def start(
            self,
            app,
            ssl_options=None,
            task_retry_policy=None,
            transport_options=None,
            conn_attrs=None,
            queues=None,
            callback=None):
        """start

        :param app: Celery app
        :param ssl_options: ssl dictionary
        :param task_retry_policy: retry policy
        :param transport_options: transport config dict
        :param conn_attrs: connection dict
        :param queues: name of queues to consume messages
        :param callback: callback method when a message is consumed
        """

        log.info(("{} - start")
                 .format(
                    self.name))

        use_queues = queues
        if not use_queues:
            use_queues = self.queues

        use_ssl_options = ssl_options
        if not use_ssl_options:
            use_ssl_options = self.ssl_options

        use_retry_policy = task_retry_policy
        if not use_retry_policy:
            use_retry_policy = self.task_publish_retry_policy

        use_transport_options = transport_options
        if not use_transport_options:
            use_transport_options = self.transport_options

        use_conn_attrs = conn_attrs
        if not use_conn_attrs:
            use_conn_attrs = self.conn_attrs

        use_callback = callback
        if not use_callback:
            use_callback = self.processor.handle_messages

        log.info(("{} - start - creating subscriber")
                 .format(
                    self.name))

        self.all_queues_sub = Subscriber(
            self.name,
            self.broker_url,
            app,
            use_ssl_options,
            **use_conn_attrs)

        log.info(("{} - start - activating consumer for "
                  "queues={} callback={}")
                 .format(
                    self.name,
                    use_queues,
                    use_callback.__name__))

        self.all_queues_sub.consume(
            callback=use_callback,
            queues=use_queues,
            exchange=None,
            routing_key=None,
            prefetch_count=use_conn_attrs["prefetch_count"])

        self.state = "ACTIVE"

        log.info(("{} - start - state={} done")
                 .format(
                    self.name,
                    self.state))
    # end of start

    def show_diagnostics(
            self):
        """show_diagnostics"""

        self.processor.show_diagnostics()
    # end of show_diagnostics

    def shutdown(
            self):
        """shutdown"""
        log.info(("{} - shutting down - start")
                 .format(
                    self.name))
        self.state = "SHUTDOWN"
        self.show_diagnostics()
        self.processor.shutdown()
        log.info(("{} - shutting down - done")
                 .format(
                    self.name))
    # end of shutdown

# end of AntiNexCore


log.info("loading Celery app")
app = Celery()

broker_url = ev(
    "BROKER_URL",
    "redis://localhost:6379/6")
train_queue_name = ev(
    "TRAIN_QUEUE",
    "webapp.train.requests")
predict_queue_name = ev(
    "PREDICT_QUEUE",
    "webapp.predict.requests")
max_msgs = int(ev(
    "MAX_MSGS",
    "100"))
max_models = int(ev(
    "MAX_MODELS",
    "10"))

log.info("Creating antinex core")
core = AntiNexCore(
    name="core",
    broker_url=broker_url,
    train_queue_name=train_queue_name,
    predict_queue_name=predict_queue_name,
    max_msgs=max_msgs,
    max_models=max_models)
try:
    log.info("Starting antinex core")
    core.start(
        app=app)
except Exception as e:
    log.info(("Core hit exception={} shutting down")
             .format(
                e))
    core.shutdown()
    log.info(("canceling consumer to queue={}")
             .format(
                train_queue_name))
    app.control.cancel_consumer(train_queue_name)
    log.info(("canceling consumer to queue={}")
             .format(
                predict_queue_name))
    app.control.cancel_consumer(predict_queue_name)
# end of try/ex
