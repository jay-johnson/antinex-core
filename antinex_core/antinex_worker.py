import json
import pandas as pd
from celery import Celery
from celery import signals
from antinex_core.log.setup_logging import build_colorized_logger
from antinex_utils.utils import ev
from antinex_utils.utils import ppj
from antinex_utils.consts import SUCCESS
from antinex_utils.make_predictions import make_predictions
from celery_connectors.subscriber import Subscriber


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
                "redis://localhost:6379/0"),
            train_queue_name=ev(
                "TRAIN_QUEUE",
                "webapp.train.requests"),
            predict_queue_name=ev(
                "PREDICT_QUEUE",
                "webapp.predict.requests")):
        """__init__

        :param name: worker name
        :param broker_url: connection string to broker
        :param train_queue_name: queue name for training requests
        :param predict_queue_name: queue name for predict requests
        """

        self.name = name
        log.info(("{} - INIT")
                 .format(
                     self.name))

        self.state = "INIT"
        self.models = {}
        self.recv_msgs = []
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
            "task_default_queue": "celery.redis.sub",
            "task_default_exchange": "celery.redis.sub",
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

        self.max_msgs = 100
    # end of __init__

    def start(
            self,
            app,
            ssl_options=None,
            task_retry_policy=None,
            transport_options=None,
            conn_attrs=None,
            use_train_queue=None,
            use_predict_queue=None):
        """start

        :param app: Celery app
        :param ssl_options: ssl dictionary
        :param task_retry_policy: retry policy
        :param transport_options: transport config dict
        :param conn_attrs: connection dict
        :param use_train_queue: name of queue for training requests
        :param use_predict_queue: name of queue for prediction requests
        """

        log.info(("{} - start")
                 .format(
                    name))

        log.info(("{} - start - creating subscriber")
                 .format(
                    name))

        self.all_queues_sub = Subscriber(
            self.name,
            self.broker_url,
            app,
            self.ssl_options,
            **self.conn_attrs)

        log.info(("{} - start - activating consumer for "
                  "queues={} callback={}")
                 .format(
                    name,
                    self.queues,
                    self.handle_train_message.__name__))

        self.all_queues_sub.consume(
            callback=self.handle_messages,
            queues=self.queues,
            exchange=None,
            routing_key=None,
            prefetch_count=self.conn_attrs["prefetch_count"])

        self.state = "ACTIVE"

        log.info(("{} - start - state={} done")
                 .format(
                    name,
                    self.state))
    # end of start

    def handle_messages(
            self,
            body,
            message):
        """handle_messages

        :param body: body contents
        :param message: message object
        """

        should_ack = True
        should_reject = False
        should_requeue = False
        try:
            delivery_info = message.delivery_info
            log.info(("{} - msg with routing_key={}")
                     .format(
                        self.name,
                        delivery_info["routing_key"]))
            if delivery_info["routing_key"] == "webapp.predict.requests":
                try:
                    self.handle_predict_message(
                        req=body,
                        message=message)
                except Exception as f:
                    log.error(("{} - failed handling PREDICT request "
                               "with ex={}")
                              .format(
                                self.name,
                                f))
                # end of try/ex
                should_ack = True
                should_ack = True
            elif delivery_info["routing_key"] == "webapp.train.requests":
                try:
                    self.handle_train_message(
                        req=body,
                        message=message)
                except Exception as f:
                    log.error(("{} - failed handling TRAIN request "
                               "with ex={}")
                              .format(
                                self.name,
                                f))
                # end of try/ex
                should_ack = True
            else:
                log.error(("{} - misconfiguration error - consumed message "
                           "from exchange={} with routing_key={} acking")
                          .format(
                            self.name,
                            delivery_info["exchange"],
                            delivery_info["routing_key"]))
                should_ack = True
            # end of handling messages from multiple queues
        except Exception as e:
            log.error(("{} - failed to handle message={} body={} ex={}")
                      .format(
                        self.name,
                        message,
                        body,
                        e))
        # end of try/ex

        self.recv_msgs.append(body)
        if len(self.recv_msgs) > self.max_msgs:
            self.recv_msgs.pop(0, None)

        if should_ack:
            message.ack()
        elif should_reject:
            message.reject()
        elif should_requeue:
            message.reject()
        else:
            log.error(("{} - acking message={} body={} by default")
                      .format(
                        self.name,
                        message,
                        body))
            message.ack()
        # end of handling for message pub/sub
    # end of handle_messages

    def handle_train_message(
            self,
            req,
            message):
        """handle_train_message

        :param req: body contents
        :param message: message object
        """
        log.info(("{} train msg={} "
                  "req={}")
                 .format(
                     self.name,
                     message.delivery_info,
                     str(req)[0:10]))
        model_name = str(
            req["label"]).strip().lstrip().lower()

        log.info(("{} loading predict_rows into a df")
                 .format(
                    model_name))

        if self.models.get(model_name, None) is not None:
            log.info(("re-training model={}")
                     .format(
                        model_name))

        predict_df = pd.read_json(req["predict_rows"])
        predict_feature = req["predict_feature"]
        res = make_predictions(req)
        if res["status"] == SUCCESS:
            res_data = res["data"]
            model = res_data["model"]
            acc_data = res_data["acc"]
            predictions = res_data["sample_predictions"]
            accuracy = acc_data.get(
                "accuracy",
                None)
            for idx, node in enumerate(predictions):
                actual_row = predict_df[(
                                predict_df["idx"] == node["_row_idx"]
                            )]
                log.info(("sample={} - {}={} predicted={}")
                         .format(
                            node["_row_idx"],
                            predict_feature,
                            str(actual_row[predict_feature]),
                            node[predict_feature]))
            # end of predicting predictions
            log.info(("{} made predictions={} found={} "
                      "accuracy={} model={}")
                     .format(
                        req["label"],
                        len(predict_df.index),
                        len(res_data["sample_predictions"]),
                        accuracy,
                        ppj(json.loads(model.model.to_json()))))

            self.models[model_name] = res_data["model"]
        else:
            log.error(("{} failed predictions={}")
                      .format(
                        req["label"],
                        len(predict_df.index)))
        # end of if good train and predict
    # end of handle_train_message

    def handle_predict_message(
            self,
            req,
            message):
        """handle_predict_message

        :param req: body contents
        :param message: message object
        """
        log.info(("{} predict msg={} "
                  "req={}")
                 .format(
                     self.name,
                     message.delivery_info,
                     str(req)[0:10]))

        model_name = str(
            req["label"]).strip().lstrip().lower()

        log.info(("{} loading predict_rows into a df")
                 .format(
                    model_name))

        if self.models.get(model_name, None) is not None:
            log.info(("Coming soon - predictions with existing model={}")
                     .format(
                        model_name))
        else:
            log.info(("{} model is not stored - training")
                     .format(
                        model_name))
            self.handle_train_message(
                req=req,
                message=message)
        # end of if can use model to predict or need to train
    # end of handle_predict_message

    def show_diagnostics(
            self):
        """show_diagnostics"""
        log.info(("{} - models={}")
                 .format(
                    self.name,
                    self.models))
        for midx, m in enumerate(self.recv_msgs):
            log.info(("msg={} contents={}")
                     .format(
                        midx,
                        ppj(m)))
    # end of show_diagnostics

    def shutdown(
            self):
        """shutdown"""
        log.info(("{} - shutting down - start")
                 .format(
                    self.name))
        self.state = "SHUTDOWN"
        self.show_diagnostics()
        log.info(("{} - shutting down - done")
                 .format(
                    self.name))
    # end of shutdown

# end of AntiNexCore


log.info("loading Celery app")
app = Celery()

broker_url = "redis://localhost:6379/6"
train_queue_name = "webapp.train.requests"
predict_queue_name = "webapp.predict.requests"

log.info("Creating antinex core")
core = AntiNexCore(
    name="core",
    broker_url=broker_url,
    train_queue_name=train_queue_name,
    predict_queue_name=predict_queue_name)
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
