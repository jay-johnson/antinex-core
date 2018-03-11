import datetime
import json
import uuid
import unittest
import pandas as pd
from antinex_core.log.setup_logging import build_colorized_logger
from celery_connectors.publisher import Publisher


name = "test_base"
log = build_colorized_logger(name=name)


class BaseTestCase(unittest.TestCase):

    def setUp(
            self):
        """setUp"""

        self.name = "testing_{}".format(
            str(uuid.uuid4()))
        self.broker_url = "memory://localhost/"
        self.ssl_options = None
        self.serializer = "json"

        self.train_exchange_name = "webapp.train.requests"
        self.train_routing_key = "webapp.train.requests"
        self.train_queue_name = "webapp.train.requests"

        self.predict_exchange_name = "webapp.predict.requests"
        self.predict_routing_key = "webapp.predict.requests"
        self.predict_queue_name = "webapp.predict.requests"

        self.pub = None

    # end of setUp

    def get_broker(
            self):
        """get_broker"""
        return self.broker_url
    # end of get_broker

    def get_ssl_options(
            self):
        """get_ssl_options"""
        return self.ssl_options
    # end of get_broker

    def get_publisher(
            self,
            broker_url=None,
            ssl_options=None):
        """get_publisher

        :param broker_url: broker url
        :param ssl_options: sll options
        """
        if self.pub:
            return self.pub

        self.pub = Publisher(
            name=self.name,
            auth_url=self.broker_url,
            ssl_options=self.ssl_options)

        return self.pub
    # end of get_publisher

    def publish(
            self,
            body=None,
            exchange=None,
            routing_key=None,
            queue=None,
            serializer="json",
            retry=True,
            silent=True):

        use_exchange = exchange
        if not use_exchange:
            use_exchange = self.train_exchange_name
        use_routing_key = routing_key
        if not use_routing_key:
            use_routing_key = self.train_routing_key
        use_queue = queue
        if not use_queue:
            use_queue = self.train_queue_name

        log.info(("Sending msg={} "
                  "ex={} rk={}")
                 .format(
                    body,
                    use_exchange,
                    use_routing_key))

        # Publish the message:
        self.pub.publish(
            body=body,
            exchange=use_exchange,
            routing_key=use_routing_key,
            queue=use_queue,
            serializer=serializer,
            retry=retry)

    # end of publish

    def build_train_antinex_request(
            self,
            data_file=("./training/"
                       "django-antinex-simple.json")):
        """build_model_and_weights

        :param data_file: train and predict request
        """
        body = {}
        with open(data_file, "r") as cur_file:
            file_contents = cur_file.read()
            body = json.loads(file_contents)

        # Now send:
        now = datetime.datetime.now().isoformat()
        body["created"] = now
        log.info("loading predict_rows")
        predict_rows_df = pd.read_csv(body["dataset"])
        predict_rows = []
        for idx, org_row in predict_rows_df.iterrows():
            new_row = json.loads(org_row.to_json())
            new_row["idx"] = len(predict_rows) + 1
            predict_rows.append(new_row)
        body["predict_rows"] = pd.DataFrame(predict_rows).to_json()

        log.info(("using predict_rows={}")
                 .format(
                    len(predict_rows)))

        return body
    # end of build_train_antinex_request

    def build_predict_antinex_request(
            self,
            data_file=("./training/"
                       "django-antinex-simple.json")):
        """build_model_and_weights

        :param data_file: predict request
        """
        body = {}
        with open(data_file, "r") as cur_file:
            file_contents = cur_file.read()
            body = json.loads(file_contents)

        # Now send:
        now = datetime.datetime.now().isoformat()
        body["created"] = now
        log.info("loading predict_rows")
        predict_rows_df = pd.read_csv(body["dataset"])
        predict_rows = []
        for idx, org_row in predict_rows_df.iterrows():
            new_row = json.loads(org_row.to_json())
            new_row["idx"] = len(predict_rows) + 1
            predict_rows.append(new_row)
        body["predict_rows"] = pd.DataFrame(predict_rows).to_json()

        log.info(("using predict_rows={}")
                 .format(
                    len(predict_rows)))

        return body
    # end of build_predict_antinex_request

    def build_predict_scaler_antinex_request(
            self,
            data_file=("./training/"
                       "scaler-django-antinex-simple.json")):
        """build_model_and_weights

        :param data_file: predict request
        """
        body = {}
        with open(data_file, "r") as cur_file:
            file_contents = cur_file.read()
            body = json.loads(file_contents)

        # Now send:
        now = datetime.datetime.now().isoformat()
        body["created"] = now
        log.info("loading predict_rows")
        predict_rows_df = pd.read_csv(body["dataset"])
        predict_rows = []
        for idx, org_row in predict_rows_df.iterrows():
            new_row = json.loads(org_row.to_json())
            new_row["idx"] = len(predict_rows) + 1
            predict_rows.append(new_row)
        body["predict_rows"] = pd.DataFrame(predict_rows).to_json()

        log.info(("using predict_rows={}")
                 .format(
                    len(predict_rows)))

        return body
    # end of build_predict_scaler_antinex_request

    def build_regression_train_request(
            self,
            data_file=("./tests/train/"
                       "regression.json")):
        """build_model_and_weights

        :param data_file: train and predict request
        """
        body = {}
        with open(data_file, "r") as cur_file:
            file_contents = cur_file.read()
            body = json.loads(file_contents)
        return body
    # end of build_regression_train_request

# end of BaseTestCase
