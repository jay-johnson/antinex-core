import mock
from tests.base_test import BaseTestCase
from tests.mock_make_predictions import mock_make_predictions_success
from tests.mock_make_predictions import mock_make_predictions_error
from tests.mock_make_predictions import mock_make_predictions_fail
from tests.mock_message import MockMessage
from antinex_core.log.setup_logging import build_colorized_logger
from antinex_core.antinex_processor import AntiNexProcessor


name = "test-train"
log = build_colorized_logger(name=name)


class TestTrain(BaseTestCase):

    @mock.patch(
        "antinex_utils.make_predictions.make_predictions",
        new=mock_make_predictions_success)
    def test_train_antinex_simple(self):

        exchange = "webapp.train.requests"
        routing_key = "webapp.train.requests"
        queue = "webapp.train.requests"
        max_models = 1
        prc = AntiNexProcessor(
            max_models=max_models)

        body = self.build_train_antinex_request()
        self.assertEqual(
            body["ml_type"],
            "classification")

        message = MockMessage(
            exchange=exchange,
            routing_key=routing_key,
            queue=queue)
        self.assertEqual(
            message.state,
            "NOTRUN")
        self.assertEqual(
            message.get_exchange(),
            exchange)
        self.assertEqual(
            message.get_routing_key(),
            routing_key)
        self.assertEqual(
            message.get_queue(),
            queue)

        self.assertEqual(
            len(prc.models),
            0)
        prc.handle_messages(
            body=body,
            message=message)
        self.assertEqual(
            message.state,
            "ACK")
        self.assertEqual(
            len(prc.models),
            max_models)

    # end of test_train_antinex_simple

    @mock.patch(
        "antinex_utils.make_predictions.make_predictions",
        new=mock_make_predictions_error)
    def test_train_antinex_simple_error(self):

        exchange = "webapp.train.requests"
        routing_key = "webapp.train.requests"
        queue = "webapp.train.requests"
        max_models = 1
        prc = AntiNexProcessor(
            max_models=max_models)

        body = self.build_train_antinex_request()
        self.assertEqual(
            body["ml_type"],
            "classification")

        message = MockMessage(
            exchange=exchange,
            routing_key=routing_key,
            queue=queue)
        self.assertEqual(
            message.state,
            "NOTRUN")
        self.assertEqual(
            message.get_exchange(),
            exchange)
        self.assertEqual(
            message.get_routing_key(),
            routing_key)
        self.assertEqual(
            message.get_queue(),
            queue)

        self.assertEqual(
            len(prc.models),
            0)
        prc.handle_messages(
            body=body,
            message=message)
        self.assertEqual(
            message.state,
            "ACK")
        self.assertEqual(
            len(prc.models),
            0)

    # end of test_train_antinex_simple_error

    @mock.patch(
        "antinex_utils.make_predictions.make_predictions",
        new=mock_make_predictions_fail)
    def test_train_antinex_simple_fail(self):

        exchange = "webapp.train.requests"
        routing_key = "webapp.train.requests"
        queue = "webapp.train.requests"
        max_models = 1
        prc = AntiNexProcessor(
            max_models=max_models)

        body = self.build_train_antinex_request()
        self.assertEqual(
            body["ml_type"],
            "classification")

        message = MockMessage(
            exchange=exchange,
            routing_key=routing_key,
            queue=queue)
        self.assertEqual(
            message.state,
            "NOTRUN")
        self.assertEqual(
            message.get_exchange(),
            exchange)
        self.assertEqual(
            message.get_routing_key(),
            routing_key)
        self.assertEqual(
            message.get_queue(),
            queue)

        self.assertEqual(
            len(prc.models),
            0)
        prc.handle_messages(
            body=body,
            message=message)
        self.assertEqual(
            message.state,
            "ACK")
        self.assertEqual(
            len(prc.models),
            0)

    # end of test_train_antinex_simple_fail

    @mock.patch(
        "antinex_utils.make_predictions.make_predictions",
        new=mock_make_predictions_success)
    def test_train_antinex_simple_model_cleanup(self):

        exchange = "webapp.train.requests"
        routing_key = "webapp.train.requests"
        queue = "webapp.train.requests"
        max_models = 1
        prc = AntiNexProcessor(
            max_models=max_models)

        body = self.build_train_antinex_request()
        self.assertEqual(
            body["ml_type"],
            "classification")

        message = MockMessage(
            exchange=exchange,
            routing_key=routing_key,
            queue=queue)
        self.assertEqual(
            message.state,
            "NOTRUN")
        self.assertEqual(
            message.get_exchange(),
            exchange)
        self.assertEqual(
            message.get_routing_key(),
            routing_key)
        self.assertEqual(
            message.get_queue(),
            queue)

        self.assertEqual(
            len(prc.models),
            0)
        prc.handle_messages(
            body=body,
            message=message)
        self.assertEqual(
            message.state,
            "ACK")
        self.assertEqual(
            len(prc.models),
            max_models)

        # now try to train a new one and test the cleanup
        body["label"] = "should-remove-the-first"
        self.assertEqual(
            len(prc.models),
            1)
        prc.handle_messages(
            body=body,
            message=message)
        self.assertEqual(
            message.state,
            "ACK")
        self.assertEqual(
            len(prc.models),
            max_models)
        for midx, model_name in enumerate(prc.models):
            self.assertEqual(
                model_name,
                body["label"])

    # end of test_train_antinex_simple_model_cleanup

    def test_train_antinex_simple_success(self):

        exchange = "webapp.train.requests"
        routing_key = "webapp.train.requests"
        queue = "webapp.train.requests"
        max_models = 1
        prc = AntiNexProcessor(
            max_models=max_models)

        body = self.build_train_antinex_request()
        self.assertEqual(
            body["ml_type"],
            "classification")

        message = MockMessage(
            exchange=exchange,
            routing_key=routing_key,
            queue=queue)
        self.assertEqual(
            message.state,
            "NOTRUN")
        self.assertEqual(
            message.get_exchange(),
            exchange)
        self.assertEqual(
            message.get_routing_key(),
            routing_key)
        self.assertEqual(
            message.get_queue(),
            queue)

        self.assertEqual(
            len(prc.models),
            0)
        prc.handle_messages(
            body=body,
            message=message)
        self.assertEqual(
            message.state,
            "ACK")
        self.assertEqual(
            len(prc.models),
            max_models)

    # end of test_train_antinex_simple_success

    def test_train_antinex_simple_success_retrain(self):

        exchange = "webapp.train.requests"
        routing_key = "webapp.train.requests"
        queue = "webapp.train.requests"
        max_models = 1
        prc = AntiNexProcessor(
            max_models=max_models)

        body = self.build_train_antinex_request()
        self.assertEqual(
            body["ml_type"],
            "classification")

        message = MockMessage(
            exchange=exchange,
            routing_key=routing_key,
            queue=queue)
        self.assertEqual(
            message.state,
            "NOTRUN")
        self.assertEqual(
            message.get_exchange(),
            exchange)
        self.assertEqual(
            message.get_routing_key(),
            routing_key)
        self.assertEqual(
            message.get_queue(),
            queue)

        self.assertEqual(
            len(prc.models),
            0)
        prc.handle_messages(
            body=body,
            message=message)
        self.assertEqual(
            message.state,
            "ACK")
        self.assertEqual(
            len(prc.models),
            max_models)

        self.assertEqual(
            len(prc.models),
            1)
        prc.handle_messages(
            body=body,
            message=message)
        self.assertEqual(
            message.state,
            "ACK")
        self.assertEqual(
            len(prc.models),
            max_models)

    # end of test_train_antinex_simple_success_retrain

# end of TestTrain
