import mock
from tests.base_test import BaseTestCase
from tests.mock_make_predictions import mock_make_predictions_success
from tests.mock_make_predictions import mock_make_predictions_error
from tests.mock_make_predictions import mock_make_predictions_fail
from tests.mock_message import MockMessage
from spylunking.log.setup_logging import build_colorized_logger
from antinex_core.antinex_processor import AntiNexProcessor


name = "test-predict"
log = build_colorized_logger(name=name)


class TestPredict(BaseTestCase):

    @mock.patch(
        "antinex_utils.make_predictions.make_predictions",
        new=mock_make_predictions_success)
    def test_predict_antinex_simple(self):

        exchange = "webapp.predict.requests"
        routing_key = "webapp.predict.requests"
        queue = "webapp.predict.requests"
        max_models = 1
        prc = AntiNexProcessor(
            max_models=max_models)

        body = self.build_predict_antinex_request()
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

    # end of test_predict_antinex_simple

    @mock.patch(
        "antinex_utils.make_predictions.make_predictions",
        new=mock_make_predictions_error)
    def test_predict_antinex_simple_error(self):

        exchange = "webapp.predict.requests"
        routing_key = "webapp.predict.requests"
        queue = "webapp.predict.requests"
        max_models = 1
        prc = AntiNexProcessor(
            max_models=max_models)

        body = self.build_predict_antinex_request()
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

    # end of test_predict_antinex_simple_error

    @mock.patch(
        "antinex_utils.make_predictions.make_predictions",
        new=mock_make_predictions_fail)
    def test_predict_antinex_simple_fail(self):

        exchange = "webapp.predict.requests"
        routing_key = "webapp.predict.requests"
        queue = "webapp.predict.requests"
        max_models = 1
        prc = AntiNexProcessor(
            max_models=max_models)

        body = self.build_predict_antinex_request()
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

    # end of test_predict_antinex_simple_fail

    @mock.patch(
        "antinex_utils.make_predictions.make_predictions",
        new=mock_make_predictions_success)
    def test_predict_antinex_simple_model_cleanup(self):

        exchange = "webapp.predict.requests"
        routing_key = "webapp.predict.requests"
        queue = "webapp.predict.requests"
        max_models = 1
        prc = AntiNexProcessor(
            max_models=max_models)

        body = self.build_predict_antinex_request()
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

        # now try to predict a new one and test the cleanup
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

    # end of test_predict_antinex_simple_model_cleanup

    def test_predict_antinex_simple_success(self):

        exchange = "webapp.predict.requests"
        routing_key = "webapp.predict.requests"
        queue = "webapp.predict.requests"
        max_models = 1
        prc = AntiNexProcessor(
            max_models=max_models)

        body = self.build_predict_antinex_request()
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

    # end of test_predict_antinex_simple_success

    def test_predict_antinex_simple_success_repredict(self):

        exchange = "webapp.predict.requests"
        routing_key = "webapp.predict.requests"
        queue = "webapp.predict.requests"
        max_models = 1
        prc = AntiNexProcessor(
            max_models=max_models)

        body = self.build_predict_antinex_request()
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

    # end of test_predict_antinex_simple_success_repredict

    def test_scaler_predict_antinex_simple_success_repredict(self):

        exchange = "webapp.predict.requests"
        routing_key = "webapp.predict.requests"
        queue = "webapp.predict.requests"
        max_models = 1
        prc = AntiNexProcessor(
            max_models=max_models)

        body = self.build_predict_scaler_antinex_request()
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

    # end of test_scaler_predict_antinex_simple_success_repredict

    def test_scaler_repredict_with_just_one_row(self):

        exchange = "webapp.predict.requests"
        routing_key = "webapp.predict.requests"
        queue = "webapp.predict.requests"
        max_models = 1
        prc = AntiNexProcessor(
            max_models=max_models)

        body = self.build_predict_scaler_antinex_request()
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

        num_rows_at_bottom = 1
        predict_rows_body = self.build_predict_rows_from_dataset(
            num_rows_at_bottom=num_rows_at_bottom)

        # make sure to remove the dataset arg to trigger
        # the predict from just the list of dictionary rows
        predict_rows_body.pop(
            "dataset",
            None)
        self.assertEqual(
            len(prc.models),
            1)
        prc.handle_messages(
            body=predict_rows_body,
            message=message)
        self.assertEqual(
            message.state,
            "ACK")
        self.assertEqual(
            len(prc.models),
            max_models)

        name_of_model = predict_rows_body["label"].lower().strip()
        self.assertTrue(
            name_of_model in prc.models)
        model_data = prc.models[name_of_model]
        self.assertTrue(
            (len(model_data["data"]["sample_predictions"])
             == num_rows_at_bottom))
    # end of test_scaler_repredict_with_just_one_row

    def test_scaler_repredict_with_multiple_rows(self):

        exchange = "webapp.predict.requests"
        routing_key = "webapp.predict.requests"
        queue = "webapp.predict.requests"
        max_models = 1
        prc = AntiNexProcessor(
            max_models=max_models)

        body = self.build_predict_scaler_antinex_request()
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

        num_rows_at_bottom = 4
        predict_rows_body = self.build_predict_rows_from_dataset(
            num_rows_at_bottom=num_rows_at_bottom)

        # make sure to remove the dataset arg to trigger
        # the predict from just the list of dictionary rows
        predict_rows_body.pop(
            "dataset",
            None)
        self.assertEqual(
            len(prc.models),
            1)
        prc.handle_messages(
            body=predict_rows_body,
            message=message)
        self.assertEqual(
            message.state,
            "ACK")
        self.assertEqual(
            len(prc.models),
            max_models)

        name_of_model = predict_rows_body["label"].lower().strip()
        self.assertTrue(
            name_of_model in prc.models)
        model_data = prc.models[name_of_model]
        self.assertTrue(
            (len(model_data["data"]["sample_predictions"])
             == num_rows_at_bottom))
    # end of test_scaler_repredict_with_multiple_rows

    def test_scaler_first_time_predict_with_rows(self):

        exchange = "webapp.predict.requests"
        routing_key = "webapp.predict.requests"
        queue = "webapp.predict.requests"
        max_models = 1
        prc = AntiNexProcessor(
            max_models=max_models)

        num_rows_at_bottom = 7
        predict_rows_body = self.build_predict_rows_from_dataset(
            num_rows_at_bottom=num_rows_at_bottom)
        name_of_model = predict_rows_body["label"].lower().strip()

        # make sure to remove the dataset arg to trigger
        # the predict from just the list of dictionary rows
        predict_rows_body.pop(
            "dataset",
            None)

        self.assertEqual(
            predict_rows_body["ml_type"],
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
            body=predict_rows_body,
            message=message)
        self.assertEqual(
            message.state,
            "ACK")
        self.assertEqual(
            len(prc.models),
            max_models)
        model_data = prc.models[name_of_model]
        print(len(model_data["data"]["sample_predictions"]))
        self.assertTrue(
            (len(model_data["data"]["sample_predictions"])
             == num_rows_at_bottom))

        num_rows_at_bottom = 4
        predict_rows_body = self.build_predict_rows_from_dataset(
            num_rows_at_bottom=num_rows_at_bottom)

        # make sure to remove the dataset arg to trigger
        # the predict from just the list of dictionary rows
        predict_rows_body.pop(
            "dataset",
            None)
        self.assertEqual(
            len(prc.models),
            1)
        prc.handle_messages(
            body=predict_rows_body,
            message=message)
        self.assertEqual(
            message.state,
            "ACK")
        self.assertEqual(
            len(prc.models),
            max_models)

        self.assertTrue(
            name_of_model in prc.models)
        model_data = prc.models[name_of_model]
        self.assertTrue(
            (len(model_data["data"]["sample_predictions"])
             == num_rows_at_bottom))
    # end of test_scaler_first_time_predict_with_rows

# end of TestPredict
