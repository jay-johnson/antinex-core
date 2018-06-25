from spylunking.log.setup_logging import test_logger
from antinex_utils.consts import SUCCESS
from antinex_utils.consts import ERR
from antinex_utils.consts import FAILED
from tests.mock_model import MockModel


log = test_logger(name='mock-predictions')


def build_response_data(
        req):
    """build_response_data

    :param req: request dict
    """
    model = MockModel(
        req=req)
    predictions = req.get(
        "test_predictions",
        [])
    sample_predictions = req.get(
        "test_predictions",
        [])
    rounded = req.get(
        "test_predictions",
        [])
    accuracy = req.get(
        "test_accuracy",
        {
            "accuracy": 52.5
        })
    error = req.get(
        "test_error",
        None)
    image_file = req.get(
        "image_file",
        None)
    history = req.get(
        "history",
        None)
    histories = req.get(
        "histories",
        None)
    indexes = req.get(
        "test_indexes",
        None)
    scores = req.get(
        "test_scores",
        None)
    cm = req.get(
        "test_cm",
        None)
    predicts_merged = req.get(
        "test_predicts_merged",
        False)
    merge_df = req.get(
        "test_merge_df",
        None)
    data = {
        "predictions": predictions,
        "rounded_predictions": rounded,
        "sample_predictions": sample_predictions,
        "acc": accuracy,
        "scores": scores,
        "history": history,
        "histories": histories,
        "image_file": image_file,
        "model": model,
        "indexes": indexes,
        "confusion_matrix": cm,
        "are_predicts_merged": predicts_merged,
        "merge_df": merge_df,
        "err": error
    }
    return data
# end of build_response_data


def mock_make_predictions_success(
        req):
    """mock_make_predictions_success

    :param req: request dict
    """

    res = {
        "status": SUCCESS,
        "err": "mock success test",
        "data": build_response_data(req)
    }
    log.info("returning SUCCESS")
    return res
# end of mock_make_predictions_success


def mock_make_predictions_error(
        req):
    """mock_make_predictions_error

    :param req: request dict
    """

    res = {
        "status": ERR,
        "err": "mock error test",
        "data": build_response_data(req)
    }
    log.info("returning ERROR")
    return res
# end of mock_make_predictions_error


def mock_make_predictions_fail(
        req):
    """mock_make_predictions_fail

    :param req: request dict
    """

    res = {
        "status": FAILED,
        "err": "mock fail test",
        "data": build_response_data(req)
    }
    log.info("returning FAILED")
    return res
# end of mock_make_predictions_fail
