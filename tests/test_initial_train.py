from tests.base_test import BaseTestCase


class TestRegression(BaseTestCase):

    def test_training_for_regression(self):
        req = self.build_regression_train_request()
        self.assertEqual(
            req["ml_type"],
            "regression")
    # end of test_training_for_regression

# end of TestRegression
