import json
import unittest


class BaseTestCase(unittest.TestCase):

    def build_regression_train_request(
            self,
            data_file=("./tests/train/"
                       "regression.json")):
        """build_model_and_weights

        :param data_file: train and predict request
        """
        file_contents = open(data_file).read()
        data = json.loads(file_contents)
        return data
    # end of build_regression_train_request

# end of BaseTestCase
