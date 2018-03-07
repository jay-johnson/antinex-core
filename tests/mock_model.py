import json


class MockInternalModel:

    def to_json(
            self):
        """to_json"""
        mock_model_dict = {
            "model": "not_real"
        }
        return json.dumps(mock_model_dict)
    # end of to_json

# end of MockInternalModel


class MockModel:

    def __init__(
            self,
            req):
        """__init__

        :param req: test body request
        """
        self.req = req
        self.called_fit = False
        self.called_evaluate = False
        self.called_predict = False
        self.called_compile = False
        self.called_load_weights = False
        self.called_save_weights = False

        self.model = MockInternalModel()

        self.loss = req.get(
            "loss",
            "binary_crossentropy")
        self.optimizer = req.get(
            "optimizer",
            "adam")
        self.metrics = req.get(
            "metrics",
            [
                "accurary"
            ])
        self.eval_scores = req.get(
            "eval_scores",
            [52.5, 52.5])
        self.histories = req.get(
            "histories",
            [
                "val_loss",
                "val_acc",
                "loss",
                "acc"
            ])
        self.predictions = req.get(
            "predictions",
            [123, 567, 432])
        self.layers = []
        self.weights = {
            "weights": [123.321]
        }
    # end of __init__

    def fit(
            self,
            x,
            y,
            validation_data,
            epochs,
            batch_size,
            shuffle,
            verbose):
        """fit

        :param x: train rows
        :param y: target rows
        :param validation_data: validation x + y tuple
        :param epochs: num epochs
        :param batch_size: batch processing size
        :param shuffle: should shuffle data
        :param verbose: is silent or not
        """

        self.called_fit = True
        return self.histories
    # end of fit

    def evaluate(
            self,
            x,
            y):
        """evaluate

        :param x: train rows
        :param y: target rows
        """
        self.called_evaluate = True
        return self.eval_scores
    # end of evaluate

    def predict(
            self,
            x):
        """predict

        :param x: train rows
        """
        self.called_predict = True
        return self.predictions
    # end of predict

    def add(
            self,
            layer_obj):
        """add

        :param layer_obj: model layer usually Dense
        """
        self.layers.append(layer_obj)
        return
    # end of add

    def compile(
            self,
            loss=None,
            optimizer=None,
            metrics=None):
        """compile

        :param loss: loss
        :param optimizer: optimizer
        :param metrics: metrics list
        """
        self.called_compile = True
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        return
    # end of compile

    def load_weights(
            self,
            weights_file):
        """load_weights

        :param weights_file: h5 file on disk
        """
        self.called_load_weights = True
        return
    # end of load_weights

    def save_weights(
            self,
            weights_file):
        """save_weights

        :param weights_file: h5 file on disk
        """
        self.called_load_weights = True
        return
    # end of save_weights

    def get_weights(
            self):
        """get_weights

        :param weights_file: h5 file on disk
        """
        return self.weights
    # end of get_weights

# end of MockModel
