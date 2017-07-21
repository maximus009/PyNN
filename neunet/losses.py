import numpy as np

class MeanSquareError:

    def forward(self, predictions, labels):
        self.current_predictions = predictions
        self.current_labels = labels
        return np.sum(np.square(predictions - labels))

    def backward(self, predictions=None, labels=None):
        if predictions is not None:
            self.current_labels = labels
            self.current_predictions = predictions
            
        _numberOfSamples = len(self.current_labels)
        return 2*_numberOfSamples*(self.current_predictions-self.current_labels)

    def __name__(self, name='mse'):
        self.name = name
        return name

MSE = MeanSquareError


class CategoricalCrossEntropy:

    def __name__(self, name='softmax'):
        self.name = name
        return name

    def forward(self, predictions, labels):
        self.current_predictions = predictions
        self.current_labels = labels
        self.num_samples = len(predictions)
        correctLogLikelihoods = -np.log(predictions[range(num_samples),labels])
        return 

    def backward(self, predictions, labels):

