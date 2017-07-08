import numpy as np

class MeanSquareError:

    def forward(self, predictions, labels):
        self.current_predictions = predictions
        self.current_labels = labels
        return np.sum(np.square(predictions - labels))

    def backward(self):
        _numberOfSamples = len(self.current_labels)
        return 2*_numberOfSamples*(self.current_predictions-self.current_labels)

MSE = MeanSquareError
