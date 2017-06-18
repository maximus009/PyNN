import numpy as np
from numpy.matlib import randn

class Linear:

    def __init__(self, inputDim, outputDim, _variance=0.01):
        self.weight = randn(inputDim, outputDim) * _variance
        self.bias = randn((1, outputDim)) * _variance

        self.gradientWeight = np.zeros_like(self.weight)
        self.gradientBias = np.zeros_like(self.bias)

    def forward(self, x):
        self.current_x = x
        return np.dot(x, self.weight) + self.bias

    def backward(self, gradientOutput):
        x = self.current_x
        self.gradientWeight = np.dot(x.T, gradientOutput)
        self.gradientBias = np.dot(np.ones((1, gradientOutput.shape[0])), gradientOutput)
        return np.dot(gradientOutput, self.weight.T)

    def get_parameters(self):
        return {'weight':self.weight, 'bias':self.bias,
                'gradientWeight':self.gradientWeight,
                'gradientBias':self.gradientBias}

FC = Linear
Dense = Linear
