import numpy as np

class Sigmoid:

    def forward(self, x):
        self.current_x = 1/(1+np.exp(-x))
        return self.current_x

    def backward(self, gradientOutput):
        return np.multiply(np.multiply(self.current_x, 1-self.current_x), gradientOutput)

class Relu:

    def forward(self, x):
        self.current_x = x
        return np.maximum(x,0)

    def backward(self, gradientOutput):
        return np.multiply(gradientOutput, self.current_x > 0)

ReLU = Relu

class TanH:

    def forward(self, x):
        self.current_x = np.divide(np.exp(x) - np.exp(-x), np.exp(x) + np.exp(-x))
        return self.current_x

    def backward(self, gradientOutput):
        return np.multiply(gradOutput, (1.0 - np.power(self.current_tanh, 2)))
