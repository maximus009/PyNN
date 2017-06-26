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
        return np.maximum(0, x)

    def backward(self, gradientOutput):
        return np.multiply(gradientOutput, np.greater(self.current_x , 0))

ReLU = Relu

class TanH:

    def forward(self, x):
        self.current_x = np.divide(np.exp(x) - np.exp(-x), np.exp(x) + np.exp(-x))
        return self.current_x

    def backward(self, gradientOutput):
        return np.multiply(gradientOutput, (1.0 - np.power(self.current_x, 2)))


def test(obj, x):
    print x
    print obj.forward(x)
    print obj.backward([1, 2])

if __name__ == '__main__':
    test(Relu(), [0.5, 1])
