import numpy as np

class Sigmoid:

    def __name__(self, name='sigmoid'):
        self.name = name
        return name

    def forward(self, x):
        self.current_x = 1/(1+np.exp(-x))
        return self.current_x

    def backward(self, gradientOutput):
        return np.multiply(np.multiply(self.current_x, 1-self.current_x), gradientOutput)

class Relu:

    def __name__(self, name='relu'):
        self.name = name
        return name

    def forward(self, x):
        self.current_x = x
        return np.maximum(0, x)

    def backward(self, gradientOutput):
        return np.multiply(gradientOutput, np.greater(self.current_x , 0))

ReLU = Relu

class TanH:

    def __name__(self, name='tanh'):
        self.name = name
        return name

    def forward(self, x):
        self.current_x = np.divide(np.exp(x) - np.exp(-x), np.exp(x) + np.exp(-x))
        return self.current_x

    def backward(self, gradientOutput):
        return np.multiply(gradientOutput, (1.0 - np.power(self.current_x, 2)))

class Softmax:

    def __name__(self, name='softmax'):
        self.name = name
        return name

    def forward(self, x, temperature = 1.0):
        if x.ndim == 1:
            x = x.reshape((1, -1))
        exp_x = np.exp((x)/temperature)
        self.current_probabilities = exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))
        return self.current_probabilities

    def backward(self, gradientOutput):
        pass


def test(obj, x):
    print obj.forward(x,1)
    print obj.forward(x,2)
    print obj.forward(x,3)
    print obj.forward(x,4)
    print obj.forward(x,5)
#    print obj.backward([1, 2])

if __name__ == '__main__':
    #x = np.array([[1,1,2,3,4],[3,2,3,2,1],[1,8,7,5,2]])
    x = np.array([1,2,3])
    test(Softmax(),x)
