import numpy as np
import argparse
import os
import sys
from neunet.layers import Dense #or Linear, or FC
from neunet.activations import Sigmoid, Relu, TanH
from neunet.losses import MeanSquareError as MSE
from neunet.models import SimpleModel, Sequential
from datasets import mnist

parser = argparse.ArgumentParser(description='NeuNet Basic Example')
parser.add_argument('--batch-size', type=int, default=1, dest='batch_size',
                            help='input batch size for training (default: 1)')
parser.add_argument('--epochs', type=int, default=10, dest='epochs',
                            help='number of epochs for training (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, dest='lr',
                            help='learning rate (default: 0.01)')
parser.add_argument('--act', type=str, default='sigm', dest='act',
                            help='Activations: (sigm)/relu/tanh')
parser.add_argument('--samples', type=int, default=1000, dest='samples',
                            help='Number of Sample Inputs (default: 1000)')
parser.add_argument('--shuffle', type=bool, default=False, dest='shuffle')
args = parser.parse_args()


def mnistFunction(layerStack, batchInput, batchOutput, learningRate, loss):
    return

def get_model():

    model = Sequential()
    model.add(Dense(784, 128))
    model.add(Sigmoid())
    model.add(Dense(128, 32))
    model.add(Relu())
    model.add(Dense(32, 10))
    model.add_loss(MSE())
    return model

if __name__=="__main__":
    inputX, outputY = mnist.load_test_data(one_hot_label=True, flatten_input=True)
    #inputX = np.reshape(inputX, (len(inputX), 784))/255.
    #outputY = np.expand_dims(outputY, -1)
    print inputX.shape, outputY.shape
    model = get_model()
    _ = model.forward(inputX)
    print _
    model.backward(_, outputY, 0.01)
