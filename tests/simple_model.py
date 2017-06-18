import sys
import numpy as np
import argparse
import os

base_path='/Users/sivaramanks/code/PyNN/'
sys.path.append(base_path)

from neunet.layers import Dense #or Linear, or FC
from neunet.activations import Sigmoid, Relu, TanH
from neunet.losses import MeanSquareError as MSE
from neunet.models import SimpleModel

def trainModel(layerStack, batchInput, batchOutput, learningRate, loss):

    loss += layerStack['mse_loss'].forward(layerStack['sigmoid'].forward(layerStack['linear'].forward(batchInput)), batchOutput)
    _ = layerStack['linear'].backward(layerStack['sigmoid'].backward(layerStack['mse_loss'].backward()))

    layerStack['linear'].weight -= layerStack['linear'].gradientWeight*learningRate
    layerStack['linear'].bias -= layerStack['linear'].gradientBias*learningRate

    return layerStack, loss

def generate_random_sin(numberOfSamples=1000):

    x = np.random.uniform(-1, 1, (numberOfSamples, 4))
    y = np.expand_dims(np.sin(x.sum(axis = 1)), -1)
    print x.shape, y.shape
    return x,y

def generate_bin_mask(inputDim=4, numberOfSamples=1000):

    x = np.random.uniform(-1, 1, (numberOfSamples,40))
    y = np.expand_dims(np.array(x.sum(axis = 1)>=0, dtype=np.int),-1)
    return x, y

if __name__=="__main__":
    model = SimpleModel(trainModel)
    model.add(Dense(40, 1), 'linear')
    model.add(Sigmoid(), 'sigmoid')
    model.add(MSE(), 'mse_loss')
    inputX, outputY = generate_bin_mask(40,10000)
    print inputX.shape, outputY.shape
    model.train(inputX, outputY, batchSize=10, epochs=1000, learningRate=0.2, shuffle=False, verbose=True)
