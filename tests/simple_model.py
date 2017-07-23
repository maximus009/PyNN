import sys
import numpy as np
import argparse
import os


from neunet.layers import Dense #or Linear, or FC
from neunet.activations import Sigmoid, Relu, TanH
from neunet.losses import MeanSquareError as MSE
from neunet.models import SimpleModel

parser = argparse.ArgumentParser(description='NeuNet Basic Example')
parser.add_argument('--batch-size', type=int, default=1, dest='batch_size',
                            help='input batch size for training (default: 1)')
parser.add_argument('--epochs', type=int, default=100, dest='epochs',
                            help='number of epochs for training (default: 100)')
parser.add_argument('--lr', type=float, default=0.01, dest='lr',
                            help='learning rate (default: 0.01)')
parser.add_argument('--act', type=str, default='sigm', dest='act',
                            help='Activations: (sigm)/relu/tanh')
parser.add_argument('--samples', type=int, default=1000, dest='samples',
                            help='Number of Sample Inputs (default: 1000)')
parser.add_argument('--shuffle', type=bool, default=False, dest='shuffle')
args = parser.parse_args()

def trainModel(layerStack, batchInput, batchOutput, learningRate, loss):

    loss += layerStack['mse_loss'].forward(layerStack['sigmoid'].forward(layerStack['linear'].forward(batchInput)), batchOutput)
    _ = layerStack['linear'].backward(layerStack['sigmoid'].backward(layerStack['mse_loss'].backward()))

    layerStack['linear'].weight -= layerStack['linear'].gradientWeight*learningRate
    layerStack['linear'].bias -= layerStack['linear'].gradientBias*learningRate

    return layerStack, loss

def generate_bin_mask(inputDim=4, numberOfSamples=args.samples):

    x = np.random.uniform(-1, 1, (numberOfSamples,inputDim))
    y = np.expand_dims(np.array(x.sum(axis = 1)>=0, dtype=np.int),-1)
    return x, y

if __name__=="__main__":
    model = SimpleModel(trainModel)
    model.add(Dense(784, 1), 'linear')
    Activation = {'sigm':Sigmoid, 'relu':Relu, 'tanh':TanH}[args.act]
    model.add(Activation(), 'sigmoid')
    model.add(MSE(), 'mse_loss')
    inputX, outputY = generate_bin_mask(784,args.samples)

    print inputX.shape, outputY.shape
    model.train(inputX, outputY, batchSize=args.batch_size, epochs=args.epochs, learningRate=args.lr,
            shuffle=args.shuffle, verbose=True)
