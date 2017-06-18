from numpy.random import choice

class SimpleModel:

    def __init__(self, trainFunction):
        print 'Creating NeuNet model'
        self.layerStack ={}
        self.trainFunction = trainFunction

    def add(self, layerObject, layerName):
        self.layerStack[layerName] = layerObject
        print layerName,'added'

    def train(self, inputX = None, outputY = None,
                batchSize = 1, epochs = 10,
                learningRate = 0.001,
                printStep = 10, shuffle = True,
                verbose = False):
        numberOfSamples = inputX.shape[0]

        if verbose:
            print 'Running{} Stochastic Gradient Descent on {} samples'.format([' Mini-batch',''][batchSize==1], numberOfSamples)

        for epoch in range(1,epochs+1):
            loss = 0

            for batchIteration in range(numberOfSamples/batchSize):
                if shuffle:
                    batchRandomIndices = choice(range(numberOfSamples), batchSize)
                    batchInput = inputX[batchRandomIndices]
                    batchOutput = outputY[batchRandomIndices]
                else:
                    batchInput = inputX[batchIteration*batchSize:(batchIteration+1)*batchSize]
                    batchOutput = outputY[batchIteration*batchSize:(batchIteration+1)*batchSize]

                self.layerStack, loss = self.trainFunction(self.layerStack, batchInput, batchOutput, learningRate, loss)
            if epoch%printStep == 0 and verbose:
                print "Epoch:{} | Loss: {}".format(epoch, loss/(batchSize*(batchIteration+1)))
