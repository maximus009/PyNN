from numpy.random import choice

class SimpleModel:

    def __init__(self, trainFunction):
        print 'Creating NeuNet model'
        self.layerStack = {}
        self.trainFunction = trainFunction

    def add(self, layerObject, layerName):
        self.layerStack[layerName] = layerObject
        print layerName,'added'

    def objective(self, lossObject, lossName = 'loss'):
        self.loss = lossObject
        print "Objective/Loss function: %s added " % lossName

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

class Sequential:

    def __init__(self, modelName='model'):
        self.__name__(modelName)
        print 'Creating NeuNet Sequential Model', self.name
        self.layerStack = []
        self.layers = {}
        self._layer_count = []
        self.loss = None

    def __name__(self, name='model'):
        self.name = name
        return name

    def add(self, layerObject, layerName=None):
        _lname = layerObject.__name__() if layerName is None else layerName
        self._layer_count.append(_lname)
        self.layerStack.append(layerObject)
        _lname = _lname+'_'+str(self._layer_count.count(_lname))
        self.layers[_lname] = layerObject
        print _lname,"added"

    def add_loss(self, lossObject=None, lossName=None):
        self.lossName = lossName if lossName is not None else lossObject.__name__()
        self.lossObject = lossObject

    def forward(self, batchInput):
        _output = batchInput
        for layer in self.layerStack:
            _output = layer.forward(_output)
        return _output

    def backward(self, score, batchOutput, learningRate):
        _output = self.lossObject.backward(score, batchOutput)
        print _output
        for layer in self.layerStack[::-1]:
            _output = layer.backward(_output)

        for layer in self.layerStack:
            if 'gradient_descent_update' in dir(layer):
                layer.gradient_descent_update(learningRate)
        

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

##                loss = self.trainFunction(self.layerStack, batchInput, batchOutput, learningRate, loss)
                self.forward()


            if epoch%printStep == 0 and verbose:
                print "Epoch:{} | Loss: {}".format(epoch, loss/(batchSize*(batchIteration+1)))


