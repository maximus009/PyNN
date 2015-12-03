import theano
from theano import tensor as T
from time import time

def create_function():
    a = theano.tensor.vector()
    out = a + a**10
    f = theano.function([a], out)

def logistic_function():
    x = T.dmatrix('x')
    S = 1/(1+ T.exp(-x))
    s = time()
    logistic = theano.function([x],S)
    print "Time taken to create sigmoid:",time()-s
    s = time()
    print logistic([[0,1],[-1,-2]])
    print "Time taken to compute:",time()-s
    s = time()
    s2 = (1 + T.tanh(x/2))/2
    print "Time taken to create hyper:",time()-s
    s = time()
    logistic2 = theano.function([x], s2)
    print "Time taken to compute:",time()-s
    print logistic2([[0,1],[-1,-2]])


a,b = T.dmatrices('a','b')
diff = a - b
abs_diff = abs(diff)
diff_sq = diff**2
f = theano.function([a,b],[diff, abs_diff, diff_sq])
print f([[0,0],[1,2]], [[2,3],[4,1]])


