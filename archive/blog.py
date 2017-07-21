from numpy.random import random
from sys import maxint as Infinity
from scipy.special import expit as sig

#LIST OF FUNCTIONS/GATES
def forwardAddGate(x, y):
    return x+y

def forwardMultiplyGate(x,y):
    return x*y

def forwardCircuit(x, y, z):
    q = forwardAddGate(x, y)
    return forwardMultiplyGate(q, z)
 
#REAL VALUED CIRCUITS
def randomLocalSearch(function = forwardMultiplyGate, x = -2, y = 3, tweak_amount = 0.01):
    best_out = -Infinity
    best_x, best_y = x, y
    for k in range(100):
        x_try = x + tweak_amount * (random() * 2 - 1)
        y_try = y + tweak_amount * (random() * 2 - 1)
        out = function(x_try, y_try)
        if out > best_out:
            best_out = out
            best_x, best_y = x_try, y_try
    print best_x, best_y, best_out

def numericalGradient(function = forwardMultiplyGate, x = -2, y = 3, h = 0.0001, step_size = 0.01):
    out = function(x, y)
    xph = x + h
    out2 = function(xph, y)
    x_derivative = (out2 - out)/h
    yph = y + h
    out3 = function(x, yph)
    y_derivative = (out3 - out)/h
    
    x += step_size*x_derivative
    y += step_size*y_derivative
    out_new = function(x, y)
    print out_new

def analyticGradient(function= forwardMultiplyGate, x = -2, y = 3, step_size = 0.01):
    x_derivative = y
    y_derivative = x
    x += step_size*x_derivative
    y += step_size*y_derivative
    out_new = function(x, y)
    print out_new

#BACKPROPOGATION
def chainRule(func1 = forwardAddGate, func2 = forwardMultiplyGate, x = -2, y = 5, z = -4, step_size = 0.01):
    q = func1(x, y)
    f = func2(q, z)
    
    derivative_f_wrt_z = q
    derivative_f_wrt_q = z

    derivative_q_wrt_x = 1.0
    derivative_q_wrt_y = 1.0

    derivative_f_wrt_x = derivative_q_wrt_x * derivative_f_wrt_q
    derivative_f_wrt_y = derivative_q_wrt_y * derivative_f_wrt_q

    gradient_f_wrt_xyz = [derivative_f_wrt_x, derivative_f_wrt_y, derivative_f_wrt_z]
    x += step_size*derivative_f_wrt_x
    y += step_size*derivative_f_wrt_y
    z += step_size*derivative_f_wrt_z
    q_new = func1(x, y)
    f_new = func2(q_new, z)
    print q_new, f_new

def numericalGradientCheck(func = forwardCircuit, x = -2, y = 5, z = -4, h = 0.0001):
    x_derivative = (func(x+h, y, z) - func(x,y,z))/h
    y_derivative = (func(x, y+h, z) - func(x,y,z))/h
    z_derivative = (func(x, y, z+h) - func(x,y,z))/h
    print x_derivative, y_derivative, z_derivative

#SINGLE NEURON
class Unit:
    def __init__(self, value, grad = 0.0):
        self.value, self.grad = value, grad

class addGate:
    def __init__(self):
        pass

    def forward(self, unit0, unit1):
        self.u0, self.u1 = unit0, unit1
        self.utop = Unit(forwardAddGate(unit0.value, unit1.value), 0.0)
        return self.utop

    def backward(self):
        print "add back"
        self.u0.grad += 1 * self.utop.grad
        self.u1.grad += 1 * self.utop.grad

class multiplyGate:
    def __init__(self):
        pass

    def forward(self, unit0, unit1):
        self.u0, self.u1 = unit0, unit1
        self.utop = Unit(forwardMultiplyGate(self.u0.value, self.u1.value), 0.0)
        return self.utop

    def backward(self):
        self.u1.grad += self.u0.grad * self.utop.grad
        self.u0.grad += self.u1.grad * self.utop.grad
        
class sigmoidGate:
    def __init__(self):
        self.sig = sig    

    def forward(self, unit0):
        self.u0 = unit0
        self.utop = Unit(self.sig(unit0.value), 0.0)
        return self.utop

    def backward(self):
        s = self.sig(self.utop.value)
        self.u0.grad += (s*(1 - s)) * self.utop.grad


