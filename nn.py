import random
import math
from value import Value

class Neuron:
    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1) * (1/math.sqrt(nin))) for _ in range (nin)]
        self.b = Value(0.0)
        self.nonlin = nonlin
    
    def __call__(self, x):
        
        out = self.b
        for wi,xi in zip(self.w, x):
            out = out + wi * xi
        
        if (self.nonlin):
            out = out.tanh()
        
        return out
    
    def parameters(self):
        return self.w + [self.b]



        
