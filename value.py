import math

   
def _tanh(a):
    t = math.tanh(a.data)
    out = Value(t)
    out._prev = (a, )

    def _backward():
        a.grad += (1 - t**2) * out.grad
    
    out._backward = _backward
    return out

class Value:
    def __init__(self, data, _prev=()):
        self.data = data
        self.grad = 0
        self._prev = _prev
        self._backward = lambda: None

    def add(a,b):
        out = Value(a.data + b.data)
        out._prev = (a,b)

        def _backward():
            a.grad += out.grad
            b.grad += out.grad

        out._backward = _backward
        return out
    
    def multiply(a,b):
        out = Value(a.data*b.data)
        out._prev = (a,b)

        def _backward():
            a.grad += b.data * out.grad
            b.grad += a.data * out.grad
        
        out._backward = _backward
        return out
    
    def subtract(a,b):
        out = Value(a.data - b.data)
        out._prev = (a,b)

        def _backward():
            a.grad += out.grad
            b.grad -= out.grad
        
        out._backward = _backward
        return out
    
    def negate(a):
        out = Value(-a.data)
        out._prev = (a,)

        def _backward():
            a.grad -= out.grad
        
        out._backward = _backward
        return out
    
    def power(a,n):
        out = Value(a.data**n)
        out._prev = (a,)

        def _backward():
            a.grad += n * (a.data ** (n-1)) * out.grad
        
        out._backward = _backward
        return out 

    def division(a,b, eps=1e-8):
        out = Value(a.data/(b.data+eps))
        out._prev = (a,b)

        def _backward():
            a.grad += 1/(b.data+eps) * out.grad
            b.grad += (-a.data / ((b.data+eps)**2)) * out.grad
        
        out._backward = _backward
        return out
    

    
    def backward(self):

        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for parent in v._prev:
                    build_topo(parent)
                topo.append(v)
        build_topo(self)

        self.grad = 1

        for node in reversed(topo):
            node._backward()
    
    @staticmethod
    def mse(y_pred, y_true):
        diff = y_pred - y_true
        return diff**2

    def __add__(self, other):
        return Value.add(self, other)
    
    def __mul__(self, other):
        return Value.multiply(self,other)
    
    def __sub__(self, other):
        return Value.subtract(self,other)
    
    def __neg__(self):
        return Value.negate(self)
    
    def __pow__(self, n):
        return Value.power(self,n)
    
    def __truediv__(self, other):
        return Value.division(self, other)
    
    def tanh(self):
        return _tanh(self)
    
    