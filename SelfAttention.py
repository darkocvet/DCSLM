import math
from nn import Layer
from value import Value

def dot_product(v1, v2):
    return sum((xi * yi for xi, yi in zip(v1, v2)), Value(0.0))

def micro_softmax(logits):

    counts = [logit.exp() for logit in logits]

    total = sum(counts, Value(0.0))

    return [c * (total**-1) for c in counts]

class SelfAttention:
    def __init__(self, n_embd, head_size):
        self.query = Layer(n_embd, head_size, nonlin=False)
        self.key = Layer(n_embd, head_size, nonlin=False)
        self.value = Layer(n_embd , head_size, nonlin=False)
        self.head_size = head_size
    
    def __call__(self, x):
        
        B = len(x)
        T = len(x[0])

        qs = [[self.query(x[b][t]) for t in range(T)] for b in range(B)]
        ks = [[self.key(x[b][t]) for t in range(T)] for b in range(B)]
        vs = [[self.value(x[b][t]) for t in range(T)] for b in range(B)]

        results = []

        for b in range(B):
            batch_results = []
            for t in range(T):
                energies = [dot_product(qs[b][t], ks[b][i]) * (self.head_size**-0.5) for i in range(t+1)]

                probs = micro_softmax(energies)

                new_vec = [Value(0.0) for _ in range(self.head_size)]
                for i,p in enumerate(probs):
                    for j in range(self.head_size):
                        new_vec[j] = new_vec[j] + p * vs[b][i][j]
                
                batch_results.append(new_vec)
            results.append(batch_results)
        
        return results



