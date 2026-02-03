import random
from value import Value

class Embedding:
    def __init__(self, vocab_size, embedding_dim):
        self.table = [
            [Value(random.uniform(-1,1)) for _ in range (embedding_dim)]
            for _ in range(vocab_size)
        ]
    
    def __call__(self, idx):
        return self.table[idx]
    
    def parameters(self):
        return [p for row in self.table for p in row]

class PositionalEmbedding:
    def __init__(self, max_len, dim):
        self.table = [[Value(random.uniform(-1,1)) for _ in range(dim)] for _ in range (max_len)]

    def __call__(self, position_idx):
        return self.table[position_idx]
    
    def parameters(self):
        return [p for row in self.table for p in row]