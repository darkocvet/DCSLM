import json

class Tokenizer:
    def __init__(self):
        self.stoi = {}
        self.itos = {}
        self.vocab = []
    
    def train(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        self.vocab = sorted(list(set(text)))
        self.stoi = {ch:i for i,ch in enumerate(self.vocab)}
        self.itos = {i:ch for i,ch in enumerate(self.vocab)}
        print(f"DCSLM Vocabulary built. Size: {len(self.vocab)}")
    
    def encode(self, s):
        return [self.stoi[c] for c in s]
    
    def decode(self,l):
        return ''.join([self.itos[i] for i in l])
    
    def save_vocab(self, path="vocab.json"):
        with open(path, 'r') as f:
            json.dump(self.stoi, f)
    
    def load_vocab(self, path="vocab.json"):
        with open(path, 'r') as f:
            self.stoi = json.load(f)
        
        self.itos = {int(i):ch for ch,i in self.stoi.items()}
        self.vocab = list(self.stoi.keys())
    


        


