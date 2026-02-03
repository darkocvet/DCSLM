from embedding import Embedding, PositionalEmbedding
from nn import MLP

class DCSLM:
    def __init__(self, vocab_size, embedding_dim, max_len):
        self.token_emb = Embedding(vocab_size, embedding_dim)
        self.pos_emb = PositionalEmbedding(max_len, embedding_dim)

        self.lm_head = MLP(embedding_dim, [32, vocab_size])

    def forward(self, char_ids):

        fused_inputs = []

        for i, char_id in enumerate(char_ids):

            c_vec = self.token_emb(char_id)
            p_vec = self.pos_emb(i)

            fused_vec = [cv + pv for cv, pv in zip(c_vec, p_vec)]
            fused_inputs.append(fused_vec)
        
        logits = [self.lm_head(vec) for vec in fused_inputs]

        return logits
    
    def parameters(self):
        return (self.token_emb.parameters() + 
                self.pos_emb.parameters() + 
                self.lm_head.parameters())
