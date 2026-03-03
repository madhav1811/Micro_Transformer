import numpy as np
from src.components import Linear, ReLU, LayerNorm, Attention, Embedding

class MultiHeadAttention:
    def __init__(self, n_heads, n_embd):
        assert n_embd % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = n_embd // n_heads
        
        # Projections
        self.q_proj = Linear(n_embd, n_embd)
        self.k_proj = Linear(n_embd, n_embd)
        self.v_proj = Linear(n_embd, n_embd)
        self.out_proj = Linear(n_embd, n_embd)
        
        self.attention = Attention(scale=1.0 / np.sqrt(self.head_dim))

    def forward(self, x, mask=None):
        B, T, C = x.shape
        
        # 1. Project to Q, K, V
        q = self.q_proj.forward(x) # (B, T, C)
        k = self.k_proj.forward(x)
        v = self.v_proj.forward(x)
        
        # 2. Split into heads
        # (B, T, C) -> (B, T, H, D) -> (B, H, T, D)
        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # 3. Scaled dot-product attention
        # (B, H, T, D) -> (B, H, T, D)
        out = self.attention.forward(q, k, v, mask)
        
        # 4. Concatenate heads
        # (B, H, T, D) -> (B, T, H, D) -> (B, T, C)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        
        # 5. Output projection
        out = self.out_proj.forward(out)
        return out

    def backward(self, grad_out):
        B, T, C = grad_out.shape
        
        # 1. Backward through output projection
        grad_out = self.out_proj.backward(grad_out)
        
        # 2. Reshape back to heads
        # (B, T, C) -> (B, T, H, D) -> (B, H, T, D)
        grad_out_heads = grad_out.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # 3. Backward through attention
        dq, dk, dv = self.attention.backward(grad_out_heads)
        
        # 4. Transpose and reshape back
        # (B, H, T, D) -> (B, T, H, D) -> (B, T, C)
        dq = dq.transpose(0, 2, 1, 3).reshape(B, T, C)
        dk = dk.transpose(0, 2, 1, 3).reshape(B, T, C)
        dv = dv.transpose(0, 2, 1, 3).reshape(B, T, C)
        
        # 5. Backward through linear projections
        grad_x = self.q_proj.backward(dq)
        grad_x += self.k_proj.backward(dk)
        grad_x += self.v_proj.backward(dv)
        
        return grad_x

    def get_params(self):
        return [self.q_proj, self.k_proj, self.v_proj, self.out_proj]

class TransformerBlock:
    def __init__(self, n_embd, n_heads):
        self.ln1 = LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_heads, n_embd)
        self.ln2 = LayerNorm(n_embd)
        self.ffn = [
            Linear(n_embd, 4 * n_embd),
            ReLU(),
            Linear(4 * n_embd, n_embd)
        ]

    def forward(self, x, mask=None):
        # ln1 -> attn -> residual
        ln1_out = self.ln1.forward(x)
        attn_out = self.attn.forward(ln1_out, mask)
        x = x + attn_out
        
        # ln2 -> ffn -> residual
        ln2_out = self.ln2.forward(x)
        ffn_out = ln2_out
        for layer in self.ffn:
            ffn_out = layer.forward(ffn_out)
        x = x + ffn_out
        return x

    def backward(self, grad_out):
        # grad_out is dL/dx_output
        
        # Split grad_out into FFN path and residual path
        grad_ffn = grad_out
        
        # Backward through FFN
        for layer in reversed(self.ffn):
            grad_ffn = layer.backward(grad_ffn)
        
        # Backward through LN2
        grad_ln2 = self.ln2.backward(grad_ffn)
        
        # Add residual gradient
        grad_attn_input = grad_out + grad_ln2
        
        # Split grad_attn_input into Attn path and residual path
        grad_attn = self.attn.backward(grad_attn_input)
        
        # Backward through LN1
        grad_ln1 = self.ln1.backward(grad_attn)
        
        # Final residual grad
        return grad_attn_input + grad_ln1

    def get_params(self):
        params = self.attn.get_params()
        params += [self.ln1, self.ln2]
        params += [self.ffn[0], self.ffn[2]] # Linear layers
        return params

class MicroGPT:
    def __init__(self, vocab_size, n_embd, n_heads, n_layers, block_size):
        self.block_size = block_size
        self.vocab_size = vocab_size
        
        self.token_embedding = Embedding(vocab_size, n_embd)
        self.pos_embedding = Embedding(block_size, n_embd)
        
        self.blocks = [TransformerBlock(n_embd, n_heads) for _ in range(n_layers)]
        self.ln_f = LayerNorm(n_embd)
        self.head = Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        
        # 1. Embeddings
        tok_emb = self.token_embedding.forward(idx) # (B, T, C)
        pos_emb = self.pos_embedding.forward(np.arange(T)) # (T, C)
        x = tok_emb + pos_emb
        
        # 2. Causal Mask
        mask = np.tril(np.ones((T, T)))
        
        # 3. Transformer Blocks
        for block in self.blocks:
            x = block.forward(x, mask)
        
        # 4. Final LN and Head
        x = self.ln_f.forward(x)
        logits = self.head.forward(x)
        return logits

    def backward(self, grad_logits):
        # 1. Backward through Head
        grad_x = self.head.backward(grad_logits)
        
        # 2. Backward through Final LN
        grad_x = self.ln_f.backward(grad_x)
        
        # 3. Backward through Blocks
        for block in reversed(self.blocks):
            grad_x = block.backward(grad_x)
            
        # 4. Gradients for embeddings
        self.token_embedding.backward(grad_x)
        # Position embedding depends on time axis sum
        self.pos_embedding.backward(np.sum(grad_x, axis=0))
        
        return grad_x

    def get_params(self):
        params = [self.token_embedding, self.pos_embedding]
        for block in self.blocks:
            params += block.get_params()
        params += [self.ln_f, self.head]
        return params

    def save_weights(self, path):
        """Save model parameters to a .npz file"""
        weights = {}
        for i, p in enumerate(self.get_params()):
            for attr in ['W', 'b', 'gamma', 'beta']:
                if hasattr(p, attr):
                    weights[f'p_{i}_{attr}'] = getattr(p, attr)
        np.savez(path, **weights)

    def load_weights(self, path):
        """Load model parameters from a .npz file"""
        data = np.load(path)
        for i, p in enumerate(self.get_params()):
            for attr in ['W', 'b', 'gamma', 'beta']:
                key = f'p_{i}_{attr}'
                if key in data:
                    loaded_val = data[key]
                    current_val = getattr(p, attr)
                    if loaded_val.shape == current_val.shape:
                        setattr(p, attr, loaded_val)
                    else:
                        print(f"Warning: Shape mismatch for {key}. Expected {current_val.shape}, got {loaded_val.shape}. Skipping.")
