import numpy as np

class Linear:
    """
    Fully connected layer: y = xW + b
    """
    def __init__(self, in_features, out_features):
        # Xavier initialization
        limit = np.sqrt(6 / (in_features + out_features))
        self.W = np.random.uniform(-limit, limit, (in_features, out_features))
        self.b = np.zeros((1, out_features))
        
        # Gradients
        self.dW = None
        self.db = None
        self.x = None

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, grad_output):
        # x shape: (batch, ..., in_features)
        # grad_output shape: (batch, ..., out_features)
        
        # Flatten leading dimensions for gradient calculation
        x_flat = self.x.reshape(-1, self.x.shape[-1])
        grad_output_flat = grad_output.reshape(-1, grad_output.shape[-1])
        
        # dL/dW = x.T * grad_output
        self.dW = np.dot(x_flat.T, grad_output_flat)
        # dL/db = sum(grad_output, axis=0)
        self.db = np.sum(grad_output_flat, axis=0, keepdims=True)
        
        # dL/dx = grad_output * W.T
        grad_x = np.dot(grad_output, self.W.T)
        return grad_x

class Embedding:
    """
    Token/Position Embedding Layer
    """
    def __init__(self, vocab_size, n_embd):
        self.W = np.random.randn(vocab_size, n_embd) * 0.01
        self.indices = None
        self.dW = None

    def forward(self, indices):
        self.indices = indices
        return self.W[indices]

    def backward(self, grad_output):
        # grad_output: (B, T, D) or (T, D)
        self.dW = np.zeros_like(self.W)
        # Use np.add.at for vectorized accumulation of gradients
        np.add.at(self.dW, self.indices, grad_output)
        return None  # No gradient for indices

class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x > 0)
        return x * self.mask

    def backward(self, grad_output):
        return grad_output * self.mask

class Softmax:
    """
    Numerically stable Softmax layer
    Note: Usually used with CrossEntropy, but here implement stand-alone
    """
    def __init__(self):
        self.output = None

    def forward(self, x, axis=-1):
        # Stable softmax: subtract max
        exps = np.exp(x - np.max(x, axis=axis, keepdims=True))
        self.output = exps / np.sum(exps, axis=axis, keepdims=True)
        return self.output

    def backward(self, grad_output):
        # This is a general softmax backward. 
        # For Transformer cross-entropy, we often combine Softmax + Loss.
        # Here we provide the Jacobian-based backward.
        return self.output * (grad_output - np.sum(grad_output * self.output, axis=-1, keepdims=True))

class Attention:
    """
    Scaled Dot-Product Attention
    """
    def __init__(self, scale):
        self.scale = scale
        self.softmax = Softmax()
        
        # Saved for backward
        self.Q = None
        self.K = None
        self.V = None
        self.attn_weights = None

    def forward(self, Q, K, V, mask=None):
        self.Q, self.K, self.V = Q, K, V
        
        # scores = (Q * K^T) / sqrt(dk)
        # Q: (B, H, T, D), K: (B, H, T, D) -> scores: (B, H, T, T)
        scores = np.matmul(Q, np.swapaxes(K, -1, -2)) * self.scale
        
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
            
        self.attn_weights = self.softmax.forward(scores) # (B, T, T)
        
        # output = attn_weights * V
        # attn_weights: (B, T, T), V: (B, T, D) -> out: (B, T, D)
        out = np.matmul(self.attn_weights, V)
        return out

    def backward(self, grad_out):
        # grad_out: (B, T, D)
        
        # 1. Backprop through MatMul (attn_weights * V)
        # dL/dV = attn_weights^T * grad_out
        dV = np.matmul(np.swapaxes(self.attn_weights, -1, -2), grad_out)
        
        # dL/dattn_weights = grad_out * V^T
        d_attn_weights = np.matmul(grad_out, np.swapaxes(self.V, -1, -2))
        
        # 2. Backprop through Softmax
        d_scores = self.softmax.backward(d_attn_weights)
        
        # 3. Backprop through scaling
        d_scores = d_scores * self.scale
        
        # 4. Backprop through MatMul (Q * K^T)
        # dL/dQ = d_scores * K
        dQ = np.matmul(d_scores, self.K)
        # dL/dK = d_scores^T * Q
        dK = np.matmul(np.swapaxes(d_scores, -1, -2), self.Q)
        
        return dQ, dK, dV

class LayerNorm:
    """
    Layer Normalization: y = gamma * (x - mean) / sqrt(var + eps) + beta
    """
    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.gamma = np.ones((1, 1, dim))
        self.beta = np.zeros((1, 1, dim))
        
        # Saved for backward
        self.x = None
        self.mean = None
        self.var = None
        self.x_hat = None
        
        # Gradients
        self.dgamma = None
        self.dbeta = None

    def forward(self, x):
        self.x = x
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.var = np.var(x, axis=-1, keepdims=True)
        self.x_hat = (x - self.mean) / np.sqrt(self.var + self.eps)
        return self.gamma * self.x_hat + self.beta

    def backward(self, grad_output):
        # grad_output shape: (B, T, D)
        B, T, D = grad_output.shape
        
        self.dgamma = np.sum(grad_output * self.x_hat, axis=(0, 1), keepdims=True)
        self.dbeta = np.sum(grad_output, axis=(0, 1), keepdims=True)
        
        dx_hat = grad_output * self.gamma
        
        # Backward through normalization
        std_inv = 1.0 / np.sqrt(self.var + self.eps)
        
        # dx = (1/D) * std_inv * (D * dx_hat - sum(dx_hat) - x_hat * sum(dx_hat * x_hat))
        dx = (1.0 / D) * std_inv * (
            D * dx_hat - 
            np.sum(dx_hat, axis=-1, keepdims=True) - 
            self.x_hat * np.sum(dx_hat * self.x_hat, axis=-1, keepdims=True)
        )
        return dx
