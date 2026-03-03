import numpy as np

class AdamW:
    """
    AdamW optimizer from scratch
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.params = params
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        
        # Initialize momentums
        self.states = []
        for p in params:
            state = {}
            # Standardize which parameters we handle
            for attr in ['W', 'b', 'gamma', 'beta']:
                if hasattr(p, attr):
                    state['m' + attr] = np.zeros_like(getattr(p, attr))
                    state['v' + attr] = np.zeros_like(getattr(p, attr))
            self.states.append(state)

    def step(self):
        self.t += 1
        b1, b2 = self.betas
        
        for i, p in enumerate(self.params):
            state = self.states[i]
            
            for attr in ['W', 'b', 'gamma', 'beta']:
                if hasattr(p, attr):
                    param = getattr(p, attr)
                    grad = getattr(p, 'd' + attr)
                    
                    # 1. Weight Decay
                    param -= self.lr * self.weight_decay * param
                    
                    # 2. Adam update
                    state['m' + attr] = b1 * state['m' + attr] + (1 - b1) * grad
                    state['v' + attr] = b2 * state['v' + attr] + (1 - b2) * (grad**2)
                    
                    m_hat = state['m' + attr] / (1 - b1**self.t)
                    v_hat = state['v' + attr] / (1 - b2**self.t)
                    
                    param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
                
    def zero_grad(self):
        for p in self.params:
            for attr in ['W', 'b', 'gamma', 'beta']:
                if hasattr(p, 'd' + attr):
                    setattr(p, 'd' + attr, np.zeros_like(getattr(p, attr)))
