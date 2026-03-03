import numpy as np

def cross_entropy_loss(logits, targets):
    """
    logits: (B, T, V)
    targets: (B, T)
    returns: loss, grad_logits, accuracy
    """
    B, T, V = logits.shape
    logits_flat = logits.reshape(-1, V)
    targets_flat = targets.reshape(-1)
    
    # Stable Softmax
    probs = np.exp(logits_flat - np.max(logits_flat, axis=-1, keepdims=True))
    probs /= np.sum(probs, axis=-1, keepdims=True)
    
    # Loss: -log(p_target)
    indices = np.arange(B * T)
    loss = -np.mean(np.log(probs[indices, targets_flat] + 1e-9))
    
    # Accuracy: Percentage of correctly predicted tokens
    predictions = np.argmax(logits_flat, axis=-1)
    accuracy = np.mean(predictions == targets_flat)
    
    # Gradient: p - y
    grad_logits = probs.copy()
    grad_logits[indices, targets_flat] -= 1
    grad_logits /= (B * T) # Normalize by number of elements
    
    return loss, grad_logits.reshape(B, T, V), accuracy

def evaluate(model, data, block_size, batch_size=32, max_batches=10):
    """
    Evaluate the model on a dataset and return average loss and accuracy.
    """
    losses = []
    accuracies = []
    
    for _ in range(max_batches):
        # Sample random batch
        ix = np.random.randint(0, len(data) - block_size, (batch_size,))
        x = np.stack([data[i:i+block_size] for i in ix])
        y = np.stack([data[i+1:i+block_size+1] for i in ix])
        
        logits = model.forward(x)
        loss, _, acc = cross_entropy_loss(logits, y)
        losses.append(loss)
        accuracies.append(acc)
        
    return np.mean(losses), np.mean(accuracies)

def generate(model, tokenizer, prompt, max_new_tokens, block_size, temperature=1.0):
    """
    Simple generation loop with temperature sampling
    """
    idx = np.array([tokenizer.encode(prompt)])
    
    for _ in range(max_new_tokens):
        # Crop context
        idx_cond = idx[:, -block_size:]
        # Forward pass
        logits = model.forward(idx_cond)
        # Focus on the last time step
        logits = logits[:, -1, :]
        
        # Apply temperature
        logits = logits / max(temperature, 1e-5)
        
        # Softmax to get probs
        probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs /= np.sum(probs, axis=-1, keepdims=True)
        # Sample
        next_idx = np.random.choice(len(probs[0]), p=probs[0])
        # Append to sequence
        idx = np.concatenate((idx, [[next_idx]]), axis=1)
        
    return tokenizer.decode(idx[0].tolist())
