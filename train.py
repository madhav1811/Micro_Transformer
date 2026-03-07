import numpy as np
import os
import csv
from src.tokenizer import CharacterTokenizer
from src.model import MicroGPT
from src.optimizer import AdamW
from src.utils import cross_entropy_loss, generate

# Hyperparameters (Accuracy Focus)
batch_size = 32
block_size = 256
n_embd = 384
n_heads = 12
n_layers = 8
max_lr = 5e-4
min_lr = 1e-5
max_iters = 10000
eval_interval = 100
print_interval = 10

# 1. Load Data (CSV Integration)
data_path = "C:\\Users\\HP\\Downloads\\Micro_Transformer\\data\\train_data.csv"
if not os.path.exists(data_path):
    print(f"Error: {data_path} not found.")
    exit()

texts = []
with open(data_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # We use the completion column as it contains both prompt and story
        texts.append(row['completion'])

# Join all stories with a separator to help the model distinguish them
text = "\n\n".join(texts)

# 2. Tokenizer
tokenizer = CharacterTokenizer(text)
vocab_size = tokenizer.vocab_size
data = np.array(tokenizer.encode(text))

# 3. Model & Optimizer
model = MicroGPT(vocab_size, n_embd, n_heads, n_layers, block_size)
optimizer = AdamW(model.get_params(), lr=max_lr)

def get_batch():
    ix = np.random.randint(0, len(data) - block_size, (batch_size,))
    x = np.stack([data[i:i+block_size] for i in ix])
    y = np.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

print(f"Starting training on {len(text)} characters...")
print(f"Vocab size: {vocab_size}")

# 4. Training Loop
for iter in range(max_iters):
    # Linear LR decay
    lr = max_lr - (max_lr - min_lr) * (iter / max_iters)
    optimizer.lr = lr
    
    # Sample batch
    xb, yb = get_batch()
    
    # Forward pass
    logits = model.forward(xb)
    loss, grad_logits, accuracy = cross_entropy_loss(logits, yb)
    
    # Backward pass
    optimizer.zero_grad()
    model.backward(grad_logits)
    
    # Update weights
    optimizer.step()
    
    # Logging
    if iter % eval_interval == 0:
        print(f"\nStep {iter}: Loss {loss:.4f}, Accuracy {accuracy*100:.2f}%, LR {lr:.2e}")
    elif iter % print_interval == 0:
        print(".", end="", flush=True)

# Save the model and vocabulary
print("\nSaving model weights and vocabulary...")
model.save_weights('model_weights.npz')
tokenizer.save_vocab('vocab.json')

# 5. Final Generation
print("\nFinal Generation:")
prompt = "Once upon a time"
result = generate(model, tokenizer, prompt, max_new_tokens=1000, block_size=block_size)
print(result)
#print(Hello)