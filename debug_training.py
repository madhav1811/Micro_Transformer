import numpy as np
import os
import csv
from src.tokenizer import CharacterTokenizer
from src.model import MicroGPT
from src.optimizer import AdamW
from src.utils import cross_entropy_loss

# Hyperparameters (Reduced for debugging)
batch_size = 8
block_size = 64
n_embd = 128
n_heads = 4
n_layers = 4
max_lr = 1e-3
max_iters = 50

# Load Data
data_path = "C:\\Users\\HP\\Downloads\\Micro_Transformer\\data\\train_data.csv"
texts = []
with open(data_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        texts.append(row['completion'])
text = "\n\n".join(texts)

tokenizer = CharacterTokenizer(text)
vocab_size = tokenizer.vocab_size
data = np.array(tokenizer.encode(text))

model = MicroGPT(vocab_size, n_embd, n_heads, n_layers, block_size)
optimizer = AdamW(model.get_params(), lr=max_lr)

def get_batch():
    ix = np.random.randint(0, len(data) - block_size, (batch_size,))
    x = np.stack([data[i:i+block_size] for i in ix])
    y = np.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

print(f"Starting debug training...")
for iter in range(max_iters):
    xb, yb = get_batch()
    logits = model.forward(xb)
    loss, grad_logits, accuracy = cross_entropy_loss(logits, yb)
    
    optimizer.zero_grad()
    model.backward(grad_logits)
    optimizer.step()
    
    if iter % 10 == 0:
        print(f"Iter {iter}: Loss {loss:.4f}, Accuracy {accuracy*100:.2f}%")

if np.isnan(loss):
    print("FAILED: Loss is NaN")
elif loss < 4.0: # Arbitrary threshold to see if it's decreasing
    print("SUCCESS: Loss is decreasing")
else:
    print("WARNING: Loss might be stagnant")
