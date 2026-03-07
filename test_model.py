import numpy as np
import os
from src.tokenizer import CharacterTokenizer
from src.model import MicroGPT
from src.utils import generate, evaluate

# 1. Load Tokenizer from Saved Vocab
vocab_path = 'vocab.json'
if os.path.exists(vocab_path):
    print(f"Loading vocabulary from {vocab_path}...")
    tokenizer = CharacterTokenizer.load_vocab(vocab_path)
    # Load sample text for evaluation context
    data_path = 'data/sample.txt'
    if os.path.exists(data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        text = "" 
else:
    print(f"Warning: {vocab_path} not found. Inferring from data/sample.txt.")
    data_path = 'data/sample.txt'
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    tokenizer = CharacterTokenizer(text)

vocab_size = tokenizer.vocab_size
data = np.array(tokenizer.encode(text)) if text else None

# 2. Hyperparameters (MUST match train.py)
block_size = 256
n_embd = 384
n_heads = 12
n_layers = 8

# 3. Initialize Model
model = MicroGPT(vocab_size, n_embd, n_heads, n_layers, block_size)

# 4. Load Weights
weights_path = 'model_weights.npz'
if os.path.exists(weights_path):
    print(f"Loading weights from {weights_path}...")
    model.load_weights(weights_path)
else:
    print(f"Warning: {weights_path} not found. Running with random weights.")

# 5. Evaluate Accuracy
if data is not None:
    print("\n--- Evaluating Model Accuracy ---")
    loss, acc = evaluate(model, data, block_size, batch_size=32, max_batches=20)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {acc*100:.2f}%")

# 6. Run Generation Tests
print("\n--- Running Story Generation ---")
T = 0.8
print(f"Using Sampling Temperature: {T}")

prompts = ["Once upon a time", "A brave robot", "In the forest"]

for prompt in prompts:
    print(f"\nPrompt: '{prompt}'")
    try:
        result = generate(model, tokenizer, prompt, max_new_tokens=1000, block_size=block_size, temperature=T)
        print(f"Story: {result}")
    except Exception as e:
        print(f"Error generating story: {e}")

print("\nGeneration complete.")

print("\nTests complete.")
