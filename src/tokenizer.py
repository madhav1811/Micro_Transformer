import json
import os

class CharacterTokenizer:
    """
    A simple character-level tokenizer that maps unique characters to integers.
    Includes an <UNK> token for unknown characters.
    """
    def __init__(self, text=None, chars=None):
        if chars is not None:
            self.chars = chars
        elif text is not None:
            # Find all unique characters in the text
            self.chars = sorted(list(set(text)))
        else:
            self.chars = []
            
        # Ensure we have a consistent set of characters and an UNK token
        # We'll use index 0 for unknown characters
        if '<UNK>' not in self.chars:
            self.chars = ['<UNK>'] + self.chars
            
        self.vocab_size = len(self.chars)
        
        # Create mapping dictionaries
        self.stoi = { ch:i for i,ch in enumerate(self.chars) }
        self.itos = { i:ch for i,ch in enumerate(self.chars) }
        
    def encode(self, s):
        """String to list of integers, mapping unseen chars to <UNK>"""
        return [self.stoi.get(c, 0) for i, c in enumerate(s)]
        
    def decode(self, l):
        """List of integers to string"""
        return ''.join([self.itos.get(i, '<UNK>') for i in l])

    def save_vocab(self, path):
        """Save character list to a JSON file"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.chars, f)

    @classmethod
    def load_vocab(cls, path):
        """Load character list from a JSON file and return a tokenizer"""
        with open(path, 'r', encoding='utf-8') as f:
            chars = json.load(f)
        return cls(chars=chars)

if __name__ == "__main__":
    # Test the tokenizer
    sample_text = "hello"
    tokenizer = CharacterTokenizer(sample_text)
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Characters: {repr(''.join(tokenizer.chars))}")
    
    encoded = tokenizer.encode("hello world") # 'world' has new characters
    print(f"Encoded 'hello world': {encoded}")
    print(f"Decoded: {tokenizer.decode(encoded)}")
