from tokenizer import Tokenizer

with open('data.txt', 'r', encoding='utf-8', errors='ignore') as f:
    text = f.read()

tokenizer = Tokenizer({"<|endoftext|>": 50257, "<|padding|>": 50258})

tokenizer.training(text, 50000)
tokenizer.save("tokenizer.json")