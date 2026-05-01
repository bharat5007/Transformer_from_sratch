from tokenizer import Tokenizer

import os, glob

text = ""
for path in sorted(glob.glob("../dataset/*.txt")):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        text += f.read()

len(text)
tokenizer = Tokenizer({"<|endoftext|>": 50257, "<|padding|>": 50258})
tokenizer.training(text, 50000)