import torch
import pdb
from transformers import AutoTokenizer
# from torch import nn
import numpy as np
import spacy
from tokenizers import ByteLevelBPETokenizer
from tokenizers.models import BPE

class GPT2Tokenizer:
    def __init__(self, vocab,  lowercase=False):
        #self._tokenizer = ByteLevelBPETokenizer(vocab_dict, lowercase=lowercase)
        self.vocab = vocab
        self._tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def __call__(self, text):
        tokens = self._tokenizer.tokenize(text) 
        words = []
        spaces = []
        pdb.set_trace()
        for i, (text, (start, end)) in enumerate(zip(tokens.tokens, tokens.offsets)):
            words.append(text)
            if i < len(tokens.tokens) - 1:
                # If next start != current end we assume a space in between
                next_start, next_end = tokens.offsets[i + 1]
                spaces.append(next_start > end)
            else:
                spaces.append(True)
        return Doc(self.vocab, words=words, spaces=spaces)


print("\nBegin next-word using HF GPT-2 demo ")

#toker = AutoTokenizer.from_pretrained("gpt2")
nlp = spacy.load("en_core_web_sm")
nlp.tokenizer = GPT2Tokenizer(nlp.vocab)

text = 'There are now rumblings that apple might enter the smart watch spacy'
#nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
pdb.set_trace()
