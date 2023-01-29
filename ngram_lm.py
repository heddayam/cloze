from nltk.util import trigrams, pad_sequence
from nltk.lm.preprocessing import pad_both_ends, flatten, padded_everygram_pipeline
from nltk.lm import MLE
import pandas as pd
import pdb

from datasets import load_dataset
import datasets
from pathlib import Path
CACHE_DIR = '/data/mourad/hf_datasets'

wiki_dataset = load_dataset("wikipedia", "20200501.en", cache_dir=CACHE_DIR)
books_dataset = load_dataset("bookcorpus", cache_dir=CACHE_DIR)
pdb.set_trace()

'''data_df = pd.read_pickle('/data/mourad/provo_data/spacy_and_gpt2_entropy_top_50_data.pkl') 
pdb.set_trace()

text = [['a', 'b', 'c'], ['a', 'c', 'd', 'c', 'e', 'f']]
text = pad_both_ends(text[1], n=3)
pdb.set_trace()
trig = list(trigrams(text))
print(trig)'''

