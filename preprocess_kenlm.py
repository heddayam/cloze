import sys
import nltk
from datasets import load_dataset
import datasets
from pathlib import Path
import pdb
from tqdm.auto import tqdm
CACHE_DIR = '/data/mourad/hf_datasets'
news_dataset = load_dataset("cc_news", cache_dir=CACHE_DIR)
wiki_dataset = load_dataset("wikipedia", "20200501.en", cache_dir=CACHE_DIR)
books_dataset = load_dataset("bookcorpus", cache_dir=CACHE_DIR)

f = open('/data/mourad/kenlm/preprocessed/lines.txt', 'w')

dataset_objs = [news_dataset, wiki_dataset, books_dataset]
for ds in dataset_objs:
    for ex in tqdm(ds['train']):
        text = ex['text']
        f.write(text + '\n')


'''for ex in tqdm(wiki_dataset['train']):
    cnt += 1
    text = ex['text']
    f.write(text + '\n')
    #for sentence in nltk.sent_tokenize(text):
        #f.write(' '.join(nltk.word_tokenize(sentence)).lower() + '\n')
    #    print(' '.join(nltk.word_tokenize(sentence)).lower())

cnt=0
for ex in tqdm(books_dataset['train']):
    cnt += 1
    text = ex['text']
    f.write(text + '\n')
    #for sentence in nltk.sent_tokenize(text):
        #f.write(' '.join(nltk.word_tokenize(sentence)).lower() + '\n')
    #    print(' '.join(nltk.word_tokenize(sentence)).lower())'''

