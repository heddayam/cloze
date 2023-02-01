from nltk.tree import Tree
from nltk.internals import find_jars_within_path
from nltk.parse.stanford import StanfordParser
import pdb
import pandas as pd
import swifter
import benepar, spacy
import sys
import nltk
#from datasets import load_dataset
#import datasets
from pathlib import Path
from tqdm.auto import tqdm
from collections import Counter
from nltk.tree import Tree
from nltk import Nonterminal
from nltk.grammar import induce_pcfg
from nltk import PCFG
from nltk.parse import ViterbiParser
from nltk.tree.probabilistic import ProbabilisticTree
from multiprocessing import Pool
import os
from glob import glob
import torch
import argparse
import random

def parallel_parse(f_start,n_cpus, step):
    print("CPU COUNT =", n_cpus)
    files = glob('/data/mourad/kenlm/preprocessed/lines_split/*')
    #files = files[f_start:f_start+n_cpus]
    files = files[f_start:f_start+step]
    files = random.sample(files, n_cpus)
    with Pool(n_cpus) as p:
        p.map(parse_data, files)

def parse_data(in_file):
    #f = open('/data/mourad/kenlm/preprocessed/100k_0003_sample.txt', 'r')
    benepar.download('benepar_en3')
    nlp = spacy.load('en_core_web_md')
    if spacy.__version__.startswith('2'):
        nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
    else:
        nlp.add_pipe("benepar", config={"model": "benepar_en3"})
    f = open(in_file, 'r')
    out_file = in_file.split('_')[-1]
    out_f = open(f'/data/mourad/pcfg/parsed/segments/{out_file}.txt', 'w')
    for line in tqdm(f):
        line = line.lower()
        doc = nlp(line)
        sent = list(doc.sents)[0]
        if len(sent) > 512:
            continue
        try:
            parse = sent._.parse_string
        except: continue
        out_f.write(parse + "\n")
    f.close()
    out_f.close()

def abstract_pcfg():
    g = '/data/mourad/pcfg/ptb_all.pcfg'
    grammar = PCFG(g)
    pdb.set_trace()

def estimate_pcfg():
    prods = []
    #f = open('/data/mourad/pcfg/parsed/100k.txt', 'r')
    f = open('/data/mourad/pcfg/parsed/all.txt', 'r')
    for parse in tqdm(f):
        try:
            t = Tree.fromstring(parse)
            prods += t.productions()
        except:
            continue

    grammar = induce_pcfg(Nonterminal('S'), prods)
    f = open('/data/mourad/pcfg/induced_pcfg_250k.txt', 'w')
    for prod in grammar.productions():
        mod = prod 
        f.write(str(prod)+'\n')
    return grammar

def score_prompts(pcfg):
    #prods = open('/data/mourad/pcfg/induced_pcfg_250k.txt', 'r').readlines()
    #pcfg = induce_pcfg(Nonterminal('S'), prods)
    pdb.set_trace()
    prompts = open('/data/mourad/kenlm/preprocessed/provo_prompts.txt', 'r').readlines()
    preds = pd.read_pickle('/data/mourad/pcfg/target_df.pkl')
    #preds = pd.read_csv('/data/mourad/pcfg/target_df.tsv', sep='\t')
    preds['prompt'] = preds.apply(lambda x: x.text_spacy[:x.i], axis=1)
    preds['prompt'] = preds.apply(lambda x: str(x.prompt).strip() + f' {x.text}', axis=1)
    vit = ViterbiParser(pcfg)
    def get_prob(seq):
        parse = vit.parse(seq.split())
        if not isinstance(parse, ProbabilisticTree):
            return parse
        prob = parse.prob()
        return prob
    pdb.set_trace()
    preds = preds.sample(n=100, random_state=42)
    preds['score'] = preds.prompt.apply(get_prob)
    #preds['score'] = preds.prompt.swifter.apply(lambda x: list(vit.parse(x.split()))[0].prob())
    #preds.text_spacy = preds.text_spacy.astype(str)
    #vit.parse('Hello I am a person')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--n_cpus', type=int, default=1)
    parser.add_argument('--step', type=int, default=1)
    args = parser.parse_args()
    
    #abstract_pcfg()
    #torch.multiprocessing.set_start_method('spawn')
    #parallel_parse(args.start, args.n_cpus, args.step)
    #parse_data()
    pcfg = estimate_pcfg()
    pdb.set_trace()
    #pcfg=None
    score_prompts(pcfg)
