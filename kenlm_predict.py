import pdb
import json
import pandas as pd
import swifter
import re
import kenlm
import numpy as np
import math
import argparse
#from prekenlm.model import KenlmModel


class TrigramPredict:
    def __init__(self, model):
        self.load_vocab()
        if model == 'hf':
            self.model = KenlmModel.from_pretrained("oscar", "en")
            self.score = self.model.get_perplexity
        elif model == 'kenlm':
            # /data/mourad/kenlm/trigram.binary
            self.model = kenlm.Model('/data/mourad/kenlm/trigram_wiki-web-books.binary')
            #self.score = self.model.perplexity

    def load_vocab(self):
        #with open('gpt2-vocab.json')
        #vocab = pd.read_csv('/data/mourad/kenlm/oscar/en.sp.vocab', sep='\t', header=None)
        #vocab.columns = ['word', 'stat']
        #self.vocab = vocab.word.tolist()
        vocab = open('/data/mourad/kenlm/ngram-vocab.txt').readlines()
        self.vocab = [word for word in vocab if re.match('^\w+$', word) is not None]
        '''vocab.remove('.\n')
        vocab.remove('?\n')
        vocab.remove('...\n')
        vocab.remove('!\n')'''
    
    def get_perplexity(self, text):
        words = len(text.split())
        return 10.0**(-self.model.score(text, eos=False) / words)

    def get_all_surprisals(self,prompt):
        #if len(prompt.split()) < 3: return ''
        #prompts = [f"{prompt} {word}" for word in vocab[:10000] if re.match('^\w+$', word) is not None]
        best_ppl = 1e6
        best_word = ''
        for word in self.vocab:
            #if re.match('^\w+$', word) is not None:
            text = f"{prompt} {word}"
            #for score in self.model.full_scores(text):
            #if len(text.split()) > 3: pdb.set_trace()
            #pdb.set_trace()
            #ppl = self.score(text)
            ppl = self.get_perplexity(text)
            #pdb.set_trace()
            #score = model.score(text)
            #score = model.full_scores(text)
            #score_adj = [math.pow(10.0, score) for score, _ in score]
            #product_inv_prob = np.prod(score_adj)
            #n = len(score_adj)
            #perplexity = math.pow(product_inv_prob, 1.0/n)
            if ppl < best_ppl:
                best_ppl = ppl
                best_word = word
        #pred = res.groupby('sentence_id').nth(-2).sort_values(by='surprisal').iloc[0]
        return f"{best_word.lower().strip()}_{best_ppl}"

    def run_predict(self, model_str):
        df = pd.read_csv('/data/mourad/provo_data/left_context_for_pred.tsv', sep='\t')
        tok_df = pd.read_csv('/data/mourad/kenlm/preprocessed/provo_prompts_tokenized.txt', sep='\t', header=None)
        tok_df.columns = ['prompt']
        df.prompt = tok_df.prompt
        df['surprisals'] = df.prompt.swifter.apply(self.get_all_surprisals)
        df[['kenlm', 'kenlm_prob']] = df.surprisals.str.split('_', expand=True)
        df.kenlm_prob = df.kenlm_prob.astype(float)
        df = df[['Word_Unique_ID', 'kenlm', 'kenlm_prob']]
        df = df.rename({'kenlm': 'trigram_pred'}, axis=1)
        df.to_csv(f'/data/mourad/provo_data/{model_str}_preds.tsv', sep='\t', index=False)
        pdb.set_trace()
        #lm_zoo.get_surprisals(model, ['my name is jack'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['kenlm', 'hf'], default='kenlm')
    args = parser.parse_args()

    predictor = TrigramPredict(args.model)
    predictor.run_predict(args.model)
