from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import numpy as np
import pickle
from tqdm import tqdm
import pdb
import re
import random
import os
import openai
import argparse
import pandas as pd

REQUEST_RATE_COUNTER = 0


def read_metaphors(examples=False, alt_pairs = False, ex_with_ppl=False, meta_word_index=False):
    if meta_word_index:
        file_name = 'metanet_scrape/metaphor_word_index.pkl'
    elif examples and not ex_with_ppl:
        file_name = 'metanet_scrape/metaphor_examples.pkl'
    elif examples and ex_with_ppl:
        file_name = f'metanet_scrape/{args.model_type}_metaphor_examples_with_ppl_target_indices.pkl'
    elif alt_pairs:
        file_name = f'metanet_output/{args.model_type}_og_alt_ppl_diffs.pkl'
    else:
        file_name = 'metanet_scrape/metaphors.pkl'

    with open(f'/data/mourad/{file_name}', 'rb') as f:
        data = pickle.load(f)
    return data

def get_concept_pairs(metaphors):
    verb_concept_map = {}
    src_trg_map = {}
    for m in metaphors:
        split = re.split(r'(\s(?:is|are)\s(?:an\s|a\s)?)', m.lower(), flags=re.IGNORECASE)
        src = split[0]
        verb = split[1].strip()
        if verb == 'is an':
            verb = 'is a'
        trg = split[-1]
        if len(trg.split()) > 1: continue
        if src  not in src_trg_map:
            src_trg_map[src] = []
        if verb not in verb_concept_map:
            verb_concept_map[verb] = []
        verb_concept_map[verb].append([src,trg])
        src_trg_map[src].append(trg)
    return verb_concept_map, src_trg_map

def encode_score_tokens(text, indices=None):
    if len(text.split()) < 3: return 0
    global REQUEST_RATE_COUNTER
    global START_TIME
    if args.model_type == 'gpt2':
        tokens_tensor = tokenizer.encode(text.lower(), add_special_tokens=False, return_tensors="pt")
        pdb.set_trace()
        loss=model(tokens_tensor, labels=tokens_tensor)[0]
        ppl = np.exp(loss.cpu().detach().numpy())
    elif args.model_type == 'cedille':
        tokens_tensor = tokenizer.encode(text.lower(), add_special_tokens=False, return_tensors="pt")
        loss=model(tokens_tensor, labels=tokens_tensor)[0]
        ppl = np.exp(loss.cpu().detach().numpy())
        # pdb.set_trace()
    elif args.model_type == 'gpt3':
        # get around 600 request/min openai api limit... 
        if REQUEST_RATE_COUNTER >= 550 and (time.time() - START_TIME) <= 55: 
            REQUEST_RATE_COUNTER = 0
            time.sleep(60 - (time.time() - START_TIME) + 5)
            START_TIME = time.time()
        if time.time() - START_TIME >= 65:
            REQUEST_RATE_COUNTER = 0
            START_TIME = time.time()
        REQUEST_RATE_COUNTER += 1
        response = openai.Completion.create(
                engine="davinci",
                prompt=text,
                max_tokens=3,
                temperature=0.0,
                logprobs=0,
                echo=True,
            )
        pdb.set_trace()
        logprob = response["choices"][0]["logprobs"]
        if indices is not None:
            if logprob['text_offset'][-1] < indices[1]:
                end_token = len(logprob['text_offset'])
            else:
                end_token = logprob['text_offset'].index(indices[1])
            try:
                start_token = logprob['text_offset'].index(logprob['text_offset'][1] if indices[0] == 0 else indices[0]-1)
            except: 
                start_token = logprob['text_offset'].index(logprob['text_offset'][1] if indices[0] == 0 else indices[0]) # account for no space before word
            avg_logprob = np.array(logprob["token_logprobs"])[start_token:end_token].mean()
        else:
            avg_logprob = np.array(logprob["token_logprobs"])[1:].mean()
        ppl = np.exp(-avg_logprob)
    else: print('Model not supported')
    return ppl

def calc_ex_ppl():
    meta_exs = read_metaphors(examples=True) 
    meta_words = read_metaphors(meta_word_index=True)
    for k, v in tqdm(meta_exs.items()):
        if k not in meta_words or len(v) != len(meta_words[k]): break
        example_ppls = []
        for i, ex in enumerate(v):
            if len(ex.split()) > 1:
                word_indices = meta_words[k][i]
                score_ex = encode_score_tokens(ex, indices = word_indices)
            else: 
                score_ex = 0
                print(ex)
            example_ppls.append(score_ex)
        meta_exs[k] = [v, example_ppls]
    with open(f'/data/mourad/metanet_scrape/{args.model_type}_metaphor_examples_with_ppl_target_indices.pkl', 'wb') as f:
        pickle.dump(meta_exs, f)


def pairwise_target_comparison():
    og_alt_meta_pairs = []
    texts = read_metaphors()
    verb_concept_map, src_trg_map = get_concept_pairs(texts)
    collection = []
    all_alt_higher_ppl = 0
    output_file = open(f'/data/mourad/metanet_output/{args.model_type}-alt-metaphors-scores.txt', 'w')
    for verb, concepts in tqdm(verb_concept_map.items()):
        for concept in tqdm(concepts):
            if verb == 'is a' and concept[1][0] in ['a', 'e', 'i', 'o', 'u']:
                    verb_mod = 'is an'
            else: verb_mod = verb
            original = f'{concept[0]} {verb_mod} {concept[1]}'
            score_og = encode_score_tokens(original)
            output_file.write(f'{score_og} - {original}\n')
            higher_ppl_ctr = 0
            prev_choices = []
            for i in range(3):
                choice = concept
                timeout_ctr = 0
                while (choice[1] in src_trg_map[concept[0]] or choice[1] in prev_choices) and timeout_ctr < 5:
                    timeout_ctr += 1
                    if verb in ['are a', 'are an']:
                        search_verb = 'is a'
                    else: search_verb = verb
                    choice = random.choice(verb_concept_map[search_verb])
                if choice == concept:
                    print('timedout search for alternatives')
                    continue
                else: prev_choices.append(choice[1])
                if verb == 'is a' and choice[1][0] in ['a', 'e', 'i', 'o', 'u']:
                    verb_mod = 'is an'
                else: verb_mod = verb
                alt = f'{concept[0]} {verb_mod} {choice[1]}'
                score_alt = encode_score_tokens(alt)
                output_file.write(f'{score_alt} - {alt}\n')
                og_alt_meta_pairs.append({
                        'ppl-diffs': score_alt-score_og, 
                        'ppl-og': f'{score_og:.1f}',
                        'og-metaphor': original,
                        'ppl-alt': f'{score_alt:.1f}',
                        'alt-metaphor': alt
                    })
                if score_alt > score_og:
                    higher_ppl_ctr += 1
            output_file.write(f'{higher_ppl_ctr} alternatives with higher perplexity than original.\n')
            output_file.write('\n')
            if higher_ppl_ctr == 3: all_alt_higher_ppl += 1
    output_file.write(f'{all_alt_higher_ppl} metaphors where all alternatives scored higher perplexity than original.')
    output_file.close()
    with open(f'/data/mourad/metanet_output/{args.model_type}_og_alt_ppl_diffs_target_indices.pkl', 'wb') as f:
        pickle.dump(og_alt_meta_pairs, f)

def score_sort_metaphors():
    texts = read_metaphors()
    metaphor_scores = []
    for text in tqdm(texts):
        metaphor_scores.append(encode_score_tokens(text))
    with open(f'/data/mourad/metanet_output/{args.model_type}_metaphor_ppl_scores.pkl', 'wb') as f:
        pickle.dump({'ppl': metaphor_scores, 'metaphor': texts}, f)
    metaphor_scores_idx = np.argsort(np.array(metaphor_scores))
    #for i in metaphor_scores_idx:
    #    print (texts[i], metaphor_scores[i])

def score_sort_meta_exs(meta_first = False):
    alt_pairs = read_metaphors(alt_pairs=True)
    meta_exs = read_metaphors(examples=True, ex_with_ppl=True) 
    meta_words = read_metaphors(meta_word_index=True)
    pairs = []
    #c=0
    if meta_first:
        meta_ex_order = "metafirst_"
    else:
        meta_ex_order = ""
    output_file = open(f'/data/mourad/metanet_output/{args.model_type}-{meta_ex_order}alt-example-meta-scores.txt', 'w')
    for pair in tqdm(alt_pairs):
        try:
            examples = meta_exs[pair['og-metaphor'].upper()]
            if not examples[0]: continue
            if pair['og-metaphor'].upper() not in meta_words or len(examples[0]) != len(meta_words[pair['og-metaphor'].upper()]): continue
        except: continue
        for i, example in enumerate(list(zip(examples[0], examples[1]))):
            meta_word_idx = meta_words[pair['og-metaphor'].upper()][i]
            example_ppl = example[1]
            example = example[0]
            if example == '': continue
            if example[-1] not in ['.', '!', '?']:
                example += '.'
            og_meta = pair['og-metaphor'].capitalize()
            alt_meta = pair['alt-metaphor'].capitalize()
            if meta_first:
                og_instance = f"{og_meta}. {example}"
                alt_instance = f"{alt_meta}. {example}"
                og_meta_word_idx = [meta_word_idx[0] + len(og_meta)+2, meta_word_idx[1] + len(og_meta)+2]
                alt_meta_word_idx = [meta_word_idx[0] + len(alt_meta)+2, meta_word_idx[1] + len(alt_meta)+2]
            else:
                og_instance = f"{example} {og_meta}."
                alt_instance = f"{example} {alt_meta}."
            score_og = encode_score_tokens(og_instance, indices=og_meta_word_idx)
            score_alt = encode_score_tokens(alt_instance, indices=alt_meta_word_idx)
            if meta_first:
                diff_sign_flip = 1 if score_og < example_ppl else 0  
            else:
                diff_sign_flip = 1 if np.sign(score_alt-score_og) != np.sign(pair['ppl-diffs']) else 0
            record = {
                    'diff-sign-flip': diff_sign_flip,
                    'ppl-diff': score_alt-score_og,
                    'ppl-og': f'{score_og:.1f}',
                    'og-instance': f'{og_instance[:og_meta_word_idx[0]]}[{og_instance[og_meta_word_idx[0]:og_meta_word_idx[1]]}]{og_instance[og_meta_word_idx[1]:]}',
                    'ppl-alt': f'{score_alt:.1f}',
                    'alt-instance': f'{alt_instance[:alt_meta_word_idx[0]]}[{alt_instance[alt_meta_word_idx[0]:alt_meta_word_idx[1]]}]{alt_instance[alt_meta_word_idx[1]:]}'
                }
            output_file.write(f"{record['diff-sign-flip']} - {record['ppl-diff']}\n({record['ppl-og']}) {record['og-instance']}\n({record['ppl-alt']}) {record['alt-instance']}\n\n")
            pairs.append(record)
    output_file.close()
    pdb.set_trace()
    with open(f'/data/mourad/metanet_output/{args.model_type}_{meta_ex_order}og_alt_exs_ppl_diffs_target_indices.pkl', 'wb') as f:
        pickle.dump(pairs, f)

def score_translator_diffs():
    df = pd.read_csv('/data/mourad/en-fr/enfr_headlines_translators.tsv', sep='\t')#.head(10)
    with open("/data/mourad/en-fr/t5large-translations.pkl", 'rb') as f:
        mt = pickle.load(f)
    df['mt'] = mt[:len(df)]
    df = df.dropna()
    # pdb.set_trace()
    df['fr_ppl'] = df.fr.apply(encode_score_tokens)
    df.to_csv('/data/mourad/en-fr/enfr_headlines_translators_ppl_cedille.tsv', sep='\t', index=False)
    df['mt_ppl'] = df.mt.apply(encode_score_tokens)
    df.to_csv('/data/mourad/en-fr/enfr_headlines_translators_ppl_cedille.tsv', sep='\t', index=False)
    # df['en_ppl'] = df.en.apply(encode_score_tokens)
    # df.to_csv('/data/mourad/en-fr/enfr_headlines_translators_ppl_en.tsv', sep='\t', index=False)
    pdb.set_trace()

    '''meta_ex = []
    metaphor_diff_scores = []

    for meta, exs in tqdm(meta_exs.items()):
        #c += 1
        #if c> 3: break
        metaphor_score = encode_score_tokens(meta)
        for ex in exs:
            if ex != '':
                ex_score = encode_score_tokens(ext)
                meta_ex.append([metaphor_score, ex_score,meta, ex])
                metaphor_diff_scores.append(metaphor_score-ex_score)
        #CHECK OTHER STATISTICS    
        #ex_score = np.mean(np.array(ex_scores))
        #dif_score = abs(metaphor_score-ex_score)
        #metaphor_scores.append(dif_score)
        #print(f"{metaphor_score} - {dif_score} - {meta}")
    
    metaphor_diff_scores_idx = np.argsort(np.array(metaphor_diff_scores))

    output_file = open('/data/mourad/metanet_output/gpt2-scores.txt', 'w')
    for i in metaphor_diff_scores_idx:
        diff_score = "{:.2f}".format(metaphor_diff_scores[i])
        output_file.write(f"{diff_score} {meta_ex[i][-2]} {meta_ex[i][-1]} {meta_ex[i][:2]}\n")
        #print(metaphor_diff_scores[i], meta_ex[i][-2], meta_ex[i][-1], meta_ex[i][:2]) 
    output_file.close()    '''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_type', choices=['gpt2', 'gpt3', 'cedille'], required=True)
    args = parser.parse_args()
    
    if args.model_type == "gpt2":
        model = GPT2LMHeadModel.from_pretrained('gpt2').to('cuda:0')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2').to('cuda:0')
    elif args.model_type == 'cedille':
        tokenizer = AutoTokenizer.from_pretrained("Cedille/fr-boris")
        model = AutoModelForCausalLM.from_pretrained("Cedille/fr-boris")
    else:
        openai.api_key = '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'
    START_TIME = time.time()
    
    score_translator_diffs()
    #calc_ex_ppl()
    # score_sort_meta_exs(meta_first=True)
    #score_sort_metaphors()
    #pairwise_target_comparison()
