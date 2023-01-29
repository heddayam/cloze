from calendar import c
from pickle import FALSE
import pandas as pd
# import plotly.express as px
import pdb
import spacy
from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex
import swifter
from transformers import pipeline, set_seed, AutoTokenizer, AutoModelForCausalLM
import plotly.express as px
import plotly.graph_objs as go
import scipy.stats as ss
import seaborn as sns
from nltk.corpus import wordnet as wn
from tqdm import tqdm
from torch.nn.functional import softmax
from provo_torch_dataset import ProvoDataset
import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse

DATA_DIR = '/data/mourad/provo_data/plots/'

def load_data():
    cols_to_keep = [
        'Word_Unique_ID',
        'Text_ID',
        'Word_Cleaned',
        'OrthographicMatch',
        'OrthoMatchModel',
        'IsModalResponse',
        'ModalResponse',
        'ModalResponseCount',
        'Certainty',
        'POS_CLAWS',
        'Word_Content_Or_Function',
        'Word_POS',
        'POSMatch',
        'POSMatchModel',
        'InflectionMatch',
        'InflectionMatchModel',
        'LSA_Context_Score',
        'LSA_Response_Match_Score'
    ]
    df_eye = pd.read_csv('/data/mourad/provo_data/Provo_Corpus-Eyetracking_Data.csv', usecols=cols_to_keep)
    df_cloze = pd.read_csv('/data/mourad/provo_data/Provo_Corpus-Predictability_Norms.csv', encoding = "ISO-8859-1")
    df_cloze = df_cloze.replace(
        df_cloze[df_cloze.Text.str.contains("Very similar")].Text.unique()[0], 
        df_cloze[df_cloze.Text.str.contains("Very similar")].Text.unique()[1]
        )
    tokenized_prompts = pd.read_csv('/data/mourad/kenlm/preprocessed/provo_prompts_tokenized.txt', sep='\t', header=None)
    left_df = pd.read_csv('/data/mourad/provo_data/left_context_for_pred.tsv', sep='\t')
    tokenized_prompts.columns = ['prompt']
    tokenized_prompts['Word_Unique_ID'] = left_df.Word_Unique_ID
    return df_eye, df_cloze, tokenized_prompts

def match_word(x):
    current_index = x.Word_Number-1
    match = x.text_spacy[current_index] 
    offset = 0
    while match.text.lower() != x.Word and (current_index + offset) < len(x.text_spacy):
        offset += 1
        try:
            match = x.text_spacy[current_index + offset]
            if match.text.lower() == x.Word: break
        except: continue
    return {'target_text': match.text.lower(), 'target_pos':x.text_spacy[match.i].pos_, 'target_i':match.i}

def replace_word_with_response(x):
    og = x.text_spacy
    target_idx = x.target_i
    start_idx = 0
    sent_idx = 0
    for i, s in enumerate(og.sents):
        if target_idx >= s.start and target_idx < s.end:
            start_idx = s.start
            sent = s
            sent_idx = i
            break
    adjusted_idx = target_idx - start_idx
    
    spacy_list = [i for t in sent for i in (t.norm_, " ")]
    spacy_list[adjusted_idx * 2] = x.Response
    text_joined = "".join(spacy_list).capitalize()
    new_spacy = nlp(text_joined)
    if len(sent) != len(new_spacy) and x.Response != "can't":
        return {}
    if x.Response == "can't":
        human_text = "can"
        adjusted_idx -= 1
    else:
        human_text = new_spacy[adjusted_idx].text
    return {'human_text': human_text, 'human_pos': new_spacy[adjusted_idx].pos_, 'human_lemma': new_spacy[adjusted_idx].lemma_}

def get_gpt3_completion_logits(seqs):
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
            max_tokens=0,
            temperature=0.0,
            logprobs=0,
            echo=True,
        )

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


def get_lm_completion_logits(seqs):
    pred_words = []
    entropies = []
    top_50_entropies = []
    top_50_pred_words = []
    device = torch.device('cuda:0')
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token_id
    tokenizer.return_special_tokens_mask = True
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    model.to(device) #cpu
    dataloader = DataLoader(ProvoDataset(seqs), batch_size=1, shuffle=False)
    for step, batch in tqdm(enumerate(dataloader)):
        encoded_input = tokenizer(batch, return_tensors='pt', padding=False)
        encoded_input.to(device)
        output = model(**encoded_input)
                    #attention_mask = encoded_input['attention_mask'],
                    #max_new_tokens=10, 
                    #return_dict_in_generate=True, 
                    #output_scores=True, 
                    #temperature=1)
        logits = output.logits[:, -1, :]
        logits = logits.squeeze(0)
        sorted_logits, sorted_preds = logits.sort(descending=True)#[:50]
        sorted_logits = sorted_logits[:50]
        sorted_preds = sorted_preds[:50]
        pred_id = logits.argmax().item()
        pred_word = tokenizer.decode(pred_id)  
        top_50 = tokenizer.batch_decode(sorted_preds)
        pred_words.append(pred_word)
        top_50_pred_words.append(top_50)
        probs = softmax(logits, dim=0).cpu().detach().numpy()
        top_50_probs = softmax(sorted_logits, dim=0).cpu().detach().numpy()
        entropy = ss.entropy(probs)
        entropies.append(entropy)
        top_50_entropy = ss.entropy(top_50_probs)
        top_50_entropies.append(top_50_probs)
    return pred_words, entropies, top_50_pred_words, top_50_entropies

def get_trigram_spacy(x):
    target_idx = x.target_i
    start_idx = 0
    for sent_i, s in enumerate(x.text_spacy.sents):
        if target_idx >= s.start and target_idx < s.end:
            start_idx = s.start
            sent = s
            break
    adjusted_idx = target_idx - start_idx
    start = sent[:adjusted_idx].text
    end =  sent[adjusted_idx+1:].text
    new_text = start + f" {x.trigram_pred} " + end
    new_text = new_text.strip().capitalize()
    new_spacy = nlp(new_text)
    # adjust for scenario where lm doesn't predict a space before completion
    adjusted_idx -= (len(sent) - len(new_spacy))
    return {'trigram_text': x.trigram_pred, 'trigram_pos': new_spacy[adjusted_idx].pos_, 'trigram_lemma': new_spacy[adjusted_idx].lemma_, 'trigram_i': adjusted_idx + sent.start}

def get_lm_spacy(x):
    target_idx = x.target_i
    start_idx = 0
    for sent_i, s in enumerate(x.text_spacy.sents):
        if target_idx >= s.start and target_idx < s.end:
            start_idx = s.start
            sent = s
            break
    adjusted_idx = target_idx - start_idx
    start = sent[:adjusted_idx].text
    end =  sent[adjusted_idx+1:].text
    new_text = start + f"{x.lm_generated} " + end
    new_text = new_text.strip().capitalize()
    new_spacy = nlp(new_text)
    # adjust for scenario where lm doesn't predict a space before completion
    adjusted_idx -= (len(sent) - len(new_spacy))
    return {'lm_text': x.lm_generated, 'lm_pos': new_spacy[adjusted_idx].pos_, 'lm_lemma': new_spacy[adjusted_idx].lemma_, 'lm_i': adjusted_idx + sent.start}

def get_lm_top_spacy(x):
    x = x.drop('lm_generated')
    df = x.to_frame().T.explode('lm_50_generated').rename({'lm_50_generated': 'lm_generated'}, axis=1)
    spacy_lm_token = df.apply(get_lm_spacy, axis=1)
    return pd.DataFrame.from_records(spacy_lm_token.tolist()).to_dict(orient='list')

def process_data(df, tokenized_prompts, trigram_context, debug):
    df['Word'] = df.Word.str.lower()
    #custom fixes
    df['Text'] = df.Text.str.replace('90%', '0.9%')
    df['Text'] = df.Text.str.replace('Inc.-owned', 'Inc. -owned')
    df['Text'] = df.Text.str.replace('womenÕ', 'women')
    df['Response'] = df.Response.str.replace("cant", "can't")
    text_spacy = pd.Series(df.Text.unique()).swifter.apply(nlp)

    print("Processing Human Response data...")
    text_spacy = pd.DataFrame(text_spacy, columns=['text_spacy'])
    text_spacy.index += 1
    df = df.merge(text_spacy, left_on='Text_ID', right_index=True)
    df = df.dropna().reset_index(drop=True)
    target_token = df.swifter.apply(match_word, axis=1)
    df = pd.concat([df, pd.DataFrame.from_records(target_token)], axis=1) 
    df = df[df.target_text == df.Word].reset_index(drop=True)
    human_token = df.apply(replace_word_with_response, axis=1)
    df = pd.concat([df, pd.DataFrame.from_records(human_token)], axis=1)
    df = df.dropna().reset_index(drop=True)


    print("Processing Trigram model...")
    trigram_df = df[['Word_Unique_ID', 'Text', 'text_spacy', 'target_i']].drop_duplicates()
    trigram_df['prompt'] = trigram_df.swifter.apply(lambda x: x.text_spacy[:x.target_i].text, axis=1)
    kenlm_df = pd.read_csv('/data/mourad/provo_data/kenlm_preds.tsv', sep='\t')
    trigram_df = trigram_df.merge(kenlm_df, on='Word_Unique_ID')
    #df = df.merge(kenlm_df, on='Word_Unique_ID')
    spacy_trigram_token = trigram_df.swifter.apply(get_trigram_spacy, axis=1)
    x = pd.DataFrame.from_records(spacy_trigram_token.tolist()).to_dict(orient='list')
    ngram_df = pd.DataFrame.from_records(x)
    trigram_df = pd.concat([trigram_df, ngram_df], axis=1)
    if not debug:
        trigram_df.to_csv('/data/mourad/provo_data/trigram_df_for_context_analysis.tsv', sep='\t', index=False)
    
    print("Processing GPT2...")
    # get LM next word prediction
    lm_df = df[['Word_Unique_ID', 'Text', 'text_spacy', 'target_i']].drop_duplicates().reset_index(drop=True)
    #lm_df['prompt'] = lm_df.swifter.apply(lambda x: x.text_spacy[:x.target_i].text, axis=1)
    lm_df = lm_df.merge(tokenized_prompts, on="Word_Unique_ID", how='inner')
    #lm_df['prompt'] = tokenized_prompts.prompt
    if trigram_context: 
        lm_df['prompt'] = lm_df.prompt.swifter.apply(lambda x: " ".join(x.split()[-2:]))
    generated_words, entropies, top_50_words, top_50_probs = get_lm_completion_logits(lm_df.prompt)
    lm_df['lm_generated'] = generated_words
    lm_df['lm_entropy'] = entropies 
    lm_df['lm_50_generated'] = top_50_words
    lm_df['lm_50_probs'] = top_50_probs
    lm_50_tokens = lm_df.swifter.apply(get_lm_top_spacy, axis=1)
    lm_df = pd.concat([lm_df.reset_index(drop=True), pd.DataFrame.from_records(lm_50_tokens)], axis=1) 

    df = df.drop(['Text_ID', 'Text', 'Word_Number', 'Sentence_Number',
                   'Word_In_Sentence_Number', 'Word', 'Response', 'Response_Count',
                   'Total_Response_Count', 'Response_Proportion'], axis=1)
    df = df.groupby(['Word_Unique_ID', 'text_spacy', 'target_text', 'target_pos', 'target_i'], sort=False, as_index=False).agg(list)
    lm_df = lm_df.drop(['Text', 'prompt', 'text_spacy', 'target_i', 'lm_generated'], axis=1)
    df = df.merge(lm_df, on='Word_Unique_ID')
    return df

def load_fix_spacy_tokenizer():
    nlp = spacy.load("en_core_web_sm")
    inf = list(nlp.Defaults.infixes)
    inf = [x for x in inf if '-|–|—|--|---|——|~' not in x] # remove the hyphen-between-letters pattern from infix patterns
    infix_re = compile_infix_regex(tuple(inf))

    def custom_tokenizer(nlp):
        return Tokenizer(nlp.vocab, prefix_search=nlp.tokenizer.prefix_search,
                                    suffix_search=nlp.tokenizer.suffix_search,
                                    infix_finditer=infix_re.finditer,
                                    token_match=nlp.tokenizer.token_match,
                                    rules=nlp.Defaults.tokenizer_exceptions)

    nlp.tokenizer = custom_tokenizer(nlp)
    nlp.tokenizer.add_special_case("inc.", [{'ORTH': "inc.", 'NORM': 'inc.'}])
    nlp.tokenizer.add_special_case("and/or", [{'ORTH': "and/or", 'NORM': 'and/or'}])
    nlp.tokenizer.add_special_case("mr.", [{'ORTH': "mr.", 'NORM': 'mr.'}])
    nlp.tokenizer.add_special_case("mrs.", [{'ORTH': "mrs.", 'NORM': 'mrs.'}])
    nlp.tokenizer.add_special_case("dr.", [{'ORTH': "dr.", 'NORM': 'dr.'}])
    return nlp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', required=False)
    parser.add_argument('--trigram', action='store_true')
    args = parser.parse_args()

    df_eye, df_cloze, tokenized_prompts = load_data()
    if args.debug:
        df_cloze = df_cloze.head(100)
        tokenized_prompts = tokenized_prompts.head(100)
    reuse_preds = False
    if reuse_preds:
        df_spacy = pd.read_csv('/data/mourad/provo_data/spacy_and_gpt2_data.pkl') #, sep='\t')
    else:
        nlp = load_fix_spacy_tokenizer()
        data_df = process_data(df_cloze, tokenized_prompts, args.trigram, args.debug)
        pdb.set_trace()
        if not args.debug:
            data_df.to_pickle(f"/data/mourad/provo_data/spacy_and{'_trigram' if args.trigram else ''}_gpt2_entropy_top_50_data.pkl")
        pdb.set_trace()
