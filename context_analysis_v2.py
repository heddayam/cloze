from prelim_analysis_full_context import load_data
import pandas as pd
import pdb
import seaborn as sns
from nltk.corpus import wordnet as wn
import plotly.express as px
import plotly.graph_objs as go
import scipy.stats as ss
import swifter
import numpy as np
import argparse
import matplotlib.pyplot as plt
import math
import utils
import spacy
import scipy.stats as ss
import functools

DATA_DIR = '/data/mourad/provo_data/new_plots/'

wn_pos_map = {
    'VERB': 'v',
    'NOUN': 'n',
    'ADV': 'r',
    'ADJ': 'a' 
}

def extract_spacy_tokens(data_df):
    target_df = data_df[['Word_Unique_ID', 'text_spacy', 'target_text', 'target_pos', 'target_i']] 
    human_df = data_df[['Word_Unique_ID', 'human_text', 'human_pos', 'human_lemma', 'target_i']]
    human_df = human_df.explode(['human_text', 'human_pos', 'human_lemma'])
    human_df = human_df.rename({'human_text': 'text', 'human_pos': 'pos', 'human_lemma': 'lemma', 'target_i': 'i'}, axis=1)
    lm_df = data_df[['Word_Unique_ID', 'lm_entropy', 'lm_50_probs', 'lm_text', 'lm_pos', 'lm_lemma', 'lm_i']]
    lm_df = prepare_lm_df(lm_df)
    lm_df = lm_df.rename({'lm_entropy': 'entropy', 'lm_text': 'text', 'lm_pos': 'pos', 'lm_lemma': 'lemma', 'lm_i': 'i'}, axis=1)
    target_df['target_token'] = target_df.swifter.progress_bar(False).apply(lambda x: x.text_spacy[x.target_i], axis=1)
    target_df['preceding_token'] = target_df.swifter.progress_bar(False).apply(lambda x: x.text_spacy[x.target_i - 1], axis=1)
    target_df = target_df.rename({'target_text': 'text', 'target_pos': 'pos', 'target_i': 'i', 'target_token': 'token'}, axis=1)
    return target_df.reset_index(drop=True), {'human': human_df.reset_index(drop=True), 'lm': lm_df.reset_index(drop=True)}

def prepare_lm_df(df):
    df = df.explode(['lm_pos', 'lm_lemma', 'lm_i', 'lm_50_probs','lm_text'])
    df.lm_50_probs = df.lm_50_probs.astype(float)
    pos_entropy_df = df.groupby(['Word_Unique_ID', 'lm_pos']).lm_50_probs.sum().to_frame() \
                        .groupby('Word_Unique_ID').lm_50_probs.apply(ss.entropy).to_frame(name='pos_entropy')
    df = df.drop('lm_50_probs', axis=1)
    df = df.groupby('Word_Unique_ID', sort=False).first().reset_index()
    df = df.merge(pos_entropy_df, left_on='Word_Unique_ID', right_index=True) 
    df['lm_text'] = df.lm_text.swifter.apply(lambda x: x.strip())
    return df

def plot_pos_dists(spacy_df, kind):
    df = spacy_df.copy()
    df['exact'] = df.swifter.apply(lambda x: x.target_token.text == x[f'{kind}_token'].text, axis=1) 
    df = df[df.exact == True]
    df['target_pos'] = df.target_token.swifter.progress_bar(False).apply(lambda x: x.pos_)
    df['human_pos'] = df[f'{kind}_token'].swifter.progress_bar(False).apply(lambda x: x.pos_)
    #df['lm_pos'] = df.lm_token.swifter.progress_bar(False).apply(lambda x: x.pos_)
    df = df[['Word_Unique_ID', 'target_pos']]
    df = df.merge(df.groupby('Word_Unique_ID').size().to_frame(name='n_correct').reset_index(), on='Word_Unique_ID')
    if kind == 'lm':
        df['n_correct'] = 1
    df = pd.melt(
                    df, 
                    id_vars=['Word_Unique_ID', 'n_correct'], 
                    value_vars=['target_pos'], 
                    var_name='Target Type', value_name='POS'
                )
    ax = sns.histplot(
        data=df, 
        x="POS", 
        hue="n_correct", 
        stat = 'percent', common_norm=False, multiple='dodge', shrink=.8, bins=5
        )
    ax.set_xlabel("Target POS")
    ax.set_ylabel(f'Percent (%)')
    #ax.set_xticklabels(ax.get_xticklabels(),rotation = 45)
    ax.tick_params(axis='x', rotation=45)
    sns.despine(top=True, right=True, ax=ax)
    fig = ax.get_figure()    
    fig.tight_layout()
    #fig.set_size_inches(15,7)
    fig.savefig(f"{DATA_DIR}{kind}_correct_pos_dist.png", dpi=400, bbox_inches='tight')
    fig.clf()

def get_dependency(token):
    return get_dep_helper(token, token.i)    
            
def get_dep_helper(token, target_i):
    token_children = [c for c in token.children]
    if token.sent.start == target_i:
        return None
    elif token.head.i < target_i: # if head is to the left of current token
        return token.head.pos_
    elif token.n_lefts > 0:
        return token_children[token.n_lefts-1].pos_
    elif token.head.i > target_i:
        return get_dep_helper(token.head, target_i)
    elif token.n_rights > 0:
        return '**CHILDS**'
    else:
        return None

def calc_mutual_information(feature_df, resp_df, conditionals_dict):
    all_ents = []
    for n, x_feat in feature_df.iterrows():
        for m, y_resp in resp_df.iterrows():
            #  p(x,y) = p(x | y) * P(y)
            if (m, n) in conditionals_dict:
                joint = conditionals_dict[(m, n)] * y_resp.item()
                log = np.log2(joint/x_feat.item())
                ent = joint * log
            else:
                ent = 0
            all_ents.append(ent)
    conditional_entropy = np.negative(np.array(all_ents).sum())
    mutual_info = ss.entropy(resp_df, base=2).item() - conditional_entropy
    return mutual_info /ss.entropy(resp_df, base=2).item()
    #return mutual_info/np.log2(len(resp_df))

def get_features():
    target_df['preceding_dep'] = target_df.token.swifter.apply(get_dependency)
    target_df['preceding_pos'] = target_df.preceding_token.swifter.progress_bar(True).apply(lambda x: x.pos_)
    target_df['position_in_text'] = target_df.swifter.apply(lambda x: round(x.i / len(x.text_spacy), 1), axis=1)
    target_df['position_in_sent'] = target_df.token.swifter.apply(lambda x: round((x.i - x.sent.start) / len(x.sent), 1))
    
def calc_accuracy():
    for resp, df in resp_dfs.items():
        df['pos_match'] = df.swifter.apply(lambda x: x.pos == target_df[target_df.Word_Unique_ID == x.Word_Unique_ID].pos.item(), axis=1)
        df['text_match'] = df.swifter.apply(lambda x: x.text == target_df[target_df.Word_Unique_ID == x.Word_Unique_ID].text.item(), axis=1)
        df = eval_synset(resp, df)
        resp_dfs[resp] = df 
    print_accuracy()

def print_accuracy():
    for acc in accuracies:
        if acc in ['pos', 'text']: acc += '_match'
        records = []
        for i, col in enumerate([   resp_dfs['human'].get(acc), 
                                    resp_dfs['human'].groupby('Word_Unique_ID').apply(lambda x: x.get(acc).mean() if x.get(acc) is not None else None), 
                                    resp_dfs['lm'].get(acc)]):
            if col is not None and len(col) > 0:
                #col *= 100
                if i == 1:
                    std = round(col.std(), 4)
                else:
                    std = '-'
                record = {
                    "Mean": round(col.mean(), 4),
                    "Std. Dev": std,
                }
                records.append(record)
            else: records.append({})
        stats_df = pd.DataFrame.from_dict(records)
        index_labels = [
            f"Human Completion {acc.replace('_', ' ').upper()}",
            f"Human Complteion Grouped {acc.replace('_', ' ').upper()}",
            f"GPT2 Completion {acc.replace('_', ' ').upper()}",
        ]
        #pdb.set_trace()
        stats_df.index = index_labels
        print(stats_df.dropna())

    # TODO don't forget to sample the data as tsv
    #sample_df = spacy_df[['human_match', 'human_token', 'lm_match', 'lm_token', 'human_lm_match', f'{feature}', 'target_token', 'text_spacy']].copy()
    #utils.save_sample(sample_df, feature=feature, match=match)

 
def calc_rmi():
    rmis = {}
    for resp, df in resp_dfs.items():
        rmis[resp] = {}
        for feature in features:
            for acc in accuracies:
                if acc in df.columns:
                    feature_df = target_df[feature].value_counts(normalize=True).to_frame()
                    resp_df = df[acc].value_counts(normalize=True).to_frame()
                    cond_df = df[['Word_Unique_ID', acc]].merge(target_df[['Word_Unique_ID', feature]], on='Word_Unique_ID')
                    conditionals_dict = cond_df.groupby(acc)[feature].value_counts(normalize=True).to_dict()
                    rmi = calc_mutual_information(feature_df, resp_df, conditionals_dict)
                    rmis[resp][f'{acc}|{feature}'] = round(rmi, 3)
    return rmis
    

def plot_accuracy():
    for feature in features:
        for acc in accuracies:
            if acc in ['pos', 'text']: acc += '_match'
            df = target_df[['Word_Unique_ID', feature]]
            dfs = []
            if acc in resp_dfs['lm'].columns:
                lm_df = resp_dfs['lm'][['Word_Unique_ID', acc]]
                lm_df = df.merge(lm_df, on='Word_Unique_ID').rename({acc: f"lm_{acc}"}, axis=1)
                lm_df = pd.melt(lm_df, id_vars = ['Word_Unique_ID', feature], value_vars = f"lm_{acc}", var_name='target', value_name=acc)
                dfs.append(lm_df)
            if acc in resp_dfs['human'].columns:
                human_df = resp_dfs['human'][['Word_Unique_ID', acc]]
                human_df = df.merge(human_df, on='Word_Unique_ID').rename({acc: f"human_{acc}"}, axis=1) #, suffixes=["_lm", "_human"])
                human_df = pd.melt(human_df, id_vars = ['Word_Unique_ID', feature], value_vars = f"human_{acc}", var_name='target', value_name=acc)
                dfs.append(human_df)
            #df = df.reset_index()
            #val_list = [col for col in df.columns if col.startswith(acc)]
            if len(dfs) == 2: 
                ttest(dfs) 
            df = pd.concat(dfs).reset_index(drop=True)
            if feature.startswith('position'):
                ax = sns.lineplot(
                    data=df,
                    x=feature,
                    y=acc,
                    hue="target",
                    #x_estimator=True, x_ci=68, scatter=True, fit_reg=False,
                    estimator='mean', ci=68,
                    )
            else:
                ax = sns.barplot(
                    data=df, 
                    x=feature, 
                    y=acc, 
                    hue="target", 
                    errwidth=1, capsize=0.1, ci=68,
                    )
                ax.set_xticklabels(ax.get_xticklabels(),rotation = 45)
            ax.set_xlabel(feature.replace('_', ' ').upper())
            ax.set_ylabel(f"Mean Target {acc.replace('_', ' ').upper()}")
            fig = ax.get_figure()
            filename = f"{DATA_DIR}accuracy/{feature}_vs_{acc}.png"
            fig = utils.set_fig(fig, filename)
    
def eval_synset(resp, df):
    def is_synonym(row):
        target_data = target_df[target_df.Word_Unique_ID == row.Word_Unique_ID]
        if row.text_match:
            is_in_synset = int(True)
            has_shared_hypernym = int(True)
        elif target_data.pos.item() not in wn_pos_map:
            return {}
        else:
            pos = wn_pos_map[target_data.pos.item()]
            target_all_synset = wn.synsets(target_data.text.item(), pos=pos)
            target_synset = [r.name().split('.')[0] for r in target_all_synset]
            is_in_synset = int(row.lemma in target_synset)
            has_shared_hypernym = shared_hypernym(row, target_all_synset)
        return {'synset_match': is_in_synset, 'hypernym_match': has_shared_hypernym}

    def shared_hypernym(row, target_synset):
        if row.pos not in wn_pos_map:
            return None
        pos = wn_pos_map[row.pos]
        human_synset = wn.synsets(row.text, pos=pos)
        path_distances = []
        synset_pairs = []
        hypernym_pairs = []
        for target_syn in target_synset:
            target_hypernyms = set(target_syn.hypernyms())
            for human_syn in human_synset:
                human_hypernyms = set(human_syn.hypernyms())
                if len(target_hypernyms.intersection(human_hypernyms)) >= 1:
                    hypernym_pairs.append((target_syn, human_syn))
        return int(len(hypernym_pairs) > 0)
    
    wordnet_data = df.swifter.apply(is_synonym, axis=1)
    df = pd.concat([df, pd.DataFrame.from_records(wordnet_data)], axis=1)
    return df

def plot_rmi(rmis):
    df = pd.DataFrame(rmis).reset_index().rename({'index': 'context'}, axis=1)
    df = pd.melt(df, id_vars='context', value_vars=['human', 'lm'], var_name='response', value_name='RMI')
    df[['target', 'context']] = df['context'].str.split('|', 1, expand=True)
    for t in df.target.unique():
        ax = sns.barplot(
                data=df[df.target == t].dropna(),
                x='response',
                y='RMI',
                hue='context',
                errwidth=1, capsize=0.1, ci=68,
                )
        ax.set_ylabel("Relative Mutual Information (RMI)")
        fig = ax.get_figure()
        filename = f"{DATA_DIR}rmi/{t}.png" 
        fig = utils.set_fig(fig, filename)

def ttest(dfs):
    lm_df = dfs[0].groupby('preceding_dep').pos_match.agg(list).to_frame()
    human_df = dfs[1].groupby('preceding_dep').pos_match.agg(list).to_frame() 
    df = lm_df.merge(human_df, left_index=True, right_index=True)
    df.columns = ['lm', 'human']
    df = df.apply(lambda x: ss.ttest_ind(x.lm, x.human), axis=1).to_frame('ttest')
    df[['stat', 'pval']] = pd.DataFrame(df.ttest.tolist(), index=df.index)
    df = df.drop('ttest', axis=1)
    df = df[df.pval <= 0.05] 
    if len(df) > 0:
        pdb.set_trace()       
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature', '-f', choices=['prev_pos','dep','len','sent_len','all'], default='all', help='Linguistic feature for analysis')
    parser.add_argument('--match', '-m', choices=['pos', 'exact', 'entropy', 'all'], default='all', help='variable to measure accuracy')
    args = parser.parse_args()
    df_eye, df_cloze = load_data()
    data_df = pd.read_pickle('/data/mourad/provo_data/spacy_and_gpt2_entropy_top_50_data.pkl') #, sep='\t')
    target_df, resp_dfs = extract_spacy_tokens(data_df)
    #for kind in ['resp', 'lm']:
    #    plot_pos_dists(spacy_df, kind=kind)
    #pdb.set_trace()
    features = ['preceding_dep', 'preceding_pos', 'position_in_text', 'position_in_sent']
    accuracies = ['pos', 'text', 'synset_match', 'hypernym_match', 'entropy', 'pos_entropy']
    get_features()
    calc_accuracy()
    rmis = calc_rmi() 
    plot_accuracy()
    plot_rmi(rmis)


