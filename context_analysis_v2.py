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
import functools
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, LabelEncoder
import statsmodels.api as sm
from stargazer.stargazer import Stargazer
from stargazer.stargazer import LaTeXRenderer
from tqdm.auto import tqdm
import nltk
#nltk.download('omw-1.4')

#DATA_DIR = '/data/mourad/provo_data/new_plots/'

wn_pos_map = {
    'VERB': 'v',
    'NOUN': 'n',
    'ADV': 'r',
    'ADJ': 'a' 
}

content_pos = {'NOUN': 'content', 'VERB': 'content', 'ADV': 'content', 'ADJ': 'content', 'NUM': 'content'}

sns.set_style("whitegrid")
sns.set(#font_scale=1.7,
     rc={
         #"figure.figsize":(15.7,7.27), 
         'axes.facecolor':'white',
         'axes.edgecolor':'black',
         'axes.grid': True,
         "axes.grid.axis" : "y",
         'xtick.bottom': True, 
        'ytick.left':True,
        'xtick.direction': 'in',
        'ytick.direction': 'in'
     }
     )
fontsize = 19

def extract_spacy_tokens(debug, trigram_gpt2):
    data_df = pd.read_pickle(f"/data/mourad/provo_data/spacy_and{'_trigram' if trigram_gpt2 else ''}_gpt2_entropy_top_50_data.pkl") #, sep='\t')
    if debug: data_df = data_df.head(debug)
    target_df = data_df[['Word_Unique_ID', 'text_spacy', 'target_text', 'target_pos', 'target_i']] 
    human_df = data_df[['Word_Unique_ID', 'human_text', 'human_pos', 'human_lemma', 'target_i']]
    human_df = human_df.explode(['human_text', 'human_pos', 'human_lemma'])
    human_df = human_df.rename({'human_text': 'text', 
                                'human_pos': 'pos', 
                                'human_lemma': 'lemma', 
                                'target_i': 'i'}, axis=1)
    lm_df = data_df[['Word_Unique_ID', 'lm_entropy', 'lm_50_probs', 'lm_text', 'lm_pos', 'lm_lemma', 'lm_i']]
    lm_df = prepare_lm_df(lm_df)
    lm_df = lm_df.rename({  'lm_entropy': 'entropy',
                            'lm_text': 'text', 
                            'lm_pos': 'pos', 
                            'lm_lemma': 'lemma', 
                            'lm_i': 'i'}, axis=1)

    trigram_df = pd.read_csv('/data/mourad/provo_data/trigram_df_for_context_analysis.tsv', sep='\t')
    if debug: trigram_df = trigram_df.head(debug)
    trigram_df = trigram_df[['Word_Unique_ID', 'kenlm_prob', 'trigram_text', 'trigram_pos', 'trigram_lemma', 'trigram_i']]
    trigram_df = trigram_df.rename({    'kenlm_prob': 'probability',
                                        'trigram_text': 'text', 
                                        'trigram_pos': 'pos', 
                                        'trigram_lemma': 'lemma', 
                                        'trigram_i': 'i'}, axis=1)

    target_df['target_token'] = target_df.swifter.progress_bar(True) \
                                            .apply(lambda x: x.text_spacy[x.target_i], axis=1)
    target_df['preceding_token'] = target_df.swifter.progress_bar(True) \
                                            .apply(lambda x: x.text_spacy[x.target_i - 1], axis=1)
    target_df = target_df.rename({  'target_text': 'text', 
                                    'target_pos': 'pos', 
                                    'target_i': 'i', 
                                    'target_token': 'token'}, axis=1)
    resp_counts = pd.read_csv('/data/mourad/provo_data/response_counts.tsv', sep='\t', index_col=0)
    resp_counts = resp_counts.rename({'Response': 'text', 'Response_Count': 'response_count'}, axis=1)
    resp_counts.text = resp_counts.text.str.lower()
    human_df = human_df.merge(resp_counts, on=['Word_Unique_ID', 'text'])
    #pdb.set_trace()
    #target_df.reset_index(drop=True)[['Word_Unique_ID', 'text_spacy', 'i', 'text']].to_pickle('/data/mourad/pcfg/target_df.pkl')
    return target_df.reset_index(drop=True), {  'human': human_df.reset_index(drop=True), 
                                                'lm': lm_df.reset_index(drop=True),
                                                'trigram': trigram_df.reset_index(drop=True)}

def prepare_lm_df(df):
    df = df.explode(['lm_pos', 'lm_lemma', 'lm_i', 'lm_50_probs','lm_text'])
    df.lm_50_probs = df.lm_50_probs.astype(float)
    pos_entropy_df = df.groupby(['Word_Unique_ID', 'lm_pos']).lm_50_probs.sum().to_frame() \
                        .groupby('Word_Unique_ID').lm_50_probs.apply(ss.entropy).to_frame(name='pos_entropy')
    df = df.drop('lm_50_probs', axis=1)
    df = df.groupby('Word_Unique_ID', sort=False).first().reset_index()
    df = df.merge(pos_entropy_df, left_on='Word_Unique_ID', right_index=True) 
    df.lm_text = df.lm_text.str.strip()
    return df

def plot_pos_dists(resp_dfs):
    all_resps=[]
    for resp, df in resp_dfs.items():
        df['content_function'] = df.pos.swifter.progress_bar(True).apply(lambda x: 'content' if x in content_pos else 'function') 
        xlabel = f"{resp.upper()} Prediction POS Distribution"
        filename = f"{resp}_pos_dist"
        df = df.reset_index(drop=True)
        print(f"{resp.upper()} Content Word {round(df[(df.text_match == 1) & (df.content_function == 'content')].size / df.size, 3) * 100}%")
        utils.histplot(
                        df=df, 
                        x='pos', 
                        xlabel=xlabel, 
                        ylabel=None, 
                        filename=filename, 
                        DATA_DIR=DATA_DIR, 
                        multiple='stack', 
                        rotate=True, 
                        shrink=0.8,
                        hue='text_match',
                        alpha=1,
                        stat='percent'
                    ) 
        '''utils.histplot(
                        df=df,
                        x='pos',
                        xlabel='Predicted POS',
                        )'''
        df['resp'] = resp
        all_resps.append(df)

    df = pd.concat(all_resps, axis=0)
    df = df.merge(target_df[['Word_Unique_ID', 'pos']], on='Word_Unique_ID', suffixes=['_pred', '_gold'])
    fig, ax = plt.subplots(figsize=(12,8))
    g = sns.barplot(data=df,
                    x='pos_pred',
                    y='text_match',
                    hue='resp',
                    errwidth=1, capsize=0.1, ci=68,
                    ax=ax)
    plt.xticks(rotation=45)
    #ax.set_xticklabels(ax.get_xticklabels(),rotation = 45)
    #if ylabel:
    filename=f'all_pos_pred_vs_text_match'
    ax.set_xlabel(f'Target Gold POS')
    ax.set_ylabel('Exact Text Match Accuracy')
    fig.savefig(f"{DATA_DIR}{filename}.png", bbox_inches='tight', dpi=400)
    fig.clf()
    pdb.set_trace()

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
    return mutual_info / math.sqrt(ss.entropy(resp_df, base=2).item() * ss.entropy(feature_df, base=2).item())
    #return mutual_info/np.log2(len(resp_df))

def get_features():
    target_df['preceding_dep'] = target_df.token.swifter.progress_bar(True).apply(get_dependency)
    target_df['preceding_pos'] = target_df.preceding_token.swifter.progress_bar(True).apply(lambda x: x.pos_)
    target_df['position_in_text'] = target_df.swifter.progress_bar(True).apply(lambda x: round(x.i / len(x.text_spacy), 1), axis=1)
    target_df['position_in_sent'] = target_df.token.swifter.progress_bar(True).apply(lambda x: round((x.i - x.sent.start) / len(x.sent), 1))
    

def calc_matches(df, resp):
    df['pos_match'] = df.swifter.progress_bar(True).apply(lambda x: x.pos == target_df[target_df.Word_Unique_ID == x.Word_Unique_ID].pos.item(), axis=1).astype(int)
    df['text_match'] = df.swifter.progress_bar(True).apply(lambda x: x.text.lower() == target_df[target_df.Word_Unique_ID == x.Word_Unique_ID] \
                            .text.item().lower(), axis=1).astype(int)
    print("CALCULATING SYNSET & HYPERNYM MATCHES")
    df = eval_synset_hypernym(resp, df)
    print("CALCULATING TARGET VECTOR SIMILARITY MATCHES") 
    # target cos sim
    #lm_df = resp_dfs['lm'].copy()
    df['spacy'] = df.drop_duplicates().text.swifter.progress_bar(True).apply(lambda x: list(nlp(x).vector))
    df['target_sim_match'] = df.swifter.progress_bar(True).apply(get_target_cos_sim, axis=1)
    return df


def calc_accuracy(acc_metric):
    for resp, df in resp_dfs.items():
        df = calc_matches(df, resp)
        if resp == 'human':
            df = df.loc[df.index.repeat(df.response_count)]
            df = df.drop('response_count', axis=1)
            if acc_metric == 'sample':
                simulated_human_accs = {'text': [], 'pos': [], 'syn': [], 'hyp': [], 'target_cos': []}
                for i in tqdm(range(100)):
                    sampled_df = df.groupby('Word_Unique_ID').sample(n=1).reset_index(drop=True)
                    #sampled_df = calc_matches(sampled_df, resp) 
                    simulated_human_accs['text'].append(sampled_df.text_match.mean())
                    simulated_human_accs['pos'].append(sampled_df.pos_match.mean())
                    simulated_human_accs['syn'].append(sampled_df.synset_match.mean())
                    simulated_human_accs['hyp'].append(sampled_df.hypernym_match.mean())
                    simulated_human_accs['target_cos'].append(sampled_df.target_sim_match.mean())
                
                simulated_df = pd.DataFrame(simulated_human_accs) 
                for acc in simulated_df.columns:
                    xlabel = f"Simulated Human {acc.upper()}"
                    filename = f"simulated_human_{acc}"
                    utils.histplot(df=simulated_df, x=acc, xlabel=xlabel, ylabel=None, filename=filename, binwidth=0.01, DATA_DIR=DATA_DIR) 
                print(simulated_df.mean())
            elif acc_metric == 'majority':
                max_df = df.groupby(['Word_Unique_ID', 'text']).size().to_frame(name='response_count').reset_index()
                max_idx = max_df.groupby(['Word_Unique_ID']).response_count.transform(max) == max_df.response_count
                max_df = max_df[max_idx].reset_index(drop=True).drop_duplicates('Word_Unique_ID')
                df = max_df.merge(df, on=["Word_Unique_ID", 'text'], how='left').drop_duplicates('Word_Unique_ID')
                #df = calc_matches(df, resp)
            #else:
                #df = calc_matches(df, resp)

                    #else:
        #    df = calc_matches(df, resp)
        resp_dfs[resp] = df 
    print_accuracy(acc_metric)

def print_accuracy(acc_metric):
    for acc in accuracies:
        bin_acc = acc
        val_acc = acc
        if acc in ['pos', 'text']: 
            acc += '_match'
            bin_acc = acc
        all_correct(feature='all', acc=val_acc, bin_acc=bin_acc)
        records = []
        for i, col in enumerate([   resp_dfs['human'].get(acc), 
                                    resp_dfs['human'].groupby('Word_Unique_ID').apply(lambda x: x.get(acc).mean() if x.get(acc) is not None else None), 
                                    resp_dfs['lm'].get(acc),
                                    resp_dfs['trigram'].get(acc)]):
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
            f"Trigram Completion {acc.replace('_', ' ').upper()}",
        ]
        stats_df.index = index_labels
        print(stats_df.dropna())
        
        if acc in resp_dfs['human'].columns: 
            means = resp_dfs['human'].groupby('Word_Unique_ID')[acc].mean().reset_index()
            xlabel = f"Human {acc.replace('_', ' ').title()}"
            filename = f"human_grouped_{acc}{'_' + acc_metric if acc_metric else ''}"
            utils.histplot(df=means, x=acc, xlabel=xlabel, ylabel=None, filename=filename, binwidth=0.02, DATA_DIR=DATA_DIR) 


    # TODO don't forget to sample the data as tsv
    #sample_df = spacy_df[['human_match', 'human_token', 'lm_match', 'lm_token', 'human_lm_match', f'{feature}', 'target_token', 'text_spacy']].copy()
    #utils.save_sample(sample_df, feature=feature, match=match)

 
def calc_rmi():
    rmis = {}
    for resp, df in resp_dfs.items():
        rmis[resp] = {}
        for feature in features:
            for acc in accuracies:
                if acc in ['pos', 'text']: acc += '_match'
                if acc in df.columns:
                    feature_df = target_df[feature].value_counts(normalize=True).to_frame()
                    resp_df = df[acc].value_counts(normalize=True).to_frame()
                    cond_df = df[['Word_Unique_ID', acc]].merge(target_df[['Word_Unique_ID', feature]], on='Word_Unique_ID')
                    conditionals_dict = cond_df.groupby(acc)[feature].value_counts(normalize=True).to_dict()
                    #pdb.set_trace()
                    #data_df = resp_df.merge(feature_df, left_index=True, right_index=True, how='outer').fillna(0)
                    rmi = calc_mutual_information(feature_df, resp_df, conditionals_dict)
                    rmis[resp][f'{acc}|{feature}'] = round(rmi, 3)
    return rmis
    

def plot_accuracy():
    for feature in features:
        for acc in accuracies:
            if acc in ['pos', 'text']: 
                bin_acc = acc + '_match'
            else: bin_acc = acc
            df = target_df[['Word_Unique_ID', feature]]
            dfs = []

            for resp, resp_df in resp_dfs.items():
                if bin_acc in resp_df.columns:
                    resp_df = resp_df[['Word_Unique_ID', bin_acc]]
                    resp_df = df.merge(resp_df, on='Word_Unique_ID').rename({bin_acc: f"{resp}_{bin_acc}"}, axis=1)
                    resp_df = pd.melt(resp_df, id_vars=["Word_Unique_ID", feature], value_vars = f"{resp}_{bin_acc}", var_name='target', value_name=bin_acc)
                    dfs.append(resp_df)

            '''if bin_acc in resp_dfs['lm'].columns:
                lm_df = resp_dfs['lm'][['Word_Unique_ID', bin_acc]]
                lm_df = df.merge(lm_df, on='Word_Unique_ID').rename({bin_acc: f"lm_{bin_acc}"}, axis=1)
                lm_df = pd.melt(lm_df, id_vars = ['Word_Unique_ID', feature], value_vars = f"lm_{bin_acc}", var_name='target', value_name=bin_acc)
                dfs.append(lm_df)
            if bin_acc in resp_dfs['trigram'].columns:
                trigram_df = resp_dfs['trigram'][['Word_Unique_ID', bin_acc]]
                trigram_df = df.merge(trigram_df, on='Word_Unique_ID').rename({bin_acc: f"trigram_{bin_acc}"}, axis=1)
                trigram_df = pd.melt(trigram_df, id_vars = ['Word_Unique_ID', feature], value_vars = f"lm_{bin_acc}", var_name='target', value_name=bin_acc)
                dfs.append(trigram_df)

            if bin_acc in resp_dfs['human'].columns:
                human_df = resp_dfs['human'][['Word_Unique_ID', bin_acc]]
                human_df = df.merge(human_df, on='Word_Unique_ID').rename({bin_acc: f"human_{bin_acc}"}, axis=1) #, suffixes=["_lm", "_human"])
                human_df = pd.melt(human_df, id_vars = ['Word_Unique_ID', feature], value_vars = f"human_{bin_acc}", var_name='target', value_name=bin_acc)
                dfs.append(human_df)'''
            #df = df.reset_index()
            if len(dfs) >= 2: 
                ttest_df = ttest(dfs, feature, bin_acc) 
                all_correct(feature, acc, bin_acc)
                for model in ['lm', 'trigram']:
                    unique_correct(model, feature, acc, bin_acc)
               # if len(ttest_df) > 0: # and acc in ['pos', 'text']::
                #    find_patterns(ttest_df, feature, acc, bin_acc)
                    #find_patterns_simple(ttest_df, feature, acc, bin_acc)
            df = pd.concat(dfs).reset_index(drop=True)
            if feature.startswith('position'):
                ax = sns.lineplot(
                    data=df,
                    x=feature,
                    y=bin_acc,
                    hue="target",
                    #x_estimator=True, x_ci=68, scatter=True, fit_reg=False,
                    estimator='mean', ci=68,
                    )
            else:
                ax = sns.barplot(
                    data=df, 
                    x=feature, 
                    y=bin_acc, 
                    hue="target", 
                    errwidth=1, capsize=0.1, ci=68,
                    )
                ax.set_xticklabels(ax.get_xticklabels(),rotation = 45)
            ax.set_xlabel(feature.replace('_', ' ').upper())
            ax.set_ylabel(f"Mean Target {bin_acc.replace('_', ' ').upper()}")
            fig = ax.get_figure()
            filename = f"{DATA_DIR}accuracy/{feature}_vs_{bin_acc}.png"
            fig = utils.set_fig(fig, filename)

def get_target_cos_sim(x):
    if type(x.spacy) is not list:
        return np.nan
    #if type(x.Word_Unique_ID) == pd.Series:
    #    target_vec = nlp(target_df[target_df.Word_Unique_ID == x.Word_Unique_ID.drop_duplicates().item()].text.item()).vector
    #else:
    target_vec = nlp(target_df[target_df.Word_Unique_ID == x.Word_Unique_ID].text.item()).vector
   # if type(x.spacy) == list:
    embs = np.array(x.spacy)
    embs_target_sim = cosine_similarity(embs.reshape(1,-1), target_vec.reshape(1,-1))
    #else:
    #   embs = np.array(x.spacy.tolist())
    #    embs_target_sim = cosine_similarity(embs, target_vec.reshape(1,-1))
    sim_mean = embs_target_sim.mean()
    return sim_mean
    
def eval_synset_hypernym(resp, df):
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
    
    wordnet_data = df.swifter.progress_bar(True).apply(is_synonym, axis=1)
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
        ax.set_xlabel(t.split('_')[0].capitalize() + " Match")
        ax.set_ylabel("Normalized Mutual Information (NMI)")
        fig = ax.get_figure()
        filename = f"{DATA_DIR}rmi/{t}.png" 
        fig = utils.set_fig(fig, filename)

def ttest(dfs, feature, bin_acc):
    lm_df = dfs[0].groupby(feature)[bin_acc].agg(list).to_frame()
    human_df = dfs[1].groupby(feature)[bin_acc].agg(list).to_frame() 
    df = lm_df.merge(human_df, left_index=True, right_index=True)
    df.columns = ['lm', 'human']
    df = df.apply(lambda x: ss.ttest_ind(x.lm, x.human), axis=1).to_frame('ttest')
    df[['stat', 'pval']] = pd.DataFrame(df.ttest.tolist(), index=df.index)
    df = df.drop('ttest', axis=1)
    ttest_df = df[df.pval <= 0.05] 
    return ttest_df
    #if len(df) > 0:
    
def find_patterns_simple(ttest_df, feature, acc, bin_acc):
    global target_df
    #lm_df = dfs[0].groupby(feature)[bin_acc].agg(list).to_frame()
    #human_df = dfs[1].groupby(feature)[bin_acc].agg(list).to_frame()
    #df = lm_df.merge(human_df, left_index=True, right_index=True)
    #df.columns = ['lm', 'human']    
    target_df = target_df[target_df[feature].isin(ttest_df.index)]
    df = resp_dfs['lm'].merge(resp_dfs['human'], on='Word_Unique_ID', suffixes=['_lm', '_human']) 
    df = df.merge(target_df, on='Word_Unique_ID')
    df = df[['Word_Unique_ID', feature, f"{acc}_lm", f"{acc}_human", f"{bin_acc}_lm", f"{bin_acc}_human"]]
    df['resp_dif'] = df[f"{bin_acc}_lm"] - df[f"{bin_acc}_human"]
    df = df.groupby(['Word_Unique_ID', feature, f'{acc}_lm', f'{acc}_human']).mean().reset_index(level=[1,2,3]).reset_index(drop=True)
    df = df.groupby([feature, f'{acc}_lm', f'{acc}_human']).agg(list).reset_index()
    df['grp_size'] = df.resp_dif.apply(len)
    df['resp_dif'] = df.resp_dif.apply(np.mean)
    df = df[[feature, f"{acc}_lm", f"{acc}_human", 'resp_dif', 'grp_size']].sort_values(by='resp_dif', ascending=False)
    df.to_csv(f'{DATA_DIR}auto_samples/simple_{feature}_{acc}.tsv', index=False, sep='\t')

def unique_correct(model, feature, acc, bin_acc):
    global target_df
    df = resp_dfs[model].merge(resp_dfs['human'], on='Word_Unique_ID', suffixes=[f'_{model}', '_human']) 
    df = df.merge(target_df, on='Word_Unique_ID')
    df = df[['Word_Unique_ID', feature, f"{acc}_{model}", f"{acc}_human", f"{bin_acc}_{model}", f"{bin_acc}_human", "text_human", f'text_{model}']]
    df = df.loc[:,~df.columns.duplicated()].copy()
    df['resp_dif'] = df[f"{bin_acc}_{model}"] - df[f"{bin_acc}_human"]
    df = df[df.resp_dif < 0][['Word_Unique_ID', feature, f"{acc}_{model}", f'text_{model}', f"{acc}_human", 'text_human']]
    df = df.merge(target_df[['Word_Unique_ID', 'text', 'text_spacy']], on='Word_Unique_ID')
    df = df.loc[:,~df.columns.duplicated()].copy()
    filepath = f'{DATA_DIR}auto_samples/human_correct_{model}_incorrect_{feature}_{acc}.tsv'
    df.to_csv(filepath, index=False, sep='\t')
    print(filepath)

def all_correct(feature, acc, bin_acc):
    cols_to_keep = ['Word_Unique_ID', acc, 'i'] #content_function 
    trigram = resp_dfs['trigram']
    try:
        trigram = trigram[trigram[bin_acc] == 1][cols_to_keep]
    except: return
    lm = resp_dfs['lm']
    lm = lm[lm[bin_acc] == 1][cols_to_keep]
    human = resp_dfs['human']
    human = human[human[bin_acc] == 1][cols_to_keep]
    trig_human = trigram.merge(human, on=['Word_Unique_ID', 'i'], suffixes=['_trigram', '_human']) 
    trig_lm = trigram.merge(lm, on=['Word_Unique_ID', 'i'], suffixes=['_trigram', '_lm'])
    trig_lm_human = trig_lm.merge(human, on=['Word_Unique_ID', 'i'])
    trig_lm_human = trig_lm_human.rename({acc: f'{acc}_human'}, axis=1)
    #pdb.set_trace()
    for name, df in [('trigram_human', trig_human), ('trigram_lm', trig_lm), ('trigram_lm_human', trig_lm_human)]:
        df = df.merge(target_df[['Word_Unique_ID', 'text_spacy']], on='Word_Unique_ID')
        filepath = f'{DATA_DIR}auto_samples/{name}_correct_{feature}_{acc}.tsv'
        df = df.drop_duplicates()
        #df.to_csv(filepath, index=False, sep='\t')

def semantic_clustering(resp_dfs):
    human_df = resp_dfs['human'].copy()
    human_df['spacy'] = human_df.text.swifter.progress_bar(True).apply(lambda x: list(nlp(x).vector))
    def get_cos_sim_mean(x):
        embs = np.array(x.tolist())
        embs_sims = cosine_similarity(embs)
        # drop diagonal
        embs_sims = embs_sims[~np.eye(embs_sims.shape[0],dtype=bool)].reshape(embs_sims.shape[0],-1)
        sim_mean = embs_sims.mean()
        sim_std = embs_sims.std()
        return sim_mean, sim_std
    
    #human_df = human_df.drop_duplicates(subset=['Word_Unique_ID', 'text'])
    sims_mean = human_df.groupby('Word_Unique_ID').spacy.apply(get_cos_sim_mean) \
                                                        .to_frame(name='cluster_stats') \
                                                        .reset_index()
    human_df = human_df.merge(sims_mean, on='Word_Unique_ID')
    
    semantic_clusters = human_df \
        .groupby(['Word_Unique_ID', 'cluster_stats', 'target_sim_match'])['text', 'pos', 'text_match', 'pos_match', 'synset_match', 'hypernym_match'] \
        .agg(list) \
        .reset_index() \
        .rename({
                'text_match': 'human_text_match', 
                'pos_match': 'human_pos_match', 
                'synset_match': 'human_synset_match', 
                'hypernym_match':'human_hypernym_match',
                'target_sim_match': 'human_target_sim_match'}, axis=1)
    semantic_clusters.human_text_match = semantic_clusters.human_text_match \
                                                        .swifter.progress_bar(True).apply(lambda x: np.array(x).mean())
    semantic_clusters.human_pos_match = semantic_clusters.human_pos_match \
                                                        .swifter.progress_bar(True).apply(lambda x: np.array(x).mean())
    semantic_clusters.human_synset_match = semantic_clusters.human_synset_match \
                                                        .swifter.progress_bar(True).apply(lambda x: np.array(x).mean())
    semantic_clusters.human_hypernym_match = semantic_clusters.human_hypernym_match \
                                                        .swifter.progress_bar(True).apply(lambda x: np.array(x).mean())

    
    semantic_clusters[['cluster_mean','cluster_std']] = pd.DataFrame(semantic_clusters.cluster_stats.tolist()
                                                                    , index= semantic_clusters.index)
    semantic_clusters = semantic_clusters.drop('cluster_stats', axis=1)
    semantic_clusters = semantic_clusters.sort_values(by='cluster_mean', ascending=False)
    semantic_clusters['len'] = semantic_clusters.text.swifter.progress_bar(True).apply(len)
    semantic_clusters = semantic_clusters.merge(resp_dfs['lm'][['Word_Unique_ID', 
                                                                'text_match', 
                                                                'pos_match', 
                                                                'synset_match', 
                                                                'hypernym_match',
                                                                'target_sim_match']], on='Word_Unique_ID')
    #semantic_clusters = semantic_clusters.merge(lm_target_sim, on='Word_Unique_ID')
    semantic_clusters = semantic_clusters.dropna().rename({
                                                        'target_sim_match': 'lm_target_sim_match',
                                                        'text_match': 'lm_text_match', 
                                                        'pos_match': 'lm_pos_match',
                                                        'synset_match': 'lm_synset_match', 
                                                        'hypernym_match':'lm_hypernym_match',
                                                        'target_sim_match': 'lm_target_sim_match'}, axis=1)
    semantic_clusters = semantic_clusters.merge(target_df[['Word_Unique_ID', 'pos',
                                                            'preceding_pos',
                                                            'preceding_dep', 
                                                            'position_in_text', 
                                                            'position_in_sent']].rename({'pos':'target_pos'}, axis=1), on='Word_Unique_ID')
    semantic_clusters.to_csv(f'{DATA_DIR}auto_samples/semantic_clusters.csv', index=False)
    for context in ['preceding_pos', 'preceding_dep', 'position_in_text','position_in_sent']:
        fig, ax = plt.subplots(figsize=(12,8))
        g = sns.barplot(
                    data=semantic_clusters,
                    x=context,
                    y='cluster_mean',
                    #hue='context',
                    errwidth=1, capsize=0.1, ci=68,
                    )
        xlabel = f"Target {context.replace('_', ' ')}"
        ylabel = 'Pairwise Mean Embedding Cosine Similarity'
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        filename = f'{context}_sim_barplot'
        fig.savefig(f"{DATA_DIR}{filename}.png", dpi=400, bbox_inches='tight')
        fig.clf()
    print('human text')
    human_text = ss.spearmanr(semantic_clusters.cluster_mean, semantic_clusters.human_text_match)
    print('human pos')
    human_pos = ss.spearmanr(semantic_clusters.cluster_mean, semantic_clusters.human_pos_match) 
    print('human syn')
    human_syn = ss.spearmanr(semantic_clusters.cluster_mean, semantic_clusters.human_synset_match)
    print('human hyp')
    human_hyp = ss.spearmanr(semantic_clusters.cluster_mean, semantic_clusters.human_hypernym_match)
    print('human resp-target sim')
    human_target_sim = ss.spearmanr(semantic_clusters.cluster_mean, semantic_clusters.human_target_sim_match)
    print('lm text')
    lm_text = ss.spearmanr(semantic_clusters.cluster_mean, semantic_clusters.lm_text_match)
    print('lm pos')
    lm_pos = ss.spearmanr(semantic_clusters.cluster_mean, semantic_clusters.lm_pos_match)
    print('lm syn')
    lm_syn = ss.spearmanr(semantic_clusters.cluster_mean, semantic_clusters.lm_synset_match)
    print('lm hyp')
    lm_hyp = ss.spearmanr(semantic_clusters.cluster_mean, semantic_clusters.lm_hypernym_match)
    print('lm resp-target sim')
    lm_target_sim = ss.spearmanr(semantic_clusters.cluster_mean, semantic_clusters.lm_target_sim_match)

    print(human_text, '\n', human_pos, '\n', human_syn, '\n', human_hyp, '\n', human_target_sim, '\n', lm_text, '\n', lm_pos, '\n', lm_syn, '\n', lm_hyp, '\n', lm_target_sim)
    return semantic_clusters

def semantic_clusters_analysis(df):
    fig, ax = plt.subplots(figsize=(12,8))
    g = sns.kdeplot(
            data=df,
            x='cluster_mean',
            #hue='Treatment',
            multiple='stack',
            bw_adjust=2,
            edgecolor='black', linewidth=1,
            ax=ax
            ) 
    ax.set_xlabel('Human Response Cosine Similarity')
    filename = f'cos_sim_mean_kde'
    fig.savefig(f"{DATA_DIR}{filename}.png", dpi=400, bbox_inches='tight')
    fig.clf()
    
    for acc in ['pos', 'text', 'synset', 'hypernym', 'target_sim']:
        fig, ax = plt.subplots(figsize=(12,8))
        g1 = sns.scatterplot(
                data = df,
                x=f'human_{acc}_match',
                y='cluster_mean',
                #hue='target_pos',
                ax=ax,
                s=4,
                legend=False,
                #hue_order=['AO', 'NL']
                )
        #df = df.groupby(['index', 'Treatment']).dif.apply(self.remove_outliers).explode().to_frame().reset_index()
        #df = df.reset_index()
        g2 = sns.lineplot(
            data = df,
            x=f'human_{acc}_match',
            y='cluster_mean',
            #fit_reg=True,
            estimator=np.mean,
            #estimator=self.remove_outliers,
            ci=68,
            #hue='target_pos',
            #line_kws={'linewidth':2},
            ax=ax,
            linewidth = 2,
            #hue_order=['AO', 'NL']
            )
        xlabel=f'Human Response {acc.upper()} Match Accuracy'
        ylabel='Human Response Cosine Similarity'
        ax.set_xlabel(xlabel) #, fontsize=fontsize)
        ax.set_ylabel(ylabel) #, fontsize=fontsize)
        fig.tight_layout()
        filename = f'{acc}_cluster_mean'
        fig.savefig(f"{DATA_DIR}{filename}.png", dpi=400, bbox_inches='tight')
        fig.clf()

def regress_context(target_df, resp_dfs, clusters):
    x_df = clusters[['preceding_pos', 'preceding_dep', 'position_in_text', 'position_in_sent']]
    x_df = pd.get_dummies(x_df)
    y_df = clusters[['cluster_mean']]
    #y_df['y_label'] = label_encoder.fit_transform(y_df.cluster_mean)
    lin_regr(x_df, y_df)

def lin_regr(all_x_df, y_df):
    results = []
    resp_var_name = 'Cloze Response Mean Cos Sim'

    #if normal
    actions = ['preceding_pos', 'preceding_dep', 'position_in_text', 'position_in_sent']
    # if using pusher type
    #actions = [('pusher', 'pusher'), ('newer', 'newer')]
    actions.append('all')
    for idx, context_var in enumerate(actions):
        x_df = all_x_df.copy()
        if context_var != 'all':
            #x_df = x_df[[context_var]]
            filter_col = [col for col in x_df if col.startswith(context_var)]
            x_df = x_df[filter_col]
        y_df = y_df.rename({'cluster_mean': resp_var_name}, axis=1)
        x = sm.add_constant(x_df)
        #result = sm.OLS(y_df[resp_var_name], x).fit()
        result = sm.OLS(y_df[resp_var_name], x).fit(cov_type='HC0')
        results.append(result)
        print(result.summary())
    stargazer = Stargazer(results)
    stargazer.dependent_variable_name(resp_var_name)
    stargazer.significant_digits(3)
    stargazer.show_sig = False
    stargazer.show_f_statistic = False
    stargazer.show_r2 = False
    stargazer.show_residual_std_err = False
    stargazer.show_notes = False
    stargazer.show_degrees_of_freedom(False)
    stargazer.cov_spacing = "\\addlinespace"
    #if not self.reduced_clusters:
    #    stargazer.rename_covariates({'buyer_n': 'Buyer new offer', 'seller_n': 'Seller new offer',
    #                                'buyer_r': 'Buyer repeat offer', 'seller_r': 'Seller repeat offer',
    #                                'buyer_p': 'Buyer push', 'seller_p': 'Seller push',
    #                                'buyer_a': 'Buyer allowance', 'seller_a': 'Seller allowance',
    #                                'buyer_c': 'Buyer comparison', 'seller_c': 'Seller comparison',
    #                                })
    #    for k,v in stargazer.cov_map.items():
    #        if k != 'buyer_n':
    #            stargazer.cov_map[k] = f"\\addlinespace {v}"
    #        #stargazer.rename_covariates({v: })
    #    stargazer.covariate_order(list(stargazer.cov_map.keys()))
    #stargazer.covariate_order(['Buyer new offer', 'Seller new offer',
    #                            'Buyer repeat offer', 'Seller repeat offer',
    #                            'Buyer comparison', 'Seller comparison',
    #                            'Buyer allowance', 'Seller allowance',
    #                            'Buyer push', 'Seller push']) 
    #renderer = LaTeXRenderer(stargazer)
    pos = [col for col in all_x_df if col.startswith('preceding_pos')]
    dep = [col for col in all_x_df if col.startswith('preceding_dep')]
    position = [col for col in all_x_df if col.startswith('position')]
    stargazer.covariate_order(pos + dep + position)
    latex = stargazer.render_latex() + '\n'
    latex = latex.replace('\cline{5-6}', '\cline{2-6}')
    latex = latex.replace('& \multicolumn{5}{c}', 'Dep. Var.: & \multicolumn{5}{c}')
    for col in all_x_df.columns:
        new_col = col.replace('_', ' ')
        new_col = new_col[0].upper() + new_col[1:]
        latex = latex.replace(col, new_col)
    filename = f"linreg_ols_resp_cluster"
    print(filename)
    with open(f'{REG_DATA_DIR}{filename}.tex', 'w') as f:
        f.write(latex)


def find_patterns(ttest_df, feature, acc, bin_acc):
    global target_df
    target_df = target_df[target_df[feature].isin(ttest_df.index)]
    df = resp_dfs['lm'].merge(resp_dfs['human'], on='Word_Unique_ID', suffixes=['_lm', '_human']) 
    df = df.merge(target_df, on='Word_Unique_ID')
    df = df[['Word_Unique_ID', feature, f"{acc}_lm", f"{acc}_human", f"{bin_acc}_lm", f"{bin_acc}_human"]]
    df['resp_dif'] = df[f"{bin_acc}_lm"] - df[f"{bin_acc}_human"]
    df = df.groupby(['Word_Unique_ID', feature, f'{acc}_lm', f'{acc}_human']).mean().reset_index(level=[1,2,3]).reset_index(drop=True)
    df = df.groupby([feature, f'{acc}_lm', f'{acc}_human']).agg(list).reset_index()
    res_dfs = []
    for name, group in df.groupby(feature)[[f'{acc}_lm', f'{acc}_human']]:
        rv_df = group[[f'{acc}_lm', f'{acc}_human']]
        combos = pd.DataFrame(np.sort(rv_df.values, axis=1), columns=rv_df.columns).value_counts().reset_index(name='counts')
        #sig_combos = combos[combos.counts == 1]#.drop('counts', axis=1)
        #combos = combos[combos.counts >= 2].drop('counts', axis=1)
        ttests = {}
        for i, data in combos.iterrows():
            data = data.tolist()
            count = data[2]
            if count == 2:
                x = group[(group[f"{acc}_lm"] == data[0]) & (group[f"{acc}_human"] == data[1])]
                y = group[(group[f"{acc}_lm"] == data[1]) & (group[f"{acc}_human"] == data[0])]
                means = [np.mean(x.resp_dif.item()), np.mean(y.resp_dif.item())]
                ttest = ss.ttest_ind(x.resp_dif.item(), y.resp_dif.item())
                if not math.isnan(ttest.statistic) and ttest.statistic != 0:
                    correct = 'lm' if np.sign(means[::int(np.sign(ttest.statistic))][0]) > 0 else 'human'
                else: correct = None
                stat = round(ttest.statistic, 2)
                pval = round(ttest.pvalue, 3)
                freq = -1
            else:
                stat, pval = -1, -1
                try:
                    x = group[(group[f"{acc}_lm"] == data[0]) & (group[f"{acc}_human"] == data[1])].resp_dif.item()
                except: 
                    x = group[(group[f"{acc}_lm"] == data[1]) & (group[f"{acc}_human"] == data[0])].resp_dif.item()
                x_mean = np.mean(x)
                if x_mean == 0: continue
                correct = 'lm' if np.sign(x_mean) > 0 else 'human'
                freq = len(x)
            ttests[tuple(data)] = { 'context': feature,
                                    'target': acc,
                                    'stat': stat, 
                                    'pval': pval, 
                                    'correct': correct,
                                    'freq': freq  }
        res_df = pd.DataFrame(ttests).T
        if len(res_df) > 0:
            print(f"**** {feature}: {name}, target: {bin_acc} ****")
            res_df = res_df[res_df.pval <= 0.05]
            res_df['([lm] / [human])'] = [r.Index[::int(np.sign(r.stat))] for r in res_df.itertuples()]
            res_df = res_df.reset_index(drop=True)
            res_df['stat'] = res_df.stat.abs()
            res_df = res_df[['context', 'target', '([lm] / [human])', 'stat', 'pval', 'correct', 'freq']]
            res_dfs.append(res_df)
            #res_df.to_csv(f'{DATA_DIR}auto_samples/feature-{feature}_target-{acc}.tsv', index=False, sep='\t')
            #print(res_df)
    pd.concat(res_dfs).to_csv(f'{DATA_DIR}auto_samples/{feature}_{acc}.tsv', index=False, sep='\t')
    pd.concat(res_dfs).to_latex(f'{DATA_DIR}auto_samples/{feature}_{acc}.tex', index=False)
    # TODO dont forget combos that only appear once

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature', '-f', choices=['prev_pos','dep','len','sent_len','all'], default='all', help='Linguistic feature for analysis')
    parser.add_argument('--match', '-m', choices=['pos', 'exact', 'entropy', 'all'], default='all', help='variable to measure accuracy')
    parser.add_argument('--accuracy_metric', '-am', choices=['majority', 'sample', 'all'])
    parser.add_argument('--trig_gpt2', action='store_true', help='use gpt2 trigram predictions')
    parser.add_argument('--debug', '-d', type=int, required=False)
    args = parser.parse_args()
    
    DATA_DIR = f"/data/mourad/provo_data/fixed_plots/{'trigram_gpt2/' if args.trig_gpt2 else ''}"
    REG_DATA_DIR = f"/data/mourad/provo_data/ols/{'trigram_gpt2/' if args.trig_gpt2 else ''}"

    nlp = spacy.load("en_core_web_lg")

    df_eye, df_cloze, tokenized_prompts = load_data()
    target_df, resp_dfs = extract_spacy_tokens(args.debug, args.trig_gpt2)
    features = ['preceding_dep', 'preceding_pos', 'position_in_text', 'position_in_sent']
    accuracies = ['pos', 'text', 'synset_match', 'hypernym_match', 'target_sim_match', 'entropy', 'pos_entropy']
    get_features()
    calc_accuracy(args.accuracy_metric)
    #pdb.set_trace()
    plot_pos_dists(resp_dfs)
    semantic_clusters = semantic_clustering(resp_dfs)
    semantic_clusters_analysis(semantic_clusters)
    #regress_context(target_df, resp_dfs, semantic_clusters)
    rmis = calc_rmi() 
    plot_accuracy()
    plot_rmi(rmis)


