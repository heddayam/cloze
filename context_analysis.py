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

DATA_DIR = '/data/mourad/provo_data/plots/'

wn_pos_map = {
    'VERB': 'v',
    'NOUN': 'n',
    'ADV': 'r',
    'ADJ': 'a' 
}

def extract_spacy_tokens(spacy_df):
    spacy_df['og_spacy_token'] = spacy_df.swifter.progress_bar(False).apply(lambda x: x.text_spacy[x.target_spacy_i], axis=1)
    spacy_df['lm_spacy_token'] = spacy_df.swifter.progress_bar(False).apply(lambda x: x.lm_spacy[x.lm_spacy_i], axis=1)
    spacy_df['resp_spacy_token'] = spacy_df.swifter.progress_bar(False).apply(lambda x: x.resp_spacy[x.resp_spacy_i], axis=1)
    spacy_df['preceding_token_spacy'] = spacy_df.swifter.progress_bar(False).apply(lambda x: x.text_spacy[x.target_spacy_i - 1], axis=1)
    return spacy_df

def plot_pos_dists(spacy_df, kind):
    df = spacy_df.copy()
    df['exact'] = df.swifter.apply(lambda x: x.og_spacy_token.text == x[f'{kind}_spacy_token'].text, axis=1) 
    df = df[df.exact == True]
    df['og_pos'] = df.og_spacy_token.swifter.progress_bar(False).apply(lambda x: x.pos_)
    df['resp_pos'] = df[f'{kind}_spacy_token'].swifter.progress_bar(False).apply(lambda x: x.pos_)
    #df['lm_pos'] = df.lm_spacy_token.swifter.progress_bar(False).apply(lambda x: x.pos_)
    df = df[['Word_Unique_ID', 'og_pos']]
    df = df.merge(df.groupby('Word_Unique_ID').size().to_frame(name='n_correct').reset_index(), on='Word_Unique_ID')
    if kind == 'lm':
        df['n_correct'] = 1
    df = pd.melt(
                    df, 
                    id_vars=['Word_Unique_ID', 'n_correct'], 
                    value_vars=['og_pos'], 
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
            
def get_dep_helper(token, og_token_idx):
    token_children = [c for c in token.children]
    if token.head.i < og_token_idx: # if head is to the left of current token
        return token.head.pos_
    elif token.n_lefts > 0:
        return token_children[token.n_lefts-1].pos_
    elif token.head.i > og_token_idx:
        return get_dep_helper(token.head, og_token_idx)
    elif token.n_rights > 0:
        return '**CHILDS**'
    else:
        return None

def get_sent_len(token):
    return round((token.i - token.sent.start) / len(token.sent), 1)

def calc_conditional_entropy(feature_df, resp_df, conditionals_dict):
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
 
def calc_entropy(spacy_df, feature, match):
    #df = spacy_df.groupby(f'{feature}').lm_feature.value_counts(normalize=True).to_frame(name='pct').reset_index()
    #max_features = df.groupby(f'{feature}').size().max()
    entropy = {}
    for kind in ['lm', 'resp']:
        if match == 'pos': 
            spacy_df[f'{kind}_resp'] = spacy_df[f'{kind}_spacy_token'].swifter.progress_bar(False).apply(lambda x: x.pos_)
        elif match == 'exact':        
            spacy_df[f'{kind}_resp'] = spacy_df[f'{kind}_spacy_token'].swifter.progress_bar(False).apply(lambda x: x.text)
        elif match in ['synset', 'hypernym']:
            spacy_df[f'{kind}_resp'] = spacy_df[f'{kind}_{match}_match']
        elif match == 'entropy':
            if kind == 'resp': 
                spacy_df[f'{kind}_resp'] = np.nan
            else:
                spacy_df[f'{kind}_resp'] = spacy_df[f'{kind}_{match}']
        
        resp_df = spacy_df[f'{kind}_resp'].value_counts(normalize=True).to_frame()
        feature_df = spacy_df[f'{feature}'].value_counts(normalize=True).to_frame()
        #pct_df = pct_df.merge(spacy_df[f'{feature}'].value_counts(normalize=True), left_index=True, right_index=True, how='outer')
        #pct_df = pct_df.fillna(0)    
        conditionals_dict = spacy_df.groupby(f'{kind}_resp')[f'{feature}'].value_counts(normalize=True).to_dict()
        mutual_info = calc_conditional_entropy(feature_df, resp_df, conditionals_dict)
        #entropy = df.groupby(f'{feature}').pct.agg(calc_entropy) #.agg(lambda x: ss.entropy(x, qk=[1/len(x) for i in range(len(x))]))
        entropy[kind] = mutual_info
        print(f'{kind.upper()}, resp={match.upper()}, feat={feature.upper()}: Mutual Information = {mutual_info}')
    
    return entropy

def compare_feature(eye_df, spacy_df, feature, match, ax):
    print(f"\nFeature: {feature}, Match: {match}")
    if feature == 'dep':    
        xlabel = 'Nearest Left Dependency POS'
        spacy_df['dep'] = spacy_df.og_spacy_token.swifter.progress_bar(False).apply(get_dependency)
    elif feature == 'prev_pos':
        xlabel = 'Preceding POS'
        spacy_df['prev_pos'] = spacy_df.swifter.progress_bar(False).apply(lambda x: x.text_spacy[x.target_spacy_i - 1].pos_, axis=1)
    elif feature == 'len':
        xlabel = 'Position in Text'
        #spacy_df['len'] = spacy_df.Word_Number - 1
        spacy_df['len'] = spacy_df.swifter.progress_bar(False).apply(lambda x: round((x.Word_Number-1)/len(x.text_spacy),1), axis=1)
    elif feature == 'sent_len':
        xlabel = 'Position in Sentence'
        spacy_df['sent_len'] = spacy_df.og_spacy_token.swifter.progress_bar(False).apply(get_sent_len)
        #spacy_df['sent_len'] = spacy_df.og_spacy_token.swifter.progress_bar(False).apply(lambda x: x)

    if match == 'pos':
        spacy_df['resp_match'] = spacy_df.swifter.progress_bar(False).apply(lambda x: x.resp_spacy_token.pos_ == x.og_spacy_token.pos_, axis=1)
        spacy_df['lm_match'] = spacy_df.swifter.progress_bar(False).apply(lambda x: x.lm_spacy_token.pos_ == x.og_spacy_token.pos_, axis=1)
        spacy_df['resp_lm_match'] = spacy_df.swifter.progress_bar(False).apply(lambda x: x.resp_spacy_token.pos_ == x.lm_spacy_token.pos_, axis=1)
    elif match == 'exact':
        spacy_df['resp_match'] = spacy_df.swifter.progress_bar(False).apply(lambda x: x.resp_spacy_token.text == x.og_spacy_token.text, axis=1)
        spacy_df['lm_match'] = spacy_df.swifter.progress_bar(False).apply(lambda x: x.lm_spacy_token.text == x.og_spacy_token.text, axis=1)
        spacy_df['resp_lm_match'] = spacy_df.swifter.progress_bar(False).apply(lambda x: x.resp_spacy_token.text == x.lm_spacy_token.text, axis=1)
    elif match in ['synset', 'hypernym']:
        spacy_df['resp_match'] = spacy_df[f'resp_{match}_match']
        spacy_df['lm_match'] = spacy_df[f'lm_{match}_match']
        spacy_df['resp_lm_match'] = spacy_df[f'resp_lm_{match}_match']
    elif match == 'entropy':
        spacy_df['lm_match'] = spacy_df[f"lm_entropy"]
        spacy_df['resp_match'] = np.nan
        spacy_df['resp_lm_match'] = np.nan

    records = []
    for i, col in enumerate([spacy_df['resp_match'], spacy_df.groupby('Word_Unique_ID').resp_match.mean(), spacy_df['lm_match']]):
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
    stats_df = pd.DataFrame.from_dict(records)
    index_labels = [
        f'Cloze Response {match.upper()} Accuracy %',
        f'Cloze Response Grouped {match.upper()} Accuracy %',
        f'DistilGPT2 Completion {match.upper()} Accuracy %',
    ]
    stats_df.index = index_labels
    print(stats_df)
     
    sample_df = spacy_df[['resp_match', 'resp_spacy_token', 'lm_match', 'lm_spacy_token', 'resp_lm_match', f'{feature}', 'og_spacy_token', 'Text']].copy()
    utils.save_sample(sample_df, feature=feature, match=match)
    
    #if match == 'entropy':    
    df = spacy_df[['resp_match', 'lm_match', 'resp_lm_match', f'{feature}']]
    df = df.reset_index()
    val_list = ['resp_lm_match', 'resp_match', 'lm_match']
    df = pd.melt(df, id_vars = ['index', f'{feature}'] , value_vars=val_list)
    
    #pdb.set_trace() 
    if feature in ['len', 'sent_len']: 
        sns.lineplot(
            data=df,
            x=f"{feature}",
            y="value",
            hue="variable",
            #x_estimator=True, x_ci=68, scatter=True, fit_reg=False,
            estimator='mean', ci=68,
            ax=ax
            )
        #ax.set_xticks(range(df[f"{feature}"].nunique()))
        #ax.set_xticklabels(df[f"{feature}"].unique())
    else:
        sns.barplot(
            data=df, 
            x=f"{feature}", 
            y="value", 
            hue="variable", 
            errwidth=1, capsize=0.1, ci=68,
            ax=ax,
            )
        ax.set_xticklabels(ax.get_xticklabels(),rotation = 45)

    #ax.set_xlabel(f"Nearest Left {'Dependency POS' if feature == 'dep' else 'POS'}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f'Mean Target {match.upper()} Match Accuracy')
    sns.despine(top=True, right=True, ax=ax)
    # ax.set(ylim=(200, 240))
    return spacy_df
    
def eval_synset(spacy_df):
    resp_df = spacy_df[['Word_Unique_ID', 'og_spacy_token', 'resp_spacy_token']]
    lm_df = spacy_df[['Word_Unique_ID', 'og_spacy_token', 'lm_spacy_token']]
    lm_df = lm_df.drop_duplicates()

    def is_in_synset(row, kind, og='og'):
        og = row[f'{og}_spacy_token']
        if row[f'{kind}_spacy_token'].text == og.text:
            return int(True)
        if og.pos_ not in wn_pos_map:
            return None
        pos = wn_pos_map[og.pos_]
        og_synset = wn.synsets(og.text, pos=pos)
        og_synset = [r.name().split('.')[0] for r in og_synset]
        return int(row[f'{kind}_spacy_token'].text in og_synset)

    def shared_hypernym(row, kind, og='og'):
        og = row[f'{og}_spacy_token']
        resp = row[f'{kind}_spacy_token']
        if resp.text == og.text:
            return int(True)
        if og.pos_ not in wn_pos_map:
            return None
        pos = wn_pos_map[og.pos_]
        og_synsets = wn.synsets(og.text, pos=pos)
        if resp.pos_ not in wn_pos_map:
            return None
        resp_pos = wn_pos_map[resp.pos_]
        resp_synsets = wn.synsets(resp.text, pos=resp_pos)
        path_distances = []
        synset_pairs = []
        hypernym_pairs = []
        for og_synset in og_synsets:
            og_hypernyms = set(og_synset.hypernyms())
            for resp_synset in resp_synsets:
                resp_hypernyms = set(resp_synset.hypernyms())
                if len(og_hypernyms.intersection(resp_hypernyms)) >= 1:
                    hypernym_pairs.append((og_synset, resp_synset))
        return int(len(hypernym_pairs) > 0)

    for kind in ['resp', 'lm']:
        spacy_df[f'{kind}_synset_match'] = spacy_df.swifter.progress_bar(False).apply(lambda x: is_in_synset(x, kind), axis=1)
        spacy_df[f'{kind}_hypernym_match'] = spacy_df.swifter.progress_bar(False).apply(lambda x: shared_hypernym(x, kind), axis=1)
    
    spacy_df['resp_lm_synset_match'] = spacy_df.swifter.progress_bar(False).apply(lambda x: is_in_synset(x, kind='lm', og='resp'), axis=1) 
    spacy_df['resp_lm_hypernym_match'] = spacy_df.swifter.progress_bar(False).apply(lambda x: shared_hypernym(x, kind='lm', og='resp'), axis=1) 
       
    return spacy_df

def subplot(df_eye, spacy_df, match):
    entropy = {}
    #fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, sharex=True)
    fig1, ax1 = plt.subplots(nrows=1, ncols=2, sharey=True)
    fig2, ax2 = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=True)
    figs = [fig1, fig2]
    axs = [ax1, ax2]
    for i, feature in enumerate(['len', 'sent_len', 'prev_pos', 'dep']):
        ax = axs[math.floor(i/2)][i % 2]
        spacy_df = compare_feature(df_eye, spacy_df, feature, match, ax)
        entropy[feature] = calc_entropy(spacy_df.copy(), feature, match)
        handles, labels = ax.get_legend_handles_labels()
        ax.get_legend().remove()
        if i % 2 == 1:
            fig = figs[math.floor(i/2)]
            fig.tight_layout()
            fig.legend(handles, labels, loc='upper right')
            fig.set_size_inches(15,7)
            fig.savefig(f"{DATA_DIR}{feature}_{match}_vs_accuracy.png", dpi=400, bbox_inches='tight')
            
            fig.clf()    

    return entropy
    
def subplot_entropy(entropies):
    fig1, ax1 = plt.subplots(nrows=1, ncols=2, sharey=True)
    fig2, ax2 = plt.subplots(nrows=1, ncols=2, sharey=True)
    fig3, ax3 = plt.subplots(nrows=1, ncols=2, sharey=True)
    figs = [fig1, fig2, fig3]
    axs = [ax1, ax2, ax3]
    #ax = ax.flatten()
    for i, (match, v) in enumerate(entropies.items()):    
        df = pd.DataFrame(v).reset_index().rename({'index': 'predictor'},axis=1).replace({'resp':'human'})
        df = pd.melt(df, id_vars='predictor', value_vars=['len', 'sent_len', 'prev_pos', 'dep'], value_name='entropy', var_name='context')
        ax = axs[math.floor(i/2)][i % 2]
        sns.barplot(
            data=df, 
            x="predictor", 
            y="entropy", 
            hue="context", 
            errwidth=1, capsize=0.1, ci=68,
            ax=ax,
            )
        #ax.set_xlabel(f"")
        ax.set_ylabel(f'Relative Mutual Information')
        #ax.set_xticklabels(ax.get_xticklabels(),rotation = 45)
        sns.despine(top=True, right=True, ax=ax)
        handles, labels = ax.get_legend_handles_labels()
        ax.get_legend().remove()
        ax.title.set_text(f'Target {match.upper()}')
        if i % 2 == 1 or (len(entropies)%2 == 1 and i == len(entropies)-1):
            fig = figs[math.floor(i/2)]
            fig.tight_layout()
            fig.legend(handles, labels, loc='upper right')
            fig.set_size_inches(8,6)
            #fig.set_size_inches(12,12)
            fig.savefig(f"{DATA_DIR}{match}_mutual_information.png", dpi=400, bbox_inches='tight')

            fig.clf()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature', '-f', choices=['prev_pos','dep','len','sent_len','all'], required=True, help='Linguistic feature for analysis')
    parser.add_argument('--match', '-m', choices=['pos', 'exact', 'entropy', 'all'], required=True, help='variable to measure accuracy')
    args = parser.parse_args()
    df_eye, df_cloze = load_data()
    spacy_df = pd.read_pickle('/data/mourad/provo_data/spacy_and_gpt2_entropy_data.pkl') #, sep='\t')
    spacy_df = extract_spacy_tokens(spacy_df)
    #for kind in ['resp', 'lm']:
    #    plot_pos_dists(spacy_df, kind=kind)
    #pdb.set_trace()
    spacy_df = eval_synset(spacy_df)
    if args.feature == 'all':
        ents = {}
        for match in ['pos', 'exact', 'synset', 'hypernym', 'entropy']:
            entropy = subplot(df_eye, spacy_df, match)
            ents[match] = entropy
        subplot_entropy(ents)
    else:
        compare_feature(df_eye, spacy_df, args.feature, args.match)
    #exact_match(spacy_df)


