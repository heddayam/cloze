import pandas as pd
import pdb
import swifter
import matplotlib.pyplot as plt
import seaborn as sns


#DATA_DIR = "/data/mourad/provo_data/samples/"

def save_sample(df, feature, match):
    filename = f'match-{match}_feature-{feature}'
    df['lm_pos'] = df.lm_spacy_token.swifter.apply(lambda x: x.pos_)
    df['resp_pos'] = df.resp_spacy_token.swifter.apply(lambda x: x.pos_)
    df['og_pos'] = df.og_spacy_token.swifter.apply(lambda x: x.pos_)
    df = df[['resp_match', 'resp_spacy_token', 'resp_pos', 'lm_match', 'lm_spacy_token','lm_pos', 'og_spacy_token', 'og_pos', f'{feature}', 'Text']]
    df.to_csv(f'{DATA_DIR}{filename}.tsv', sep='\t', index=False)

def set_fig(fig, filename):
    fig.tight_layout()
    fig.set_size_inches(7,7)
    fig.savefig(filename, dpi=400, bbox_inches='tight')
    fig.clf()

def histplot(df, x, xlabel, ylabel, filename, DATA_DIR, binwidth=None, multiple='dodge', rotate=False, shrink=1, hue=None, alpha=1, stat='count'):
    fig, ax = plt.subplots(figsize=(12,8))
    g = sns.histplot(
        data=df,
        x=x,
        hue=hue,
        binwidth=binwidth,
        stat=stat,
        #hue='Treatment',
        multiple=multiple,
        shrink=shrink,
        #bw_adjust=2,
        alpha=alpha,
        edgecolor='black', linewidth=1,
        ax=ax
        ) 
    ax.set_xlabel(xlabel)
    if rotate:
        plt.xticks(rotation=45)
        #ax.set_xticklabels(ax.get_xticklabels(),rotation = 45)
    if ylabel:
        ax.set_ylabel(ylabel)
    fig.savefig(f"{DATA_DIR}{filename}.png", bbox_inches='tight', dpi=400)
    fig.clf()

