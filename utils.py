import pandas as pd
import pdb
import swifter

DATA_DIR = "/data/mourad/provo_data/samples/"

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
