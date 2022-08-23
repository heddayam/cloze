from calendar import c
import pandas as pd
# import plotly.express as px
import pdb
import spacy
import swifter
from transformers import pipeline, set_seed, AutoTokenizer
import plotly.express as px
import plotly.graph_objs as go
import scipy.stats as ss
import seaborn as sns
from nltk.corpus import wordnet as wn



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
    return df_eye, df_cloze

def match_word(x):
    current_index = x.Word_Number-1
    match = x.text_spacy[current_index] 
    offset = 0
    # while match.text.lower() != x.Word.lower() and ((current_index - offset) > 0 or (current_index + offset) < len(x.text_spacy)):
    while match.text.lower() != x.Word.lower() and (current_index + offset) < len(x.text_spacy):
        offset += 1
        try:
            match = x.text_spacy[current_index + offset] #.text.lower()
            if match.text.lower() == x.Word.lower(): break
        except: continue
        # try:
        #     match = x.text_spacy[current_index - offset] #.text.lower()
        # except: continue
    return x.text_spacy[match.i - 1], match

def replace_word_with_response(x):
    new_text = x.text_spacy[:x.spacy_token.i].text + f" {x.Response}"
    new_spacy = nlp(new_text)
    return new_spacy[-1]

# def replace_word_with_gpt_completion(x, generator):
#     text = x.text_spacy[:x.spacy_token.i].text
#     output = generator(text, max_new_tokens = 1, temperature=0.0001)
#     new_spacy = nlp(output[0]['generated_text'])
#     return new_spacy[-1]

def get_lm_completion(seqs):
    generator = pipeline('text-generation', model='distilgpt2')
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    set_seed(42)
    print("Generating LM Completions, might take a couple min...")
    output = generator(seqs.tolist(), max_new_tokens = 1, temperature=0.0001, pad_token_id=tokenizer.eos_token_id)
    output = [o[0]['generated_text'] for o in output]
    return output

def get_lm_spacy(x):
    new_spacy = nlp(x)
    return new_spacy[-1]

def spacy_analysis(df):
    text_spacy = pd.Series(df.Text.unique()).swifter.apply(nlp)
    text_spacy = pd.DataFrame(text_spacy, columns=['text_spacy'])
    text_spacy.index += 1
    df = df.merge(text_spacy, left_on='Text_ID', right_index=True)
    res = df.swifter.apply(match_word, axis=1)
    prev_spacy, curr_spacy = list(zip(*res.tolist()))

    df['spacy_token'] = curr_spacy
    df['preceding_token_spacy'] = prev_spacy

    pdb.set_trace()

    df['spacy_word'] = df.spacy_token.swifter.apply(lambda x: x.text.lower())
    ## IMPORTANT... Here I'm ignoring hyphenated strings and a couple other unusual constructions (427 isntances total ignored)
    df = df[df.spacy_word == df.Word]
    # TODO UNCOMMENT NEXT THREE LINES
    df['response_spacy_token'] =  df.swifter.apply(replace_word_with_response, axis=1)
    df['response_text'] = df.response_spacy_token.swifter.apply(lambda x: x.text)
    ## IMPORTANT... Here I'm ignoring hyphenated strings and a couple other unusual constructions (427 isntances total ignored)
    df = df[df.response_text == df.Response]

    # get LM next word prediction
    # pdb.set_trace()
    lm_df = df[['Word_Unique_ID', 'Text', 'text_spacy', 'spacy_token']].drop_duplicates()
    lm_df['prompt'] = lm_df.swifter.apply(lambda x: x.text_spacy[:x.spacy_token.i].text, axis=1)
    lm_df['lm_generated'] = get_lm_completion(lm_df.prompt)
    lm_df['lm_spacy_token'] = lm_df.lm_generated.swifter.apply(get_lm_spacy)
    # lm_df['lm_response_text'] = lm_df.lm_spacy_token.swifter.apply(lambda x: x.text)
    lm_df = lm_df[['Word_Unique_ID', 'lm_spacy_token']]
    df = df.merge(lm_df, on='Word_Unique_ID')
    return df
    # df['text_spacy'] = df.Text.swifter.apply(nlp)
    # pdb.set_trace()

def compare_pos(eye_df, spacy_df):
    spacy_df.to_csv('/data/mourad/provo_data/spacy_and_gpt2_data.tsv', sep='\t', index=False)
    pdb.set_trace()
    spacy_df['lm_pos_match'] = spacy_df.swifter.apply(lambda x: int(x.lm_spacy_token.pos_ == x.spacy_token.pos_), axis=1)
    print('POS matches between LM completions and original:')
    print(spacy_df['lm_pos_match'].value_counts(normalize=True))

    spacy_df['pos_match'] = spacy_df.swifter.apply(lambda x: int(x.response_spacy_token.pos_ == x.spacy_token.pos_), axis=1)
    spacy_df['lm_response_pos_match'] = spacy_df.swifter.apply(lambda x: int(x.response_spacy_token.pos_ == x.lm_spacy_token.pos_), axis=1)
    
    spacy_matches_df = spacy_df.groupby(['Word_Unique_ID', 'preceding_token_spacy'])['pos_match', 'lm_response_pos_match', 'Word_Number', 'lm_pos_match'].mean()
    # spacy_matches_df = spacy_df.groupby('Word_Unique_ID')['pos_match', 'lm_response_pos_match', 'Word_Number', 'lm_pos_match'].mean()
    spacy_matches_df = spacy_matches_df.reset_index()
    
    df = spacy_matches_df.merge(eye_df[['Word_Unique_ID', 'POSMatch']], on='Word_Unique_ID')    
    print(df.describe())

    counts = df.groupby('Word_Number').Word_Unique_ID.nunique().reset_index(name='count')
    df = df.merge(counts, on="Word_Number")
    df = df[df['count'] > 1]
    df = df.drop('count', axis=1)
    counts = counts[counts > 1].dropna()['count']

    lm_response_pos_match_se =  df.groupby('Word_Number').lm_response_pos_match.std().to_numpy() / counts.to_numpy()
    pos_match_se = df.groupby('Word_Number').pos_match.std().to_numpy() / counts.to_numpy()
    lm_pos_match_se = df.groupby('Word_Number').lm_pos_match.std().to_numpy() / counts.to_numpy()

    length_df = df.groupby('Word_Number').mean()
    length_df = length_df.reset_index()
    length_df['counts'] = counts.tolist()
    
    length_df['lm_response_pos_match_se'] = lm_response_pos_match_se
    length_df['pos_match_se'] = pos_match_se
    length_df['lm_pos_match_se'] = lm_pos_match_se

   

    df['prev_pos'] = df.preceding_token_spacy.swifter.apply(lambda x: x.pos_)
    df = df.drop(['preceding_token_spacy', 'Word_Number'], axis=1)
    prev_pos_df = df.groupby('prev_pos').mean()
    prev_pos_df = prev_pos_df.reset_index()

    print('Mean response POS accuracy correlation with context length:')
    print(ss.spearmanr(length_df.pos_match, length_df.Word_Number))
    fig = px.scatter(length_df, x='Word_Number', y='pos_match', hover_data=['counts'], error_y="pos_match_se")
    fig.write_html('/data/mourad/provo_data/plots/pos_match_length.html')

    print('Mean LM POS and response POS accuracy correlation with context length:')
    print(ss.spearmanr(length_df.lm_response_pos_match, length_df.Word_Number))
    fig = px.scatter(length_df, x='Word_Number', y='lm_response_pos_match', hover_data=['counts'], error_y="lm_response_pos_match_se")
    fig.write_html('/data/mourad/provo_data/plots/lm_response_pos_match_length.html')

    print('Mean LM POS accuracy correlation with context length:')
    print(ss.spearmanr(length_df.lm_pos_match, length_df.Word_Number))
    fig = px.scatter(length_df, x='Word_Number', y='lm_pos_match', hover_data=['counts'], error_y="lm_pos_match_se")
    fig.write_html('/data/mourad/provo_data/plots/lm_pos_match_length.html')
    pdb.set_trace()

    # x = length_df.Word_Number
    # y = length_df.pos_match
    # y_upper = length_df.pos_match + length_df.pos_match_se
    # y_lower = length_df.pos_match - length_df.pos_match_se

    # fig = go.Figure([
    #     go.Scatter(
    #         x=x,
    #         y=y,
    #         line=dict(color='rgb(0,100,80)'),
    #         mode='lines'
    #     ),
    #     go.Scatter(
    #         x=x.tolist()+x.tolist()[::-1], # x, then x reversed
    #         y=y_upper.tolist()+y_lower.tolist()[::-1], # upper, then lower reversed
    #         fill='toself',
    #         fillcolor='rgba(0,100,80,0.2)',
    #         line=dict(color='rgba(255,255,255,0)'),
    #         hoverinfo="skip",
    #         showlegend=False
    #     )
    # ])

    # fig = px.histogram(prev_pos_df, x='prev_pos', y='pos_match', histfunc='avg')
    # fig.write_html('/data/mourad/provo_data/plots/pos_match_preceding.html')

    # fig = px.histogram(prev_pos_df, x='prev_pos', y='lm_response_pos_match', histfunc='avg')
    # fig.write_html('/data/mourad/provo_data/plots/lm_response_pos_match_preceding.html')

    # fig = px.histogram(prev_pos_df, x='prev_pos', y='lm_pos_match', histfunc='avg')
    # fig.write_html('/data/mourad/provo_data/plots/lm_pos_match_preceding.html')

    df = df.reset_index()
    val_list = ['pos_match', 'lm_response_pos_match', 'lm_pos_match']
    df = pd.melt(df, id_vars = ['index', 'prev_pos'] , value_vars=val_list)


    # for data_name in ['pos_match', 'lm_response_pos_match', 'lm_pos_match']:
    ax = sns.barplot(
        data=df, 
        x="prev_pos", 
        y="value", 
        hue="variable", 
        errwidth=1, capsize=0.1, ci=68
        )
    ax.set_xlabel('Part of Speech')
    ax.set_ylabel('Mean POS Match Accuracy')
    ax.set_xticklabels(ax.get_xticklabels(),rotation = 45)
    # ax.set(ylim=(200, 240))
    fig = ax.get_figure()
    fig.savefig(f"{DATA_DIR}all_pos_preceding.png", dpi=400)
    # fig.clf()
    # ax.cla()

    pdb.set_trace()

def compare_dep(eye_df, spacy_df):
    # spacy_df.to_pickle('/data/mourad/provo_data/spacy_and_gpt2_data.pkl') #, sep='\t', index=False)
    
    pdb.set_trace()
    
    spacy_df['lm_pos_match'] = spacy_df.swifter.apply(lambda x: int(x.lm_spacy_token.head.pos_ == x.spacy_token.head.pos_), axis=1)
    print('POS matches between LM completions and original:')
    print(spacy_df['lm_pos_match'].value_counts(normalize=True))

    spacy_df['pos_match'] = spacy_df.swifter.apply(lambda x: int(x.response_spacy_token.head.pos_ == x.spacy_token.head.pos_), axis=1)
    spacy_df['lm_response_pos_match'] = spacy_df.swifter.apply(lambda x: int(x.response_spacy_token.head.pos_ == x.lm_spacy_token.head.pos_), axis=1)
    
    spacy_matches_df = spacy_df.groupby(['Word_Unique_ID', 'preceding_token_spacy'])['pos_match', 'lm_response_pos_match', 'Word_Number', 'lm_pos_match'].mean()
    # spacy_matches_df = spacy_df.groupby('Word_Unique_ID')['pos_match', 'lm_response_pos_match', 'Word_Number', 'lm_pos_match'].mean()
    spacy_matches_df = spacy_matches_df.reset_index()
    
    df = spacy_matches_df.merge(eye_df[['Word_Unique_ID', 'POSMatch']], on='Word_Unique_ID')    
    print(df.describe())

    counts = df.groupby('Word_Number').Word_Unique_ID.nunique().reset_index(name='count')
    df = df.merge(counts, on="Word_Number")
    df = df[df['count'] > 1]
    df = df.drop('count', axis=1)
    counts = counts[counts > 1].dropna()['count']

    lm_response_pos_match_se =  df.groupby('Word_Number').lm_response_pos_match.std().to_numpy() / counts.to_numpy()
    pos_match_se = df.groupby('Word_Number').pos_match.std().to_numpy() / counts.to_numpy()
    lm_pos_match_se = df.groupby('Word_Number').lm_pos_match.std().to_numpy() / counts.to_numpy()

    length_df = df.groupby('Word_Number').mean()
    length_df = length_df.reset_index()
    length_df['counts'] = counts.tolist()
    
    length_df['lm_response_pos_match_se'] = lm_response_pos_match_se
    length_df['pos_match_se'] = pos_match_se
    length_df['lm_pos_match_se'] = lm_pos_match_se

   

    df['prev_pos'] = df.preceding_token_spacy.swifter.apply(lambda x: x.head.pos_)
    df = df.drop(['preceding_token_spacy', 'Word_Number'], axis=1)
    prev_pos_df = df.groupby('prev_pos').mean()
    prev_pos_df = prev_pos_df.reset_index()

    print('Mean response POS accuracy correlation with context length:')
    print(ss.spearmanr(length_df.pos_match, length_df.Word_Number))
    fig = px.scatter(length_df, x='Word_Number', y='pos_match', hover_data=['counts'], error_y="pos_match_se")
    fig.write_html('/data/mourad/provo_data/plots/dep_pos_match_length.html')

    print('Mean LM POS and response POS accuracy correlation with context length:')
    print(ss.spearmanr(length_df.lm_response_pos_match, length_df.Word_Number))
    fig = px.scatter(length_df, x='Word_Number', y='lm_response_pos_match', hover_data=['counts'], error_y="lm_response_pos_match_se")
    fig.write_html('/data/mourad/provo_data/plots/lm_response_dep_pos_match_length.html')

    print('Mean LM POS accuracy correlation with context length:')
    print(ss.spearmanr(length_df.lm_pos_match, length_df.Word_Number))
    fig = px.scatter(length_df, x='Word_Number', y='lm_pos_match', hover_data=['counts'], error_y="lm_pos_match_se")
    fig.write_html('/data/mourad/provo_data/plots/lm_dep_pos_match_length.html')
    pdb.set_trace()

    # x = length_df.Word_Number
    # y = length_df.pos_match
    # y_upper = length_df.pos_match + length_df.pos_match_se
    # y_lower = length_df.pos_match - length_df.pos_match_se

    # fig = go.Figure([
    #     go.Scatter(
    #         x=x,
    #         y=y,
    #         line=dict(color='rgb(0,100,80)'),
    #         mode='lines'
    #     ),
    #     go.Scatter(
    #         x=x.tolist()+x.tolist()[::-1], # x, then x reversed
    #         y=y_upper.tolist()+y_lower.tolist()[::-1], # upper, then lower reversed
    #         fill='toself',
    #         fillcolor='rgba(0,100,80,0.2)',
    #         line=dict(color='rgba(255,255,255,0)'),
    #         hoverinfo="skip",
    #         showlegend=False
    #     )
    # ])

    # fig = px.histogram(prev_pos_df, x='prev_pos', y='pos_match', histfunc='avg')
    # fig.write_html('/data/mourad/provo_data/plots/pos_match_preceding.html')

    # fig = px.histogram(prev_pos_df, x='prev_pos', y='lm_response_pos_match', histfunc='avg')
    # fig.write_html('/data/mourad/provo_data/plots/lm_response_pos_match_preceding.html')

    # fig = px.histogram(prev_pos_df, x='prev_pos', y='lm_pos_match', histfunc='avg')
    # fig.write_html('/data/mourad/provo_data/plots/lm_pos_match_preceding.html')

    df = df.reset_index()
    val_list = ['pos_match', 'lm_response_pos_match', 'lm_pos_match']
    df = pd.melt(df, id_vars = ['index', 'prev_pos'] , value_vars=val_list)


    # for data_name in ['pos_match', 'lm_response_pos_match', 'lm_pos_match']:
    ax = sns.barplot(
        data=df, 
        x="prev_pos", 
        y="value", 
        hue="variable", 
        errwidth=1, capsize=0.1, ci=68
        )
    ax.set_xlabel('Part of Speech')
    ax.set_ylabel('Mean POS Match Accuracy')
    ax.set_xticklabels(ax.get_xticklabels(),rotation = 45)
    # ax.set(ylim=(200, 240))
    fig = ax.get_figure()
    fig.savefig(f"{DATA_DIR}all_dep_pos_preceding.png", dpi=400)
    # fig.clf()
    # ax.cla()

    pdb.set_trace()

def exact_match(spacy_df):
    # resp_df = spacy_df[['Word_Unique_ID', 'Text_ID', 'Text', 'Word_Number', 'Word', 'response_text']]
    # lm_df = spacy_df[['Word_Unique_ID', 'Text_ID', 'Text', 'Word_Number', 'Word', 'lm_spacy_token']]
    resp_df = spacy_df[['Word_Unique_ID', 'Word', 'response_text']]
    lm_df = spacy_df[['Word_Unique_ID', 'Word', 'lm_spacy_token']]
    lm_df = lm_df.drop_duplicates()
    resp_df['response_exact_match'] = resp_df.Word.str.lower() == resp_df.response_text.str.lower()
    lm_df['lm_exact_match'] = lm_df.Word.str.lower() == lm_df.lm_spacy_token.str.lower()
    # print(f'Response exact match acc: {resp_df.response_exact_match.sum()/len(resp_df)}')
    # print(f'DistilGPT2 exact match acc: {lm_df.lm_exact_match.sum()/len(lm_df)}')
    # pdb.set_trace()
    # resp_df.to_csv('/data/mourad/provo_data/samples/human_exact_matches.tsv', sep='\t', index=False)
    # lm_df.to_csv('/data/mourad/provo_data/samples/lm_exact_matches.tsv', sep='\t', index=False)
    records = []
    for col in [resp_df.response_exact_match, lm_df.lm_exact_match]:
        col *= 100
        record = {
            "Mean": round(col.mean(), 2),
            "Std. Dev": round(col.std(), 2),
        }
        records.append(record)
    stats_df = pd.DataFrame.from_dict(records)
    index_labels = [
        'Cloze Response Accuracy %',
        'DistilGPT2 Completion Accuracy %',
    ]
    stats_df.index = index_labels
    print(stats_df)
    pdb.set_trace()
    stats_df.to_csv(f"/data/mourad/provo_data/analyses/exact_match.csv")

def check_sem_cat(spacy_df):
    resp_df = spacy_df[['Word_Unique_ID', 'Word', 'response_text']]
    lm_df = spacy_df[['Word_Unique_ID', 'Word', 'lm_spacy_token']]
    lm_df = lm_df.drop_duplicates()

    def check_synset(word):
        res = wn.synsets(word)
        res = [r.name().split('.')[0] for r in res]
        return word in res

    resp_df['response_synset_match'] = resp_df.Word.swifter.apply(check_synset)
    lm_df['lm_synset_match'] = lm_df.Word.swifter.apply(check_synset)

    records = []
    for col in [resp_df.response_synset_match, lm_df.lm_synset_match]:
        col *= 100
        record = {
            "Mean": round(col.mean(), 2),
            "Std. Dev": round(col.std(), 2),
        }
        records.append(record)
    stats_df = pd.DataFrame.from_dict(records)
    index_labels = [
        'Cloze Response Synset Accuracy %',
        'DistilGPT2 Completion Synset Accuracy %',
    ]
    stats_df.index = index_labels
    print(stats_df)
    stats_df.to_csv(f"/data/mourad/provo_data/analyses/synset_match.csv")

    pdb.set_trace()


if __name__ == '__main__':
    df_eye, df_cloze = load_data()
    reuse_preds = True
    if reuse_preds:
        df_spacy = pd.read_csv('/data/mourad/provo_data/spacy_and_gpt2_data.tsv', sep='\t')
    else:
        nlp = spacy.load("en_core_web_sm")
        df_spacy = spacy_analysis(df_cloze)

    pdb.set_trace()
    # compare_pos(df_eye, df_spacy)
    # compare_dep(df_eye, df_spacy)
    exact_match(df_spacy)
    # check_sem_cat(df_spacy)






# df_eye = df_eye.groupby('Text_ID')['InflectionMatch', 'POSMatch', 'LSA_Response_Match_Score'].mean()

# fig = px.scatter(df_eye, x=df_eye.index, y='POSMatch')
# fig.write_html('/data/mourad/provo_data/plots/posmatch.html')

# fig = px.scatter(df_eye, x=df_eye.index, y='InflectionMatch', hover_data=[])
# fig.write_html('/data/mourad/provo_data/plots/inflectionmatch.html')

# fig = px.scatter(df_eye, x=df_eye.index, y='LSA_Response_Match_Score', hover_data=[])
# fig.write_html('/data/mourad/provo_data/plots/lsamatch.html')
