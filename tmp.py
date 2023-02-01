gpt3_df = df[['Word_Unique_ID', 'Text', 'text_spacy', 'target_i']].drop_duplicates().reset_index(drop=True)
    #gpt3_df['prompt'] = gpt3_df.swifter.apply(lambda x: x.text_spacy[:x.target_i].text, axis=1)
    gpt3_df = gpt3_df.merge(tokenized_prompts, on="Word_Unique_ID", how='inner')
    #gpt3_df['prompt'] = tokenized_prompts.prompt
    if trigram_context: 
        gpt3_df['prompt'] = gpt3_df.prompt.swifter.apply(lambda x: " ".join(x.split()[-2:]))
    generated_words, entropies, top_50_words, top_50_probs = get_gpt3_completion_logits(gpt3_df.prompt)
    gpt3_df['gpt3_generated'] = generated_words
    gpt3_df['gpt3_entropy'] = entropies 
    gpt3_df['gpt3_50_generated'] = top_50_words
    gpt3_df['gpt3_50_probs'] = top_50_probs
    gpt3_50_tokens = gpt3_df.swifter.apply(get_gpt3_top_spacy, axis=1)
    gpt3_df = pd.concat([gpt3_df.reset_index(drop=True), pd.DataFrame.from_records(gpt3_50_tokens)], axis=1) 

