# author: Lukas Preis
# date: September 2022
# version: 1.0
# ----------------------- #
# Input: database (.csv), number of topics
# Process: apply LDA algorithm from https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
# Output: topic model represented by clusters thematicly grouped publications; overview (.txt), dominant topcis (.csv)
# ----------------------- #


from stop_words import get_stop_words
import nltk
import pandas as pd
import os
import re
from nltk.stem.porter import *
import gensim
import gensim.corpora as corpora # https://pypi.org/project/gensim/ # it was a pain!!
from gensim.models import CoherenceModel
# import pyLDAvis
# import pyLDAvis.gensim

file_name = "UAM_formatted.csv"
n_topics = 6
title_abstract = 'abstract'


### READ AND PREPARE DATABASE ###

def prepare_database():

    dir = os.path.dirname(os.path.dirname(__file__))
    file = os.path.join(dir, 'input', file_name)

    print("file: {}\nnumber of topcis: {}\n".format(file, n_topics))

    print("read and clean data...")

    df = pd.read_csv(file, delimiter=';', dtype=object, index_col='doi')

    stop_words = get_stop_words('en')
    stemmer = PorterStemmer()

    data_raw = df[title_abstract].dropna().values.tolist()
    dois = df.index.tolist()
    data_token = [re.sub('[^a-zA-Z_]+',' ',d).lower().split(' ') for d in data_raw]  # https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string

    data_stemmed = []
    for d in data_token:
        for s in stop_words:
            while s in d:
                d.remove(s)
        stemmed_words = [stemmer.stem(token) for token in d if len(token) > 3]
        data_stemmed.append(stemmed_words)

    print("data cleaned!\n")

    return data_stemmed, dois, data_raw, dir

### CREATE LDA MODEL ###

def create_LDA_modell():

    print("create LDA model...")

    # Create Dictionary
    id2word = corpora.Dictionary(data_stemmed)

    # Term Document Frequency
    corpus = [id2word.doc2bow(d) for d in data_stemmed]

    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=n_topics, random_state=100,
                                                update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)

    topics = lda_model.print_topics(num_words=20)

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_stemmed, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence() # https://groups.google.com/g/gensim/c/sBy0TDhZCes
    # coherence_lda = None

    perplexity_lda = lda_model.log_perplexity(corpus)

    print("LDA model done!\n")

    return corpus, lda_model, topics, coherence_lda, perplexity_lda


### SHOW TOPICS ###

def show_topics():

    print("TOPICS (perplexity = {:.2f}, coherence = {:.2f})".format(perplexity, coherence)) # perplexity the lower the better
    for k, t in enumerate(topics):
        topic_number = t[0]
        topic_keywords = t[1].replace(" ","").split('+')
        topic_confidence = sum([float(k.split('*')[0]) for k in topic_keywords[:10]])
        rel_weight = rep_documents[k]['rel_weight']
        print("\nNo. {} (rel. weight: {:.3f}, top-10 conf.: {:.2f})".format(topic_number, rel_weight, topic_confidence))
        for k in topic_keywords:
            print("\t{:20s} ({:5.3f})".format(k.split('*')[1].replace('"',''), float(k.split('*')[0])))


### FIND REPRESENTATIVE DOCUMENT ###

def map_topics():

    print("export mapped topcis...")

    # temp = lda_model.get_document_topics(corpus, minimum_probability=0.0) # does the same as lda_model[corpus]
    # temp1 = lda_model.top_topics(corpus, data) # shows the topics in order of weight?
    # temp2 = [a for a in lda_model[corpus]]
    # Init output
    df_topics = pd.DataFrame(dtype=object)

    rep_document = {}
    for i in range(n_topics):
        rep_document[i] = {'pct': 0.0, 'doi': None, title_abstract: None, 'rel_weight': 0.0}

    # Get main topic in each document
    for topic_lda, topic_abstract, topic_doi in zip(lda_model.get_document_topics(corpus, minimum_probability=0.0), data_raw, dois):
        # row = sorted(row, key=lambda x: (x[1]), reverse=True)
        topic_lda_sorted = [{'topic': t[0], 'pct': t[1]} for t in sorted(topic_lda, key=lambda x: x[1], reverse=True)]
        topic_number = str(topic_lda_sorted[0]['topic'])#str(int(topic_lda_sorted[0][0]))
        topic_pct = str(round(topic_lda_sorted[0]['pct'], 3)) #str(round(topic_lda_sorted[0][1], 3))
        df_topics = df_topics.append(pd.Series([topic_doi, topic_number, topic_pct, topic_abstract]), ignore_index=True)

        for t in topic_lda_sorted:
            rep_document[t['topic']]['rel_weight'] += t['pct']
            if t['pct'] > rep_document[t['topic']]['pct']:
                rep_document[t['topic']]['pct'] = t['pct']
                rep_document[t['topic']]['doi'] = topic_doi
                rep_document[t['topic']][title_abstract] = topic_abstract

    for k in range(n_topics):
        rep_document[k]['rel_weight'] = rep_document[k]['rel_weight'] / len(dois)


        # Get the Dominant topic, Perc Contribution and Keywords for each document
    #     for j, (topic_num, prop_topic) in enumerate(row):
    #         if j == 0:  # => dominant topic
    #             wp = lda_model.show_topic(topic_num)
    #             topic_keywords = ", ".join([word for word, prop in wp])
    #             sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
    #         else:
    #             break
    # sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords'])

    df_topics.columns = ['doi', 'dominant_topic', 'pct', title_abstract]
    df_topics.set_index('doi')

    df = pd.read_csv(os.path.join(dir, 'input', file_name), delimiter=';', dtype=object).dropna(subset=[title_abstract])
    df.insert(1, 'dominant_topic', df_topics['dominant_topic'].values.tolist())
    df.insert(2, 'pct', df_topics['pct'].values.tolist())

    df.fillna('N/A').to_csv(os.path.join(dir, 'logfiles', '{}_topics_{}.csv'.format(title_abstract, n_topics)), sep=';', index=False)
    # df_topics.fillna('N/A').to_csv(os.path.join(dir, 'logfiles', '{}_topics_{}.csv'.format(title_abstract, n_topics)), sep=';', index=False)

    df_represent = pd.DataFrame.from_dict(rep_document, orient='index', columns=['pct', 'rel_weight', 'doi', title_abstract])
    df_represent.reset_index(inplace=True)
    df_represent.rename(columns={'index': 'dominant_topic'}, inplace=True)

    df_represent.fillna('N/A').to_csv(os.path.join(dir, 'logfiles', 'topic_representative_{}_{}.csv'.format(title_abstract, n_topics)), sep=';', index=False)

    # Add original text to the end of the output
    # contents = pd.Series(data)
    # sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    # Format
    # df_dominant_topic = sent_topics_df.reset_index()
    # df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

    # https://stackoverflow.com/questions/27842613/pandas-groupby-sort-within-groups
    # df_represent = df_topics.sort_values(['dominant_topic','contribution_percentage'],ascending=False).groupby('dominant_topic').head(2)

    print("export done!\n")

    return rep_document

# View
# print(corpus[:1])
# print([[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]])
# print(lda_model.print_topics())

# pyLDAvis.enable_notebook()
# vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)


if __name__ == '__main__': # https://www.data-science-architect.de/__name____main__/
    data_stemmed, dois, data_raw, dir = prepare_database()
    corpus, lda_model, topics, coherence, perplexity = create_LDA_modell()
    rep_documents = map_topics()
    show_topics()
