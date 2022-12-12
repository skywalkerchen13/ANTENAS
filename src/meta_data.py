# author: Lukas Preis
# date: 9.3.2021
# version: 1.5
# update: include composition of database

import pandas as pd
from statistics import *
from stop_words import get_stop_words
from collections import Counter
import os
from nltk import ngrams
from nltk.stem.porter import *
import operator
import math
import xml.etree.ElementTree as ET

# to log console: Run > Edit Configurations > Logs > Save console output to file

# @todo: Some databases have authors in capital letters only - adapt for that!

filename = "Aviation_Library_2016-20.csv"

token_length = 2
fields = ['title']
n_top_papers = 5
n_keywords = 40
n_years_keywords = 0
n_authors = 10
n_top_authors = 0
n_coauthors = 5
n_authors_keywords = 5
fullname = True
count_method = 'papers' # values: occurences | papers
include_citation = False
include_doc_type = False

### READ AND PREPARE DATABASE ###

dir = os.path.dirname(os.path.dirname(__file__))
file = os.path.join(dir, 'input', filename)
df = pd.read_csv(file, delimiter=';', dtype=object, index_col='doi')

stop_words = get_stop_words('en')
stemmer = PorterStemmer()
costum_words = ["use","analysis","approach","study","case",
                "development","evaluation","also",
                "effect","new","impact","result","article",
                "assessment","problem","can","paper",
                "may","propose","will","however","one","research","based","two",
                "three","found","present","investigate","this","examine"]
stop_words = stop_words + [stemmer.stem(token) for token in costum_words]
text_cleaner = [["-"," "],[":",""],["'s",""],[",",""],["?",""],[")",""],["(",""],
                [".",""],['"',''],["'",""],["’",""],["  "," "],["”",""],["“",""],["=",""]]

sources_raw = ET.parse(os.path.join(dir, 'src', 'source_dict.xml')).getroot()
temp_keys = [s.get('name') for s in sources_raw if s.tag == 'source']
temp_values = [[d.get('code') for d in s.find('dois')] for s in sources_raw if s.tag == 'source']
source_dict = dict(zip(temp_keys, temp_values))
temp_keys = [s.get('name') for s in sources_raw if s.tag == 'publisher']
temp_values = [['10.{}/'.format(d.get('code')) for d in s.find('dois')] for s in sources_raw if s.tag == 'publisher']
publisher_dict = dict(zip(temp_keys, temp_values))

# small aircraft DB: Continue at line 50 of dois unkown

# source_dict = {'TRA': ['10.1016/j.tra', '10.1016/0965-8564', '10.1016/S0965-8564'],
#                'TRB': ['10.1016/j.trb', '10.1016/0191-2615', '10.1016/S0191-2615'],
#                'TRC': ['10.1016/j.trc', '10.1016/0968-090X', '10.1016/S0968-090X'],
#                'TRD': ['10.1016/j.trd', '10.1016/S1361-9209'], 'TRE': ['10.1016/j.tre', '10.1016/S1366-5545'],
#                'TRF': ['10.1016/j.trf', '10.1016/S1369-8478'], 'IJST': '10.1080/1556831',
#                'JATM': ['10.1016/j.jairtraman', '10.1016/S0969-6997', '10.1016/0969-6997'],
#                'JTR': '10.1177/0047287', 'STrans': ['10.1007/s11116', '10.1007/BF0', '10.1023/A:10', '10.1023/B:PORT.000'],
#                'TBS': '10.1016/j.tbs', 'TM': ['10.1016/j.tourman', '10.1016/S0261-5177', '10.1016/0261-5177'],
#                'TS': '10.1287/trsc', 'TTRv': ['10.1080/0144164', '10.1080/71'],
#                'ARC': '10.2514/6.', 'OR': '10.1287/opre.', 'TRR': ['10.3141/', '10.1177/03611981'],
#                'P&G': '10.17645/pag.', 'FENNIA': '10.11143/FENNIA.'}

### FUNCTIONS ###

def count_characters(string):
    count = len(string)
    return count

def count_words(string):
    words = string.split(" ")
    count = len(words)
    return count

def count_sentences(string):
    sentences = string.count(".")
    return sentences

def unsupervised_count(df,field,token_length):
    c = Counter()
    word_count = 0
    for element in df[field]:
        for cleaner in text_cleaner:
            element = element.replace(cleaner[0],cleaner[1])
        single_words = element.lower().split(" ")
        stemmed_words = [stemmer.stem(token) for token in single_words]
        for s in stop_words:
            while s in stemmed_words:
                stemmed_words.remove(s)
        for s in stemmed_words:
            if s.isnumeric():
                stemmed_words.remove(s)
        # stemmed_words = [stemmer.stem(token) for token in single_words]

        word_count += len(stemmed_words) + 1 - token_length
        if count_method == 'occurences':
            n_grams = list(ngrams(stemmed_words, token_length))
        elif count_method == 'papers':
            n_grams = list(set(list(ngrams(stemmed_words, token_length))))
        c.update([' '.join(n) for n in n_grams])
    keywords = dict(c)
    # word_count = sum(keywords.values())
    # for s in stop_words:
    #     if s in keywords.keys():
    #         keywords.pop(s)
    return keywords, word_count

def modify_name(author):
    if author is not "":
        names = author.split("+")
        try:
            if fullname:
                name = names[0] + " " + names[-1]
            else:
                name = names[0][0] + ". " + names[-1]
        except:
            print("problem with author name {}".format(author))
            exit()
    else:
        return None
    return name

def author_count(df):
    authors_per_paper = []
    c_author = Counter()
    if include_citation:
        authors_citations = {}
        authors_start_year = {}
    else:
        authors_citations = None
        authors_start_year = None

    if include_citation:
        for paper_authors, paper_citations, paper_year in zip(df['authors'], df['citations'], df['year']):
            c_author.update([modify_name(names) for names in paper_authors.split(" ")])
            for name in [modify_name(names) for names in paper_authors.split(" ")]:
                if name in authors_citations.keys():
                    authors_citations[name] += paper_citations
                    authors_start_year[name] = min(authors_start_year[name], int(paper_year))
                else:
                    authors_citations[name] = paper_citations
                    authors_start_year[name] = int(paper_year)
            authors_per_paper.append(len(paper_authors.split(" ")))
    else:
        for paper_authors in df['authors']:
            c_author.update([modify_name(names) for names in paper_authors.split(" ")])
            authors_per_paper.append(len(paper_authors.split(" ")))
    authors = dict(c_author)
    total_authors = len(c_author)

    if include_citation:
        authors_citations = dict(sorted(authors_citations.items(), key=lambda item: item[1], reverse=True))
        authors_citescore = dict(zip(authors.keys(), [c/(int(max(df['year']))-y+1) for c,y in zip(authors_citations.values(), authors_start_year.values())]))
        authors_citescore = dict(sorted(authors_citescore.items(), key=lambda item: item[1], reverse=True))
    else:
        authors_citescore = None

    return authors, total_authors, authors_per_paper, authors_citations, authors_citescore, authors_start_year

def count_sources(df, dict):
    sources = {}
    for doi in df.index.values.tolist():
        if str(doi) == 'nan':
            continue
        for k,v in dict.items():
            if not isinstance(v, list):
                v = list([v])
            for v_i in v:
                if v_i in str(doi):
                    if k in sources.keys():
                        if include_citation:
                            sources[k]['count'] += 1
                            try:
                                temp1 = df.loc[doi, 'citations']
                                sources[k]['citations'] += int(df.loc[doi, 'citations'])
                            except:
                                temp = 0
                        else:
                            sources[k] += 1
                    else:
                        if include_citation:
                            sources[k] = {'count': 1, 'citations': df.loc[doi, 'citations']}
                        else:
                            sources[k] = 1
                    break
    return sources

### AVAILABLE DATA ###

print(file)
print("\nDatabase has {} sets and {} fields\n".format(*df.shape))
for field in df.columns:
    filt = df[field].isnull()
    print("Field {} is missing {} sets".format(field, df.loc[filt, field].shape[0]))

if not include_citation:
    if 'citations' in df.columns:
        df.drop(columns=['citations'], inplace=True)
        print("(Not considering field citations)")
if not include_doc_type:
    if 'doc_type' in df.columns:
        df.drop(columns=['doc_type'], inplace=True)
        print("(Not considering field doc_type)")

df.dropna(axis='index', how='any', inplace=True) # drop all rows that are not complete

print("\nEvaluating {} complete data sets\n".format(df.shape[0]))


### META INFORMATION ###

title_characters = df['title'].apply(count_characters)
title_words = df['title'].apply(count_words)
abstract_characters = df['abstract'].apply(count_characters)
abstract_words = df['abstract'].apply(count_words)
abstract_sentences = df['abstract'].apply(count_sentences)
meta_data_bins = [title_characters,title_words,
                  abstract_characters,abstract_words,abstract_sentences]
meta_data_labels = [['Character','title'],['Word','title'],
                    ['Character','abstract'],['Word','abstract'],['Sentence','abstract']]

meta_data_counts = []
for bin in meta_data_bins:
    meta_data_counts.append([int(median(bin)), min(bin), max(bin)])

print("\nMeta text data and word count:\n")
for i in range(len(meta_data_bins)):
    print("{}s per {}: {} (shortest: {}, longest: {})".
          format(*meta_data_labels[i],*meta_data_counts[i]))

if 'citations' in df.columns:
    df['citations'] = df['citations'].astype(int)
    print("\n\nAverage number of citations over all papers: {:.1f}".format(mean(df['citations'])))
    h_index = 0
    while True:
        if sum(h >= h_index for h in df['citations'].values.tolist()) >= h_index:
            h_index += 1
        else:
            h_index -= 1
            break
    print("Hirsch-Index of paper collection: h={}".format(h_index))
    max_index = df['citations'].astype(float).idxmax()
    print("\nMost cited publication ({} citations): \n{}\n{}\n{}"
          .format(int(float(df.loc[max_index, 'citations'])), df.loc[max_index, 'title'],
                  df.loc[max_index, 'authors'].replace(' ',', ').replace('+',' '), df.loc[max_index, 'year']))
    top_papers = df.sort_values(by='citations', ascending=False).head(n_top_papers)[['year','citations','title']]
    print("\nTop papers by citation:")
    for i in range(n_top_papers):
        temp = top_papers.index[i]
        if not isinstance(top_papers.index[i], str):
            print("{:40s}{:5d}".format(top_papers['title'].iloc[i][:30], top_papers['citations'].iloc[i]))
        else:
            print("{:40s}{:5d}".format(top_papers.index[i], top_papers['citations'].iloc[i]))
    print("\nTop papers by citations per year (cite-score):")
    top_citescores = dict(zip(df.index, [c/(max(df['year'].astype(int))-y+1) for c,y in zip(df['citations'].values, df['year'].astype(int))]))
    top_citescores = dict(sorted(top_citescores.items(), key=lambda item: item[1], reverse=True))
    for i in range(n_top_papers):
        temp = df.loc[list(top_citescores.keys())[i]]['year']
        print("{:40s}{:5.1f} (publication year: {}, total citations: {:3d})"
              .format(*list(top_citescores.items())[i], df.loc[list(top_citescores.keys())[i]]['year'], df.loc[list(top_citescores.keys())[i]]['citations']))

if 'doc_type' in df.columns:
    doc_types = df['doc_type'].unique()
    print("\n\nOverview of document types ({} different types):".format(len(doc_types)))
    doc_types = df.groupby(['doc_type']).size().sort_values(ascending=False)
    temp_length = max([len(i) for i,n in doc_types.items() if n>10])
    temp_n = 0
    for i,n in doc_types.items():
        if n > 10:
            print("{:{}}\t{}".format(i, temp_length, n))
        else:
            temp_n += n
    print("{:{}}\t{}".format("Miscellaneous", temp_length, temp_n))


year_counts = df['year'].value_counts()
print("\n\nPublications between years {} and {}".format(min(df['year']), max(df['year'])))
print("(most active: {}, least active: {})".format(year_counts.index[0],year_counts.index[-1]))
print(year_counts.sort_index().to_string())

if 'citations' in df.columns:
    df['citations'] = df['citations'].astype(float).astype(int)
    years = sorted(df['year'].unique().tolist())
    citations = [df.loc[df['year'] == year, 'citations'].sum() for year in years]

    print("\nAccumulated citations of papers published between years {} and {}".format(min(df['year']), max(df['year'])))
    print("(publishing years with papers most cited: {}, least cited: {})".
          format(years[citations.index(max(citations))],
                 years[citations.index(min(citations))]))
    citation_counts = pd.Series(citations, index=years)
    print(citation_counts.sort_index().to_string())

    averages = [c/y for c,y in zip(citations, year_counts.sort_index().tolist())]
    print("\nAverage citation per publication published between years {} and {}".format(min(df['year']), max(df['year'])))
    print("(publishing years with best citation average: {}, worst citation average: {})".
          format(years[averages.index(max(averages))],
                 years[averages.index(min(averages))]))
    averages_counts = pd.Series(["{:.1f}".format(a) for a in averages], index=years)
    print(averages_counts.sort_index().to_string())

sources = count_sources(df, source_dict)
publishers = count_sources(df, publisher_dict)
# best_source = max(sources.items(), key=operator.itemgetter(1))[0]

if include_citation:
    print("\n\nComposition of database\n({} papers from {} know sources, {} papers from unknown sources, {} papers without doi)"
          .format(sum([v['count'] for v in sources.values()]), len(sources.keys()),
                  df.loc[~df.index.isnull()].shape[0]-sum([v['count'] for v in sources.values()]), df.loc[df.index.isnull()].shape[0]))

    print("\n".join("{:12s}{:4d} (total citations: {:4d}, cite-score: {:4.1f})"
                    .format(k, v['count'], v['citations'], v['citations']/v['count']) for k, v in
                    sorted(sources.items(), key=lambda item: item[1]['count'], reverse=True)))
    temp_count = df.shape[0]-sum([v['count'] for v in sources.values()])
    temp_citations = sum(df['citations'])-sum([v['citations'] for v in sources.values()])
    print("Other       {:4d} (total citations: {:4d}, cite-score: {:4.1f})".format(temp_count, temp_citations, temp_citations/temp_count))
    temp_count = df.loc[~df.index.isnull()].shape[0]-sum([v['count'] for v in sources.values()])
    temp_citations = sum(df.loc[~df.index.isnull()]['citations']) - sum([v['citations'] for v in sources.values()])
    print("Unknown DOI {:4d} (total citations: {:4d}, cite-score: {:4.1f})".format(temp_count, temp_citations, temp_citations/temp_count))
    temp_count = df.loc[df.index.isnull()].shape[0]
    temp_citations = sum(df.loc[df.index.isnull()]['citations'])
    if temp_count > 0:
        print("No DOI      {:4d} (total citations: {:4d}, cite-score: {:4.1f})".format(temp_count, temp_citations, temp_citations/temp_count))

else:
    print("\n\nComposition of database\n({} papers from {} know sources, {} papers from unknown sources, {} papers without doi)"
          .format(sum(sources.values()), len(sources.keys()), df.loc[~df.index.isnull()].shape[0]-sum(sources.values()), df.loc[df.index.isnull()].shape[0]))
    print("\n".join("{:12s}{:4d}".format(k, v) for k, v in sorted(sources.items(), key=lambda item: item[1], reverse=True)))
    print("Other       {:4d}".format(df.shape[0]-sum(sources.values())))
    print("Unknown DOI {:4d}".format(df.loc[~df.index.isnull()].shape[0]-sum(sources.values())))
    print("No DOI      {:4d}".format(df.loc[df.index.isnull()].shape[0]))

print("\nPublisher Prominence")
if include_citation:
    temp = sorted(publishers.items(), key=lambda item: item[1]['count'], reverse=True)
    print("\n".join("{:12s}{:4d} (total citations: {:4d}, cite-score: {:4.1f})"
                    .format(k, v['count'], v['citations'], v['citations']/v['count']) for k, v in sorted(publishers.items(), key=lambda item: item[1]['count'], reverse=True)))
    temp_count = df.shape[0] - sum([v['count'] for v in publishers.values()])
    temp_citations = sum(df['citations']) - sum([v['citations'] for v in publishers.values()])
    print("Unknown Pub.{:4d} (total citations: {:4d}, cite-score: {:4.1f})".format(temp_count, temp_citations, temp_citations/temp_count))
else:
    print("\n".join("{:12s}{:4d}".format(k, v) for k, v in sorted(publishers.items(), key=lambda item: item[1], reverse=True)))
    print("Unknown Pub.{:4d}".format(df.shape[0]-sum(publishers.values())))


### UNSUPERVISED TEXT SEARCH ###

print("\n\nUnsupervised keyword analysis\n(counting method is number of {}, token lenght is {}-gram)".format(count_method, token_length))
for field in fields:
    keywords,word_count = unsupervised_count(df,field,token_length)
    df_field = pd.Series(keywords, index=keywords.keys())
    print("\nTop {} keywords in {}s:".format(n_keywords,field))
    print("({}k tokens total, {}k unique tokens, excluding {} stop words)\n"
          .format(int(word_count/1000),int(len(keywords)/1000),len(stop_words)))
    temp = df_field.nlargest(n_keywords)
    print(df_field.nlargest(n_keywords).to_string())

yr_grp = df.groupby(['year'])
if n_years_keywords > 0:
    for field in fields:
        year_top = {}
        for yr, grp in yr_grp:
            keywords,_ = unsupervised_count(grp,field,token_length)
            df_year = pd.Series(keywords, index=keywords.keys())
            year_top[yr] = df_year.nlargest(n_years_keywords).to_string()

        print("\nTop {} keywords per year in {}s:\n".format(n_years_keywords, field))
        for k,v in year_top.items():
            print(k)
            v = "\t" + v.replace("\n","\n\t")
            print(v)

print("\n\n")


### AUTHOR INFORMATION ###

authors, total_authors, authors_per_paper, _, _, _ = author_count(df)
df_list_authors = pd.Series(authors, index=authors.keys())

print("Top {} authors by number of publications:".format(n_authors))
print("(total unique authors: {}, avg authors per paper: {:.1f})\n"
      .format(total_authors,mean(authors_per_paper)))
print(df_list_authors.nlargest(n_authors).to_string())

if include_citation:
    _, _, _, authors_citations, authors_citescore, authors_start_year = author_count(df)
    # temp = [c/a for c,a in zip(authors.values(), authors_citations.values())]
    df_list_citations = pd.Series(authors_citations, index=authors_citations.keys())
    df_list_avg_citations = pd.Series(authors_citescore, index=authors_citescore.keys())
    # df_list_avg_citations = pd.Series(authors_citations, index=authors_citations.keys())
    print("\n\nTop {} authors by citations:".format(n_authors))
    print("(avg citations per author: {:.1f})\n".format(mean(authors_citations.values())))
    for i in range(n_authors):
        a = list(authors_citations.keys())[i]
        print("{:20s}{:4d} (number of papers: {:2d})".format(a, authors_citations[a], authors[a]))
    # print(df_list_citations.nlargest(n_authors).to_string())
    print("\n\nTop {} authors by citations per year (cite-score):".format(n_authors))
    print("(avg cite-score per author: {:.2f})\n".format(mean(authors_citescore.values())))
    for i in range(n_authors):
        a = list(authors_citescore.keys())[i]
        print("{:20s}{:4.1f} (first year of publication: {})".format(a, authors_citescore[a], authors_start_year[a]))
    # print(df_list_avg_citations.nlargest(n_authors).to_string())


print("\nTop author per year and corresponding publications:")
for yr, grp in df.groupby('year'):
    authors, _, _, _, _, _ = author_count(grp)
    top_author = max(authors, key=authors.get)
    print(str(yr) + "\t" + top_author + "\t" + str(authors[top_author]))

if n_top_authors > 0:
    print("\n{} most active authors in detail:".format(n_top_authors))
    for author in df_list_authors.nlargest(n_top_authors).index:
        filt = [False]*len(df)
        for i,paper_authors in enumerate(df['authors']):
            for single_author in paper_authors.split(" "):
                # names = single_author.split("+")
                # if author == names[0][0] + ". " + names[-1]:
                if author == modify_name(single_author):
                    filt[i] = True

        df_author = df.loc[filt]
        print("\n{} with a total of {} papers between {} and {}"
              .format(author,df_author.shape[0],df_author['year'].min(),
                      df_author['year'].max()))

        top_score = 1
        for yr, grp in df_author.groupby('year'):
            if grp.shape[0] >= top_score:
                top_year = yr
                top_score = grp.shape[0]
            print(str(yr) + "\t" + str(grp.shape[0]))
        print("(most active year was {})\n".format(top_year))

        coauthors, _, coauthors_per_paper, _, _, _ = author_count(df_author)
        df_list_coauthors = pd.Series(coauthors, index=coauthors.keys())
        df_list_coauthors.drop(author,inplace=True)
        print("Favorite {} co-authors (average of {:.1f} authors per paper):".format(n_coauthors,mean(coauthors_per_paper)))
        print(df_list_coauthors.nlargest(n_coauthors).to_string())

        print("\nTopics of author (top {} keywords):".format(n_authors_keywords))
        keywords,_ = unsupervised_count(df_author,'title',token_length)
        df_title = pd.Series(keywords, index=keywords.keys())
        print(df_title.nlargest(n_authors_keywords).to_string())


# normalize number of keywords to length of title and keyword
# combine values for titles and abstracts
# search keywords via total occurunce or just count elements where they occur as one, independent of repition