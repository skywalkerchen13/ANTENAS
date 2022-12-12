# Author: Lukas Preis
# Creation Date: 24.9.2020
# Version: 2.2

import pandas as pd
import csv
# import os
from os.path import dirname, join

name_database = "RG_Library_Com&Inno.csv"
name_filter = "keywords_Com&Inno.csv" # can be .csv or .txt file

dir = dirname(dirname(__file__))
file_database = join(dir, 'input', name_database)
file_filter = join(dir, 'input', name_filter)

# format of database: 5 columns, seperated by semicolon: doi, year, authors, title, abstract
df = pd.read_csv(file_database, delimiter=';', dtype=object, index_col='doi')

# format of filter: first row name of database column to filter, then one filter per row
# keywords all in small letters, authors in the format: F. LLLL (F: First name, L: Last Name)
if 'csv' in name_filter:
    with open(file_filter,'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        filter_field = next(csv_file).replace("\n","")
        filter_keywords = []
        for row in csv_file:
            filter_keywords.append(row.replace("\n",""))

elif 'txt' in name_filter:
    filter_field = 'title'
    keyword_string = open(file_filter, 'r').read()
    if '(' and ')' in keyword_string:
        filter_keywords = keyword_string[1:-1].split(') | (')
        for i,line in enumerate(filter_keywords):
            if '[' and ']' in line:
                filter_keywords[i] = line[1:-1].split('] & [')
                for j,row in enumerate(filter_keywords[i]):
                    filter_keywords[i][j] = row.split('|')
            else:
                filter_keywords[i] = line.split('&')
    else:
        filter_keywords = keyword_string.split('|')

### INPUT FILTER ###

# filter_field = 'authors' # field to be filtered
# filter_keywords = ['R. Rothfeld', 'J. Sutton'] # list of keywords to be searched for
query = '#AIR#'
# results_field = 'year'


### MANUAL FILTERING ###

## find paper by DOI
# doi = "10.3141/1723-02"
# print(df.loc[doi])

## find paper by title
# title = "Leading Through Intervals versus Leading Pedestrian Intervals: More Protection with Less Capacity Impact"
# print(df[df['title'].apply(str.lower) == title.lower()]['title'])


### FILTER BY KEYWORDS ###

# filt = df.isnull().any(axis=1)
filt = df[filter_field].isnull()
df.drop(index=df[filt].index, inplace=True) # drop all rows that are not complete

if filter_field == "title" or filter_field == "abstract":

    if type(filter_keywords[0]) == str:
        search_dic = {} # initializing dictionary to replace keywords by query
        for keyword in filter_keywords:
            search_dic[keyword] = query
        df_query = df[filter_field].apply(str.lower).replace(search_dic, regex=True) # replace keywords by query
        filt = df_query.str.contains(query) # filter dataframe by query

    elif type(filter_keywords[0]) == list:
        filt = []
        for search_text in df[filter_field]:
            hit = False
            for element in filter_keywords:
                if type(element) == list:
                    hit = True
                    for line in element:
                        if type(line) == list:
                            hit = False
                            for row in line:
                                if row in search_text:
                                    hit = True
                                    break
                            if not hit:
                                break
                        else:
                            if not line in search_text:
                                hit = False
                                break
                    if hit:
                        break
                else:
                    if element in search_text:
                        hit = True
                        break
            filt.append(hit)

elif filter_field == 'authors':
    filt = [False]*df.shape[0]
    for i,paper_authors in enumerate(df[filter_field]):
        for single_author in paper_authors.split(" "):
            names = single_author.split("+")
            author = names[0][0] + ". " + names[-1] # format of author: F. LLLL (F: First name, L: Last name)
            if author in filter_keywords:
                filt[i] = True

elif filter_field == 'year':
    filt = [True if row in filter_keywords else False for row in df[filter_field]]
    # filt = [False]*df.shape[0]
    # for paper_years in filter_keywords:


### EXPORT FILTERED DATABASE ###

df_filtered = df.loc[filt]
print(df_filtered)
df_filtered.reset_index(inplace=True)
df_filtered.fillna('N/A', inplace=True)
df_filtered.to_csv(file_database[:-4] + "_filtered_by_" + filter_field + ".csv", sep=';', index=False)