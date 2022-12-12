# 3.3.2021 | Lukas Preis | v1.0
# Version News: basic functionality

import csv
import pandas as pd
import os
import numpy as np

name_filter = "keywords_TechTransfer.csv"
name_database = "RG_Library_TechTransfer.csv"
analysis_title = 'TechTransfer'

dir = os.path.dirname(os.path.dirname(__file__))
file_database = os.path.join(dir, 'input', name_database)
file_filter = os.path.join(dir, 'input', name_filter)

with open(file_filter,'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_file)
    keywords = []
    for row in csv_file:
        keywords.append(row.replace("\n",""))

df = pd.read_csv(file_database, delimiter=';', dtype=object, index_col='doi')

dtm = []
count = [0]*len(keywords)
sum = [0]*len(keywords)

for i,doi in enumerate(df.index):
    if doi is np.nan:
        dtm.append([df.iloc[i]['title'][:30].replace(',', '')])
    else:
        dtm.append([doi])
    for j,k in enumerate(keywords):
        # abstract = df.loc[doi]['abstract']
        if type(df.iloc[i]['title']) == str:
            hits = df.iloc[i]['title'].lower().count(k)
        if type(df.iloc[i]['abstract']) == str:
            hits += df.iloc[i]['abstract'].lower().count(k)
        dtm[i].append(hits)
        sum[j] += hits
        if hits > 0:
            count[j] += 1

with open(os.path.join(dir, 'input', 'DTM_{}.csv'.format(analysis_title)), 'w', newline='', encoding='utf-8') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',')
    csv_writer.writerow(['dtm_'+analysis_title] + keywords)
    csv_writer.writerows(dtm)
    csv_writer.writerow(['sum'] + sum)
    csv_writer.writerow(['count'] + count)
