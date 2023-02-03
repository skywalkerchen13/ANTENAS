# Author: Lukas Preis
# Creation Date: 3.3.2021
# Version: 1.1
# ----------------------- #
# Input: filtered database (.csv), list of keywords (.csv)
# Process: publications' title and abstract in database are screened for keywords
# Output: document term matrix indicating the occurence of each keyword in each publication
# ----------------------- #


import csv
import pandas as pd
from os.path import dirname, join
import numpy as np

### INPUT ###

name_filter = "keywords_TechTransfer.csv"
name_database = "RG_Library_TechTransfer.csv"
analysis_title = 'TechTransfer'


### READ DATABASE AND KEYWORDS ###

dir = dirname(dirname(__file__))
file_database = join(dir, 'input', name_database)
file_filter = join(dir, 'input', name_filter) # only possible with basic filter (csv-file) and for fields title/abstract

df = pd.read_csv(file_database, delimiter=';', dtype=object, index_col='doi') # database
print("\ndatabase loaded from " + file_database)

with open(file_filter,'r') as csv_file: # filter
    csv_reader = csv.reader(csv_file)
    next(csv_file)
    keywords = []
    for row in csv_file:
        keywords.append(row.replace("\n",""))
print("filter loaded from " + file_filter)



### MAIN SCRIPT ###

print("\nanalyzing filter...")
dtm = []
count = [0]*len(keywords)
sum = [0]*len(keywords)

for i,doi in enumerate(df.index):
    if doi is np.nan:
        dtm.append([df.iloc[i]['title'][:30].replace(',', '')])
    else:
        dtm.append([doi])
    for j,k in enumerate(keywords):
        if type(df.iloc[i]['title']) == str:
            hits = df.iloc[i]['title'].lower().count(k)
        if type(df.iloc[i]['abstract']) == str:
            hits += df.iloc[i]['abstract'].lower().count(k)
        dtm[i].append(hits)
        sum[j] += hits
        if hits > 0:
            count[j] += 1


### EXPORT FILTERED DATABASE ###

file_export = join(dir, 'input', 'DTM_{}.csv'.format(analysis_title))
print("done analyzing filter! Export to " + file_export)
with open(file_export, 'w', newline='', encoding='utf-8') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',')
    csv_writer.writerow(['dtm_'+analysis_title] + keywords)
    csv_writer.writerows(dtm)
    csv_writer.writerow(['sum'] + sum)
    csv_writer.writerow(['count'] + count)
