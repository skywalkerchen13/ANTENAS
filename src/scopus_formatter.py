import pandas as pd
import csv

title = 'UAM (906)'

file = "C:\\Users\lukas.preis\Downloads\{}.csv".format(title)
file_out = "C:\\Users\lukas.preis\Downloads\{}.csv".format('_'.join(title.split(' ')[:-1]))

df = pd.read_csv(file)
with open(file_out, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=';')
    csv_writer.writerow(['doi','year','authors','title','abstract','citations'])

for i in range(df.shape[0]):

    doi = df['DOI'].iloc[i]
    if str(doi) == 'nan':
        doi = 'N/A'
    year = df['Year'].iloc[i]

    authors_string = df['Authors'].iloc[i]
    if authors_string == '[No author name available]':
        authors_formatted = 'N/A'
    else:
        authors_names = [a.split(' ') for a in authors_string.split(', ')]
        authors = [str(a[-1].replace('-', '').replace('.', '. ') + " " + a[0].title()).replace("  "," ").replace(" ", "+") for a in authors_names]
        authors_formatted = ' '.join(authors)

    title = df['Title'].iloc[i].title()
    abstract = df['Abstract'].iloc[i].replace(';', ',').replace('"', '').replace('©', '').replace('  ', ' ')

    if abstract == '[No abstract available]':
        abstract = 'N/A'

    try:
        citations = int(df['Cited by'].iloc[i])
    except:
        citations = 'N/A'

    authors_formatted = authors_formatted.encode("ascii", "ignore").decode()
    title = title.encode("ascii", "ignore").decode()
    abstract = abstract.encode("ascii", "ignore").decode()
    with open(file_out, 'a', newline='\n') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=';')
        csv_writer.writerow([doi, year, authors_formatted, title, abstract, citations])
