# Author: Lukas Preis
# Creation Date: 3.3.2021
# Version: 1.4
# ----------------------- #
# Input: (Option 1) two databases, (Option 2) a folder with N databses, more options ...
# Process: compare databases and create files according to two-circle ven-diagram
# Output: (Option 1) four files without duplicates, (Option 2) one file without duplicates
# ----------------------- #

from os.path import isdir, isfile, dirname, join, basename, splitext
from os import mkdir, listdir
import pandas as pd
import xml.etree.ElementTree as ET

### INPUT ###

print("DATABASE COMBINER v1.4")
print("(c) 2021 by Lukas Preis at Bauhaus Luftfahrt")
print("\n--------------------------------------------------------------------------\n")
print("full path database A OR folder to merge multiple databases: ")
path = input().replace('"','')

if not isdir(path):
    file_A = path
    print("full path database B OR full path to list of dois OR full path to sources: ")
    file_B = input().replace('"','')
    check_duplicates = True
else:
    check_duplicates = False

print("\n--------------------------------------------------------------------------\n")
print("combining databases...")


### MAIN SCRIPT ###

if isfile(path): # combining two files

    df_A = pd.read_csv(file_A, delimiter=';', dtype=object, index_col='doi')

    if 'xml' in file_B: # compare database with source dictionary to identify missing doi patterns
        sources_raw = ET.parse(file_B).getroot()
        dois = [d.get('code') for s in sources_raw if s.tag == 'source' for d in s.find('dois')]
        df_nodoi = df_A[df_A.index.isnull()]
        df_nodoi.index = df_nodoi.index.fillna('N/A')
        df_doi = df_A[~df_A.index.isnull()]
        filt = []
        for s in df_doi.index.tolist():
            hit = False
            for d in dois:
                if d in s:
                    hit = True
                    break
            filt.append(hit)

        path = dirname(file_A)
        df_nodoi.fillna('N/A').to_csv(join(path, 'DOIs_nan.csv'), sep=';')
        df_doi[filt].fillna('N/A').to_csv(join(path, 'DOIs_known.csv'), sep=';')
        df_doi[[not f for f in filt]].fillna('N/A').sort_index().to_csv(join(path, 'DOIs_unknown.csv'), sep=';')

        print('\nDatabase split according to sources! Please find three databases at ' + path)

    elif 'doi' in file_B.lower(): # compare database with list of dois and identify missing meta data
        df_doi = pd.read_csv(file_B)
        missing_dois = []
        for doi in df_doi['dois']:
            if doi not in df_A.index:
                missing_dois.append(doi)

        path = dirname(file_B)
        df_doi_missing = pd.DataFrame(missing_dois, columns=['dois'])
        df_doi_missing.to_csv(join(path, 'missing_dois2.csv'), index=False)

        print('\nDatabase and list of dois compared! Please find missing dois at ' + path)

    else: # compare and merge two databases
        path = join(dirname(file_B), 'combined databases')
        try:
            mkdir(path)
        except:
            pass

        df_B = pd.read_csv(file_B, delimiter=';', dtype=object, index_col='doi')

        df_A = df_A.loc[df_A.index.dropna()]
        df_B = df_B.loc[df_B.index.dropna()]

        file_AX = join(path, splitext(basename(file_A))[0] + '_exclusive.csv')
        file_BX = join(path, splitext(basename(file_B))[0] + '_exclusive.csv')
        file_AB = join(path, 'Shared_Data.csv')
        file_sum = join(path, 'Combined_Data.csv')

        df_sum = pd.concat([df_A, df_B], sort=False)
        df_sum = df_sum[~df_sum.index.duplicated(keep='first')]

        if 'citations' in df_sum.columns.values:
            columns = ['doi', 'year', 'authors', 'title', 'abstract', 'citations']
        else:
            columns = ['doi', 'year', 'authors', 'title', 'abstract']

        df_AX = pd.DataFrame(columns=columns).set_index('doi')
        df_BX = pd.DataFrame(columns=columns).set_index('doi')
        df_AB = pd.DataFrame(columns=columns).set_index('doi')

        for doi in df_sum.index:
            temp = df_sum.loc[doi]
            if doi in df_A.index and doi in df_B.index:
                df_AB = df_AB.append(df_sum.loc[doi])
            elif doi in df_A.index:
                df_AX = df_AX.append(df_sum.loc[doi])
            elif doi in df_B.index:
                df_BX = df_BX.append(df_sum.loc[doi])
            else:
                print(doi)

        df_sum.fillna('N/A').to_csv(file_sum, sep=';')
        df_AB.fillna('N/A').to_csv(file_AB, sep=';')
        df_AX.fillna('N/A').to_csv(file_AX, sep=';')
        df_BX.fillna('N/A').to_csv(file_BX, sep=';')
        print('\nDatabases combined! Please find them at ' + path)


else: # combining files of entire folder

    csv_files = [join(path, f) for f in listdir(path) if isfile(join(path, f)) and f[-3:] == 'csv']

    mkdir(join(path, 'combined data'))

    # https://stackoverflow.com/questions/12468179/unicodedecodeerror-utf8-codec-cant-decode-byte-0x9c
    df_sum = pd.read_csv(csv_files[0], delimiter=';', dtype=object, engine='python')

    for file in csv_files[1:]:
        try:
            df_new = pd.read_csv(file, delimiter=';', dtype=object, engine='python')
        except:
            print("Issues with database " + file)
            print("press enter to terminate script...")
            input()
            exit()

        df_sum = pd.concat([df_sum, df_new])

    df_sum.fillna('N/A').to_csv(join(path, "combined data", 'Combined_Data.csv'), sep=';', index=False)
    df_sum.drop_duplicates(inplace=True)
    df_sum.fillna('N/A').to_csv(join(path, "combined data", 'Combined_Data_no_duplicates.csv'), sep=';', index=False)
    filt = df_sum.notnull()['doi']
    df_sum.loc[filt].fillna('N/A').to_csv(join(path, "combined data", 'Combined_Data_doi_present.csv'), sep=';', index=False)
    filt = df_sum.isnull().any(axis=1)
    df_sum.loc[~filt].fillna('N/A').to_csv(join(path, "combined data", 'Combined_Data_no_incomplete.csv'), sep=';', index=False)

    print('\nDatabases combined! Please find them at ' + join(path, "combined data"))

input()