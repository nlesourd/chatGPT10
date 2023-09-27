from sqlitedict import SqliteDict
import csv 

def textStoring(path_collection, path_sqldict):
    dictText = SqliteDict(path_sqldict)
    # Limit of size 
    csv.field_size_limit(4096 * 4096)
    # Open the TSV file
    with open(path_collection, 'r', newline='', encoding='utf-8') as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter='\t')
        
        # Iteration on lines
        for line in tsvreader:
            docno = line[0]
            text = line[1]
            dictText[docno] = text

    dictText.update()
    dictText.commit()
    print("Index updated.")

textStoring('data/collection.tsv','data/collection.sqlite')