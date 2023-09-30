from sqlitedict import SqliteDict
import csv 

def study_text(path_collection, path_queries):
    # Limit of size 
    csv.field_size_limit(4096 * 4096)
    doc_under_ex = []
    # Open the TSV file
    with open(path_queries, 'r', newline='', encoding='utf-8') as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter='\t')
        for line in tsvreader:
            print(line[0].split()[2])
            doc_under_ex.append(line[0].split()[2])

    # Open the TSV file
    with open(path_collection, 'r', newline='', encoding='utf-8') as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter='\t')
        # Iteration on lines
        short_texts = 0
        for line in tsvreader:
            docno = line[0]
            text = line[1]
            # First interesting thing : short texts -> 3911
            if len(text.split())<10:
                short_texts += 1
                print(short_texts)
            
            # See the texts under examination
            if docno in doc_under_ex:
                print(docno)
                with open("data/reduced_qrels/under_exam.txt", "a") as results:
                    results.write(text + "\n")

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

# textStoring('data/collection.tsv','data/collection.sqlite')
study_text('data/collection.tsv', 'data/reduced_qrels/first_query_qrels.txt')