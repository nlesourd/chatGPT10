import csv
import pyterrier as pt
import re
import pandas as pd

if not pt.started():
    pt.init()

def set_results_with_trec_format(results: pd.core.frame.DataFrame, path_file_output: str, qid:int, 
                                 kaggle : bool, run_id="monoT5"):
    """ Add 1000 lines to results with the trec format

    Args:
        results: Dataframe results of the retrieval ranking of each document with qid, docid, docno, rank, score and query
        path_file_output: path of the output file with TREC format
        run_id, optional, id of retrieval process  
    """
    results.sort_values(by=["score"], ascending=False, inplace=True)
    rank = 0
    lines = ""
    max_score = results['score'].max()
    min_score = results['score'].min()
    for document_no, retrieval_score in zip(results['docno'], results['score']):
        Q0 = "Q0"
        retrieval_score = (retrieval_score - min_score) / (max_score - min_score)
        if kaggle:
            lines += f"{qid},{int(document_no)}\n"
        else:
            lines += f"{qid} {Q0} {int(document_no)} {rank} {retrieval_score} {run_id}\n"
    with open(path_file_output, "a") as results_file:
        results_file.write(lines)

def rank_queries(path_queries: str, path_file_output, method, kaggle = False):
    # Empty the file
    with open(path_file_output, 'w') as tsvfile:
        tsvfile.truncate()
    # Open the queries's file
    with open(path_queries, 'r', newline='', encoding='utf-8') as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter='\t')
        next(tsvreader)

        # Iteration on queries
        for l in enumerate(tsvreader):
            # Recover the data
            line = re.sub(r',', ' ', l[1][0]).split()
            qid = line[0]
            query = ' '.join(line[1:len((line))-2])

            # Apply a documents's ranking
            ranking = method.rank_query(query)
            set_results_with_trec_format(ranking, path_file_output, qid, kaggle, run_id="monoT5")