from typing import List, Dict, TextIO, Union
from preprocessing import query_preprocessing
import csv
from collections import defaultdict
import pyterrier as pt
import re
import numpy as np
import pandas as pd

if not pt.started():
    pt.init()

def set_results_with_trec_format(results: pd.core.frame.DataFrame, path_file_output: str, qid:int, 
                                 nb_lines = 1000, run_id = "monoT5"):
    """ Add 1000 lines to results with the trec format

    Args:
        results: Dataframe results of the retrieval ranking of each document with qid, docid, docno, rank, score and query
        path_file_output: path of the output file with TREC format
        nb_lines: optional, nuumber of documents ranked
        run_id, optional, id of retrieval process  
    """
    results.sort_values(by=["score"], ascending=False, inplace=True)
    rank = 0
    lines = ""
    for document_no, retrieval_score in zip(results['docno'], results['score']):
        query_id = qid
        Q0 = "Q0"
        lines += f"{query_id} {Q0} {document_no} {rank} {retrieval_score} {run_id}\n"
        rank += 1

    with open(path_file_output, "w") as results_file:
        results_file.write(lines)
    
    
def training_queries(path_queries_train : str, path_queries_rels : str,
                      inverted_index_path : str) -> np.array((5,5)) :
    """ Try the model and evaluate the performance by returning the matrix of confusion.

    Args:
        path_queries_train : path of the file with the id of the queries and the queries
        path_queries_rels : path of the file with the id of the queries and actual scores
        inverted_index_path : path of the inverted index 

    Returns:
        Array of size (5 * 5) representing the computation of the confusion matrix 
        for the data under evaluation.
    """
    # Model for ranking
    bm25 = pt.BatchRetrieve(inverted_index_path, num_results = 1000, wmodel="BM25")
    # Model for re-ranking
    pl2 = pt.BatchRetrieve(inverted_index_path, wmodel="PL2", 
                           controls={"wmodel": "default", "ranker.string": "default"})
    nb_reranked = 200
        
    # Open the TSV file
    with open(path_queries_train, 'r', newline='', encoding='utf-8') as tsvfile:
        print(type(tsvfile))
        tsvreader = csv.reader(tsvfile, delimiter='\t')
        next(tsvreader)

        # Init of the predictions and actuals labels
        actuals = []
        predictions = []
        query_before = ""
        # Iteration on lines
        for l in enumerate(tsvreader):
            # Recover the data
            line = l[1][0]
            line = re.sub(r',', ' ', line)
            line = line.split()
            qid = line[0]

            # with open("data/reduced_qrels/first_query_queries.csv", "a") as file:
            #     if int(qid[-1])==1:
            #         print(qid)
            #         print("écriture")
            #         file.write(str(l) + "\n")

            query = ' '.join(line[1:len((line))-2])
            query_pp = ' '.join(query_preprocessing(query, STOPWORDS_DEL = True))

            # Ranking the docs relatively to the query
            bm25_rank = bm25.search(query_pp)

            # if too few results, search with the entire query
            if bm25_rank.shape[0] < 1000:
                query_pp = ' '.join(query_preprocessing(query, STOPWORDS_DEL = False))
                print(query)
                bm25_rank = bm25.search(query_pp)
                # if too few results, search with the query before combined
                if bm25_rank.shape[0] < 1000:
                    query_pp = ' '.join(query_preprocessing(query_before + " " + query, STOPWORDS_DEL = True))
                    print(query_pp)
                    bm25_rank = bm25.search(query_pp)

            # memorize the last query        
            query_before = query_pp

            # Re-ranking
            pipeline = (bm25 % nb_reranked) >> pl2
            pl2_re_ranked = pipeline.search(query_pp)

            # Fill the results.txt file (TREC)
            with open("data/results.txt", "a") as results:
                # Create 1000 new lines in the runfile
                # Put None instead of pl2_re_ranked if don't want a re-rank
                add_lines_trec_format(bm25_rank, pl2_re_ranked, results, qid)

            # Fill actuals and predictions
            for doc in queries_scores[qid]:
                predictions.append(grade_out_of_4_rank(doc, bm25_rank, pl2_re_ranked)) 
                actuals.append(queries_scores[qid][doc])

    # return the confusion matrix
    return confusion_matrix(actuals, predictions)

"""
# Test functions
# path_queries_rels = "data/qrels_train.txt"
inverted_index_path = "./data/inverted_index"
# path_queries = "data/queries_train.csv"

# With only the first queries
path_queries = "data/reduced_qrels/first_query_queries.csv"
path_queries_rels = "data/reduced_qrels/first_query_qrels.txt"
training_queries(path_queries, path_queries_rels, inverted_index_path)
"""