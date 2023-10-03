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
    for rank in range(0, nb_lines):
        current_line = results.loc[rank]
        query_id = qid
        Q0 = "Q0"
        document_no = current_line['docno']
        retrieval_score = current_line['score']
        lines += f"{query_id} {Q0} {int(document_no)} {rank} {retrieval_score} {run_id}\n"

    with open(path_file_output, "a") as results_file:
        results_file.write(lines)
    
    
def rank_query(query : str, model_rank : pt.BatchRetrieve, model_rerank : pt.BatchRetrieve, 
               nb_reranked : int) -> pd.core.frame.DataFrame:
    # Ranking & Re-ranking the docs relatively to the query
    if model_rerank == None:
        pipeline = model_rank
    else:
        pipeline = (model_rank % nb_reranked) >> model_rerank
    final_rank = pipeline.search(query)
    return final_rank

def rank_queries(path_queries : str, path_file_output, model_rank : pt.BatchRetrieve, 
                 model_rerank : pt.BatchRetrieve, nb_reranked : str) :
    # Empty the file
    with open(path_file_output, 'w') as tsvfile:
        tsvfile.truncate()
    # Open the queries's file
    with open(path_queries, 'r', newline='', encoding='utf-8') as tsvfile:
        print(type(tsvfile))
        tsvreader = csv.reader(tsvfile, delimiter='\t')
        next(tsvreader)

        # Iteration on queries
        for l in enumerate(tsvreader):
            # Recover the data
            line = re.sub(r',', ' ', l[1][0]).split()
            qid = line[0]
            query = ' '.join(line[1:len((line))-2])

            # preprocess the query
            query_pp = query_preprocessing(query, STOPWORDS_DEL = False)
            print(query_pp)

            # Apply a documents's ranking
            ranking = rank_query(query_pp, model_rank, model_rerank, nb_reranked)
            set_results_with_trec_format(ranking, path_file_output, qid, nb_lines = 1000, run_id = "monoT5")

# model_rank = pt.BatchRetrieve("./data/inverted_index", wmodel="BM25", 
#                               controls={"wmodel": "default", "ranker.string": "default", "results" : 1000})
# model_rerank = pt.BatchRetrieve("./data/inverted_index", wmodel="PL2", 
#                                 controls={"wmodel": "default", "ranker.string": "default", "results" : 1000})

# rank_queries("data/reduced_qrels/first_query_queries.csv", "data/results.txt", model_rank , 
#                  model_rerank, nb_reranked=1000)