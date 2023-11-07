from typing import List, Dict, TextIO, Union
from preprocessing import query_preprocessing
import csv
from collections import defaultdict
import pyterrier as pt
import re
import numpy as np
import pandas as pd
from pyterrier_t5 import MonoT5ReRanker
from elasticsearch import Elasticsearch

# Init
es = Elasticsearch()
if not pt.started():
    pt.init()

def set_results_with_trec_format(results: pd.core.frame.DataFrame, path_file_output: str, qid:int, 
                                 nb_lines : int, kaggle : bool, run_id = "monoT5"):
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

def rank_query(query : str, model_rank : pt.BatchRetrieve, model_rerank : pt.BatchRetrieve, 
               nb_reranked : int, query_before : str) -> pd.core.frame.DataFrame:
    # Ranking the docs relatively to the query
    results_rank = model_rank.search(query)
    if results_rank.shape[0]<1000:
        print(query)
        results_rank = model_rank.search(query_before + " " + query)
    # Re-Ranking the docs relatively to the query
    if model_rerank == None:
        pipeline = model_rank
        final_rank = pipeline.search(query)
    # Mono-T5 re-ranking
    elif isinstance(model_rerank, MonoT5ReRanker):
        # Add the text to the results
        l_texts = []
        for i in range(nb_reranked):
            doc_no = results_rank.loc[i]['docno']
            l_texts.append(es.get(index="index_texts", id=doc_no)['_source']['content'])
        results_rank['text'] = l_texts
        final_rank = model_rerank.transform(results_rank)
        # Sort
        final_rank.sort_values(by='score', ascending=False, inplace=True)
        print(final_rank)
    # Other re-ranking
    else:
        final_rank = model_rerank.transform(results_rank)
    return final_rank

def rank_queries(path_queries : str, path_file_output, model_rank : pt.BatchRetrieve, 
                 model_rerank : pt.BatchRetrieve, nb_reranked : str, nb_lines = 1000, kaggle = False) :
    query_before = ""
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
            ranking = rank_query(query_pp, model_rank, model_rerank, nb_reranked, query_before)
            query_pp = query_before + " " + query_pp
            set_results_with_trec_format(ranking, path_file_output, qid, nb_lines, kaggle, run_id = "monoT5")

            # keep track of the last query
            query_before = query_pp

# model_rank = pt.BatchRetrieve("./data/inverted_index", wmodel="BM25", 
#                               controls={"wmodel": "default", "ranker.string": "default", "results" : 1000})
# model_rerank = pt.BatchRetrieve("./data/inverted_index", wmodel="PL2", 
#                                 controls={"wmodel": "default", "ranker.string": "default", "results" : 1000})

# rank_queries("data/reduced_qrels/first_query_queries.csv", "data/results.txt", model_rank , 
#                  model_rerank, nb_reranked=1000)