from typing import List, Dict
from preprocessing import query_preprocessing
import csv
from collections import defaultdict
import pyterrier as pt
import re
import numpy as np
import pandas as pd

if not pt.started():
    pt.init()

def scores_recovery(path_queries_rels:str) -> Dict:
    """Recover the defined scores between 0 and 4 for every query.

    Args:
        path_queries_rels: path of the files with the queries and their scores

    Returns:
        Dictionnary of dictionnaries of score. First key is the query id. 
        Second key is the docid. It allows to access to score for a given
        query id and docid.
    """
    # Take the score of documents for queries
    queries_scores = defaultdict(dict)
    # Open the TSV file
    with open(path_queries_rels, 'r', newline='', encoding='utf-8') as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter='\t')
        # Iteration on lines
        for l in enumerate(tsvreader):
            line = l[1][0].split()
            qid = line[0]
            doc_id = line[2]
            score = int(line[3])
            queries_scores[qid][doc_id] = score
    return queries_scores

def grade_out_of_4_rank(doc: str, bm25_rank: pd.core.frame.DataFrame) -> int:
    """Recover the defined scores between 0 and 4 for every query.

    Args:
        path_queries_rels: path of the files with the queries and their scores

    Returns:
        return the predicted score between 0 to 4
    """
    score = 0
    # # if re-ranked
    # if len(np.where(pl2_reranked['docid'] == int(doc))[0]) > 0:
    #     rank = np.where(pl2_reranked['docid'] == int(doc))[0][0]
    #     if rank < 30:
    #         score = 4
    #     elif rank < 150:
    #         score = 3
    #     else:
    #         score = 2
    # if ranked
    # elif len(np.where(bm25_rank['docid'] == int(doc))[0]) > 0:
    #     rank = np.where(bm25_rank['docid'] == int(doc))[0][0]
    #     if rank < 500 :
    #         score = 2
    #     elif rank<1000:
    #         score = 1
    
    if len(np.where(bm25_rank['docid'] == int(doc))[0]) > 0:
        rank = np.where(bm25_rank['docid'] == int(doc))[0][0]
        if rank < 30:
            score = 4
        elif rank < 150:
            score = 3
        elif rank < 600:
            score = 2
        else:
            score = 1
    return score

def confusion_matrix(actuals:List[int], predictions:List[int]) -> np.array((5,5)):
    """Computes confusion matrix from lists of actual or predicted labels.

    Args:
        actuals: List of integers (0, 1, 2, 3 or 4) representing the actual classes of
            some instances.
        predictions: List of integers (0, 1, 2, 3 or 4) representing the predicted classes
            of the corresponding instances.

    Returns:
        Array of size (5 * 5) representing the confusion matrix.
    """
    sum_errors = 0
    errors = np.zeros((5, 5))
    for i in range(len(actuals)):
        sum_errors += abs(predictions[i] - actuals[i])
        errors[predictions[i]][actuals[i]] += 1
    print("Error rate : " + str(sum_errors/(len(actuals)*4) * 100) + "%")
    print("Errors repartition : (lines : predictions 0 to 4, columns : actuals 0 to 4")
    print(errors)
    return errors

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
    # Recover the querie's scores
    queries_scores = scores_recovery(path_queries_rels)

    # Open the TSV file
    with open(path_queries_train, 'r', newline='', encoding='utf-8') as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter='\t')
        next(tsvreader)
        # Model for ranking
        bm25 = pt.BatchRetrieve(inverted_index_path, num_results = 2000, wmodel="BM25")
        # Model for re-ranking
        pl2 = pt.BatchRetrieve(inverted_index_path, wmodel="PL2")

        # Init of the predictions and actuals labels
        actuals = []
        predictions = []
        # Iteration on lines
        for l in enumerate(tsvreader):
            # Recover the data
            line = l[1][0]
            line = re.sub(r',', ' ', line)
            line = line.split()
            qid = line[0]
            query = ','.join(line[1:len((line))-2])
            query_pp = ' '.join(query_preprocessing(query))

            # Ranking the docs relatively to the query
            bm25_rank = bm25.search(query_pp)

            # Re-ranking
            # pl2_re_ranked = (bm25 % 200) >> pl2.search(query_pp)

            # Fill actuals and predictions
            for doc in queries_scores[qid]:
                predictions.append(grade_out_of_4_rank(doc, bm25_rank)) # ,pl2_re_ranked)) 
                actuals.append(queries_scores[qid][doc])

    # return the confusion matrix
    return confusion_matrix(actuals, predictions)

# Test functions
path_queries_rels = "data/qrels_train.txt"
inverted_index_path = "./data/inverted_index"
path_queries = "data/queries_train.csv"
training_queries(path_queries, path_queries_rels, inverted_index_path)