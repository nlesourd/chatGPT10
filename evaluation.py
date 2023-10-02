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

def grade_out_of_4_rank(doc: str, bm25_rank: pd.core.frame.DataFrame, 
                        pl2_reranked: pd.core.frame.DataFrame) -> int:
    """Recover the defined scores between 0 and 4 for every query.

    Args:
        doc : the document to score
        bm25_rank : the bm25 rank
        pl2_reranked : the pl2 re-ranked

    Returns:
        return the predicted score between 0 to 4
    """
    score = 0
    re_rank = pl2_reranked[pl2_reranked['docid'] == int(doc)].index
    rank = bm25_rank[bm25_rank['docid'] == int(doc)].index
    # if re-ranked
    if len(re_rank) > 0:
        re_rank = re_rank[0]
        if re_rank < 30:
            score = 4
        elif re_rank < 150:
            score = 3
        else:
            score = 2

    # else if ranked
    elif len(rank) > 0:
        rank = rank[0]
        if rank < 500 :
            score = 2
        elif rank < 1000:
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

def add_lines_trec_format(bm25_rank: pd.core.frame.DataFrame, pl2_reranked: Union[pd.core.frame.DataFrame, None],
                          results: TextIO, qid: str):
    """ Add 1000 lines to results with the trec format

    Args:
        bm25_rank : the bm25 rank
        pl2_reranked : the pl2 re-ranked
        results : file where we write
        qid : id of the query
    """
    if type(pl2_reranked) != pd.core.frame.DataFrame:
        nb_lines = 1000
        max_bm25 = bm25_rank.loc[0]['score']
        min_bm25 = bm25_rank.loc[nb_lines-1]['score']
        for rank in range(0, nb_lines):
            current_line = bm25_rank.loc[rank]
            query_id = qid
            Q0 = "Q0"
            document_id = current_line['docid']
            # adapt the retrieval score to correspond to the same order than pl2
            retrieval_score = (current_line['score'] - min_bm25) / (max_bm25 - min_bm25)
            run_id = "BM25"
            runfile_line = f"{query_id} {Q0} {document_id} {rank + 1} {retrieval_score} {run_id}"
            results.write(runfile_line + "\n")
    
    else:
        # init extreme values
        nb_lines_rr = pl2_reranked.shape[0]
        nb_lines_r = bm25_rank.shape[0]
        print(pl2_reranked.loc[0]['docid'])
        max_pl2 = pl2_reranked.loc[0]['score']
        max_bm25 = bm25_rank.loc[0]['score']
        min_pl2 = pl2_reranked.loc[nb_lines_rr-1]['score']
        min_bm25 = bm25_rank.loc[nb_lines_r-1]['score']

        # create a relative link between the scores of the 2 models
        for rank in range(nb_lines_rr):
            current_line = pl2_reranked.loc[rank]
            query_id = qid
            Q0 = "Q0"
            document_id = current_line['docid']
            # adapt the retrieval score to correspond to the same order than bm25
            retrieval_score = (current_line['score'] - min_bm25 / (min_pl2 - min_bm25)) / (max_pl2 - min_bm25 / (min_pl2 - min_bm25))
            run_id = "PL2"
            runfile_line = f"{query_id} {Q0} {document_id} {rank + 1} {retrieval_score} {run_id}"
            results.write(runfile_line + "\n")

        # create a relative link between the scores of the 2 models
        link_pl2_bm25 = retrieval_score
        max200_bm25 = bm25_rank.loc[nb_lines_rr]['score']
        for rank in range(nb_lines_rr, nb_lines_r):
            current_line = bm25_rank.loc[rank]
            query_id = qid
            Q0 = "Q0"
            document_id = current_line['docid']
            # adapt the retrieval score to correspond to the same order than pl2
            retrieval_score = (current_line['score'] - min_bm25) / (max200_bm25 - min_bm25) * link_pl2_bm25
            run_id = "BM25"
            runfile_line = f"{query_id} {Q0} {document_id} {rank + 1} {retrieval_score} {run_id}"
            results.write(runfile_line + "\n")
        
        # To see if there are errors
        for rank in range(nb_lines_r, 1000):
            # fill with a false line
            runfile_line = f"{query_id} {Q0} {'FAUX'} {rank + 1} {0} {run_id}"
        results.write(runfile_line + "\n")

def set_results_with_trec_format(results: pd.core.frame.DataFrame, path_file_output: str, qid:int, 
                                 nb_lines = 1000, run_id = "monoT5"):
    """ Add 1000 lines to results with the trec format

    Args:
        results: Dataframe results of the retrieval ranking of each document with qid, docid, docno, rank, score and query
        path_file_output: path of the output file with TREC format
        nb_lines: optional, nuumber of documents ranked
        run_id, optional, id of retrieval process  
    """
    results_max = max(results['score'])
    lines = ""
    for rank in range(0, nb_lines):
        current_line = results.loc[rank]
        query_id = qid
        document_id = current_line['docid']
        retrieval_score = current_line['score'] / results_max # to normalize the score
        Q0 = "Q0"
        lines += f"{query_id} {Q0} {document_id} {rank + 1} {retrieval_score} {run_id}\n"

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
    # Recover the querie's scores
    queries_scores = scores_recovery(path_queries_rels)
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