import preprocessing as ppc
import evaluation as eval
import pyterrier as pt
from pyterrier_t5 import MonoT5ReRanker, DuoT5ReRanker

collection_path = "./data/collection.tsv"
subcollection_path = "./data/subcollection.tsv"
inverted_index_path = "./data/inverted_index"
trec_file_path = "./data/trec_file_results.txt"

query = "What was the neolithic revolution"
query = "What are Cubesats"

# create the inverted index if not init
print("Inverted index")
index = ppc.load_inverted_index(inverted_index_path, collection_path)
print(index.getCollectionStatistics().toString())

model_rank = pt.BatchRetrieve("./data/inverted_index", wmodel="BM25", 
                              controls={"wmodel": "default", "ranker.string": "default", "results" : 1000})
# model_rerank = pt.BatchRetrieve("./data/inverted_index", wmodel="PL2", 
#                                 controls={"wmodel": "default", "ranker.string": "default", "results" : 1000})

eval.rank_queries("data/reduced_qrels/first_query_queries.csv", "data/results.txt", model_rank , 
                 None, nb_reranked=1000)


"""
# 1st Rank with BM25
num_documents_1st_rank = 1000
bm25 = pt.BatchRetrieve(index, wmodel="BM25", num_results=num_documents_1st_rank, controls={"wmodel": "default", "ranker.string": "default", "results": num_documents_1st_rank})
bm25_results = bm25.transform(query)
"""
input_rerank = ppc.rerank_preprocessing(collection_path, bm25_results)

# Rerank top 1000 documents with monoT5
monoT5 = MonoT5ReRanker() # loads castorini/monot5-base-msmarco by default
monoT5_results = monoT5.transform(input_rerank)
"""
#Evaluation
#ppc.set_subcollection(collection_path, subcollection_path, results['docid'])
qid = "4_1"
<<<<<<< HEAD
eval.set_results_with_trec_format(bm25_results, trec_file_path, qid, nb_lines=num_documents_1st_rank, run_id="bm25")
=======
eval.set_results_with_trec_format(monoT5_results, trec_file_path, qid, nb_lines=num_documents_1st_rank, run_id="mono_T5")
"""
>>>>>>> fa1bba8059eb44ad6d144be94751f5b0be491751

"""
# some queries that have to be treated specifiquely
query = "What are Cubesats"
query = "What can be done to stop it"
query = "done stop"
query = "What are some others"
query = "What about Ivanka"
"""