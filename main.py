import preprocessing as ppc
import evaluation as eval
import pyterrier as pt
from pyterrier_t5 import MonoT5ReRanker, DuoT5ReRanker

collection_path = "./data/collection.tsv"
subcollection_path = "./data/subcollection.tsv"
inverted_index_path = "./data/inverted_index"
trec_file_path = "./data/trec_file_results.txt"

query = "What was the neolithic revolution"

# create the inverted index if not init
print("Inverted index")
index = ppc.load_inverted_index(inverted_index_path, collection_path)
print(index.getCollectionStatistics().toString())

# 1st Rank with BM25
num_documents_1st_rank = 10
bm25 = pt.BatchRetrieve(index, wmodel="BM25", num_results=num_documents_1st_rank, controls={"wmodel": "default", "ranker.string": "default"})
bm25_results = bm25.transform(query)
bm25_results.sort_values(by=["score"], ascending=False, inplace=True)

input_rerank = ppc.rerank_preprocessing(collection_path, bm25_results)

# Rerank top 1000 documents with monoT5
monoT5 = MonoT5ReRanker() # loads castorini/monot5-base-msmarco by default
monoT5_results = monoT5.transform(input_rerank)

#Evaluation
#ppc.set_subcollection(collection_path, subcollection_path, results['docid'])
qid = "4_1"
eval.set_results_with_trec_format(monoT5_results, trec_file_path, qid, nb_lines=num_documents_1st_rank, run_id="mono_T5")

"""
# some queries that have to be treated specifiquely
query = "What are Cubesats"
query = "What can be done to stop it"
query = "done stop"
query = "What are some others"
query = "What about Ivanka"
"""