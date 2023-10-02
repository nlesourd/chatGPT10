import preprocessing as ppc
import evaluation as eval
import pyterrier as pt
from pyterrier_t5 import MonoT5ReRanker, DuoT5ReRanker
import numpy as np

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
bm25 = pt.BatchRetrieve(index, wmodel="BM25", num_results=1000, controls={"wmodel": "default", "ranker.string": "default"})
bm25_results = bm25.transform(query)
bm25_results.sort_values(by=["score"], ascending=False, inplace=True)

# Rerank top 1000 documents with monoT5
monoT5 = MonoT5ReRanker() # loads castorini/monot5-base-msmarco by default
duoT5 = DuoT5ReRanker() # loads castorini/duot5-base-msmarco by default
mono_pipeline = bm25 >> monoT5
duo_pipeline = mono_pipeline % 100 >> duoT5
results = duo_pipeline.transform(query)

#Evaluation
#ppc.set_subcollection(collection_path, subcollection_path, results['docid'])
qid = "4_1"
eval.set_results_with_trec_format(results, trec_file_path, qid)

"""
# some queries that have to be treated specifiquely
query = "What are Cubesats"
query = "What can be done to stop it"
query = "done stop"
query = "What are some others"
query = "What about Ivanka"
"""