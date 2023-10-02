import preprocessing as ppc
import evaluation as eval
import pyterrier as pt
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
bm25 = pt.BatchRetrieve(inverted_index_path, wmodel="BM25", num_results=2000, controls={"wmodel": "default", "ranker.string": "default"})
bm25_results = bm25.transform([query])
bm25_results.sort_values(by=["score"], ascending=False, inplace=True)

# For the top 2000 rerank with PL2
pl2 = pt.BatchRetrieve(inverted_index_path, wmodel="PL2", num_results=1000, controls={"wmodel": "default", "ranker.string": "default"})
pl2_results = pl2.transform([query])
ppc.set_subcollection(collection_path, subcollection_path, pl2_results['docid'])

qid = "4_1"
eval.set_results_with_trec_format(pl2_results, trec_file_path, qid)



"""
# some queries that have to be treated specifiquely
query = "What are Cubesats"
query = "What can be done to stop it"
query = "done stop"
query = "What are some others"
query = "What about Ivanka"
"""