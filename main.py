import preprocessing as ppc
import evaluation as eval
import pyterrier as pt
from pyterrier_t5 import MonoT5ReRanker, DuoT5ReRanker
import csv
from elasticsearch import Elasticsearch

# import torch
# import onmt
# from torchvision import models

# model = models.resnet18()
# model.load_state_dict(torch.load('../rewrite_onmt/rewrite_onmt_step_200000.pt', map_location=torch.device('cpu')))
# model.eval()
# print(model)

es = Elasticsearch()
print(es.get(index="index_texts", id='7231102')['_source']['content'])
# print(es.indices.stats(index="index_texts"))

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


bm25_results = model_rank.transform(model_rank.search(query))
print(type(bm25_results))
bo1 = pt.rewrite.Bo1QueryExpansion(index)
dph = model_rank
pipelineQE = dph >> bo1 >> dph

# eval.rank_queries("data/reduced_qrels/first_query_queries.csv", "data/results.txt", model_rank, 
#                  None, nb_reranked=1000)

monoT5 = MonoT5ReRanker()
# eval.rank_queries("data/queries_train.csv", "data/results.txt", pipelineQE, 
#                  None, nb_reranked=1000, nb_lines = 1000, kaggle = False)

# eval.rank_queries("data/queries_test.csv", "data/kaggle.csv", pipelineQE, 
#                   monoT5, nb_reranked=1000, nb_lines = 1000, kaggle = True)

eval.rank_queries("data/reduced_qrels/queries_test_suite.csv", "data/kaggle.csv", pipelineQE, 
                  monoT5, nb_reranked=1000, nb_lines = 1000, kaggle = True)

input_rerank = ppc.rerank_preprocessing(collection_path, bm25_results)

# Rerank top 1000 documents with monoT5
monoT5 = MonoT5ReRanker() # loads castorini/monot5-base-msmarco by default
monoT5_results = monoT5.transform(input_rerank)
"""
#Evaluation
#ppc.set_subcollection(collection_path, subcollection_path, results['docid'])
qid = "4_1"
eval.set_results_with_trec_format(monoT5_results, trec_file_path, qid, nb_lines=num_documents_1st_rank, run_id="mono_T5")
"""

"""
# some queries that have to be treated specifiquely
query = "What are Cubesats"
query = "What can be done to stop it"
query = "done stop"
query = "What are some others"
query = "What about Ivanka"
"""