import preprocessing as ppc
import evaluation as eval
from elasticsearch import Elasticsearch
import method

COLLECTION_PATH = "./data/collection.tsv"
SUBCOLLECTION_PATH = "./data/subcollection.tsv"
INVERTED_INDEX_PATH = "./data/inverted_index"
QUERIES_TRAIN_PATH = "./data/queries_train.csv"
QUERIES_REWRITED_TRAIN_PATH = "./data/queries_rewrited_train.csv"
QUERIES_TEST_PATH = "./data/queries_test.csv"
QUERIES_REWRITED_TEST_PATH = "./data/queries_rewrited_test.csv"
TREC_FILE_PATH = "./data/trec_file_results.txt"

# initialization of elesticsearch
es = Elasticsearch()

# create the inverted index if not init
print("Inverted index")
inverted_index = ppc.load_inverted_index_trec(INVERTED_INDEX_PATH)
print(inverted_index.getCollectionStatistics().toString())

baseline = method.Baseline(inverted_index)
advanced_method = method.AdvancedMethod(inverted_index)

eval.rank_queries(QUERIES_TRAIN_PATH, "data/results.txt", method=advanced_method, nb_lines=1000, kaggle=False)

