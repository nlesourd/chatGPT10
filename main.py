import preprocessing as ppc
import evaluation as eval
import method

COLLECTION_PATH = "./data/collection.tsv"
SUBCOLLECTION_PATH = "./data/subcollection.tsv"
INVERTED_INDEX_PATH = "./data/inverted_index"
QUERIES_TRAIN_PATH = "./data/queries_train.csv"
QUERIES_REWRITED_TRAIN_PATH = "./data/queries_rewrited_train.csv"
QUERIES_TEST_PATH = "./data/queries_test.csv"
QUERIES_REWRITED_TEST_PATH = "./data/queries_rewrited_test_v2.csv"
TREC_FILE_PATH = "./data/trec_file_results.txt"

# create the inverted index if not init
print("Inverted index")
inverted_index = ppc.load_inverted_index_trec(INVERTED_INDEX_PATH)
print(inverted_index.getCollectionStatistics().toString())

# initialization of methods
baseline = method.Baseline(inverted_index)
advanced_method = method.AdvancedMethod(inverted_index)

# evaluation
eval.rank_queries(QUERIES_REWRITED_TEST_PATH, "./data/kaggle_advanced_method_v3_2.csv", method=advanced_method, kaggle=True)