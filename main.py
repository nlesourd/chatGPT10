import preprocessing as ppc
import pyterrier as pt
import numpy as np

collection_path = "./data/collection.tsv"
inverted_index_path = "./data/inverted_index"

# create the sqlite dict if not init
# print("sqlDict")
# ppc.textStoring('data/collection.tsv','data/collection.sqlite')

# create the inverted index if not init
print("Inverted index")
index = ppc.load_inverted_index(inverted_index_path, collection_path)
print(index)

print(index.getCollectionStatistics().toString())

#1st Rank with BM25
batch_retriever = pt.BatchRetrieve(inverted_index_path, wmodel="BM25", 
                                   num_results=2000, controls={"wmodel": "default", "ranker.string": "default"})

query = "Mickael Jackson"
query = "Tell me about mechanical energy"
# some queries that have to be treated specifiquely
query = "What are Cubesats"
query = "What can be done to stop it"
query = "done stop"
query = "What are some others"
query = "What about Ivanka"
results = batch_retriever.transform([query])
print(results)
print('8648995' in results['docid'])
print(type(results))
print(results)
print(results.loc[2])
size = results.shape
nb_lines = size[0]
print(nb_lines)

pl2 = pt.BatchRetrieve(inverted_index_path, wmodel="PL2",
                       num_results=200, controls={"wmodel": "default", "ranker.string": "default"})
pipeline = (batch_retriever % 200) >> pl2
pl2_re_ranked = pipeline.search(query)
print(pl2_re_ranked)
print(np.where(pl2_re_ranked['docid'] == int('5092060'))[0])

# print(np.where(results['docid'] == 6391464)[0][0])
# type(results['docid'])

# #Top 1000 document
# print(results["docid"])
# #print(ppc.get_body(collection_path, 7421124))
# pl2 = pt.BatchRetrieve(inverted_index_path, wmodel="PL2")
# nb_docs_reranked = 100
# pipeline = (batch_retriever % nb_docs_reranked) >> pl2
# query = "Mickael Jackson"
# results = pipeline.transform([query])


# # print("top 1 : BM25")
# # print(ppc.get_body(collection_path, 6391464))
# # print("top 1 : PL2")
# # print(ppc.get_body(collection_path, 3688940))
