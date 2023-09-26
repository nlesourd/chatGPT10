import preprocessing as ppc
import pyterrier as pt

collection_path = "./data/collection.tsv"
inverted_index_path = "./data/inverted_index"

index = ppc.load_inverted_index(inverted_index_path, collection_path)
print(index.getCollectionStatistics().toString())

#1st Rank with BM25
batch_retriever = pt.BatchRetrieve(inverted_index_path, wmodel="BM25")

query = "Mickeal Jackson"
results = batch_retriever.transform([query])

#Top 1000 document
print(results["docid"])
#print(ppc.get_body(collection_path, 7421124))