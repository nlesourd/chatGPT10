import preprocessing as ppc
import pyterrier as pt

collection_path = "./data/collection.tsv"
inverted_index_path = "./data/inverted_index"

# create the sqlite dict if not init
print("sqlDict")
ppc.textStoring('data/collection.tsv','data/collection.sqlite')
# create the inverted index if not init
print("Inverted index")
index = ppc.load_inverted_index(inverted_index_path, collection_path)


print(index.getCollectionStatistics().toString())

#1st Rank with BM25
batch_retriever = pt.BatchRetrieve(inverted_index_path, wmodel="BM25")

query = "Mickael Jackson"
results = batch_retriever.transform([query])

#Top 1000 document
print(results["docid"])
#print(ppc.get_body(collection_path, 7421124))
pl2 = pt.BatchRetrieve(index, wmodel="PL2")
pipeline = (batch_retriever % 100) >> pl2
query = "Mickael Jackson"
results = pipeline.transform([query])
print(results["docid"])

print("top 1 : BM25")
print(ppc.get_body(collection_path, 6391464))
print("top 1 : PL2")
print(ppc.get_body(collection_path, 3688940))
