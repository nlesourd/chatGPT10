import pandas as pd
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
import pyterrier as pt

if not pt.started():
    pt.init()

# Load data from TSV file
file_path = './data/collection_test.tsv'
data = pd.read_csv(file_path, delimiter='\t')

df = pd.DataFrame({'docno': data.values[:,0].astype(str),
'text': data.values[:,1]})

inverted_index_path = "./data/inverted_index"
if not os.path.exists(inverted_index_path):
    pd_indexer = pt.DFIndexer(inverted_index_path)
    indexref = pd_indexer.index(df["text"], df["docno"])
    index = pt.IndexFactory.of(indexref)
    print(index.getCollectionStatistics().toString())
else:
    indexref = pt.IndexRef.of(inverted_index_path)
    index = pt.IndexFactory.of(indexref)
    print(index.getCollectionStatistics().toString())