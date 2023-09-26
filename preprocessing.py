import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
import pyterrier as pt
import pandas as pd

def load_inverted_index(inverted_index_path: str, collection_path: str) -> None:
    """Generate inverted index if it not exists otherwise load it.

    Args:
        inverted_index_path: Path for the folder of inverted index.
        collection_path: Path of TSV collection file.

    Returns:
        Index loaded with pyterrier type.
    """
    if not pt.started():
        pt.init()

    if not os.path.exists(inverted_index_path):
        data = pd.read_csv(collection_path, delimiter='\t')
        df = pd.DataFrame({'docno': data.values[:,0].astype(str), 'text': data.values[:,1]})
        pd_indexer = pt.DFIndexer(inverted_index_path)
        indexref = pd_indexer.index(df["text"], df["docno"])
        index = pt.IndexFactory.of(indexref)
    else:
        indexref = pt.IndexRef.of(inverted_index_path)
        index = pt.IndexFactory.of(indexref)

    return index

def get_body(collection_path: str, docid: int) -> str:
    data = pd.read_csv(collection_path, delimiter='\t')
    return data.values[docid,1]


