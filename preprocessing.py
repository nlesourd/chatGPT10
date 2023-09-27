import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
import pyterrier as pt
import pandas as pd
from sqlitedict import SqliteDict
import csv 
from typing import List
import nltk
import re
nltk.download("stopwords")
STOPWORDS = set(nltk.corpus.stopwords.words("english"))


def textStoring(path_collection, path_sqldict):
    if not os.path.exists(path_sqldict):
        dictText = SqliteDict(path_sqldict)
        # Limit of size 
        csv.field_size_limit(4096 * 4096)
        # Open the TSV file
        with open(path_collection, 'r', newline='', encoding='utf-8') as tsvfile:
            tsvreader = csv.reader(tsvfile, delimiter='\t')
            
            # Iteration on lines
            for i, line in enumerate(tsvreader):
                if i % 10000 == 0:
                    print(i)
                docno = line[0]
                text = line[1]
                dictText[docno] = text
        dictText.update()
        dictText.commit()
        print("Index updated.")

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

def query_preprocessing(doc: str) -> List[str]:
    """Preprocesses a string of text.

    Arguments:
        doc: A string of text.

    Returns:
        List of strings.
    """
    return [
        term
        for term in re.sub(r"[^\w]|_", " ", doc).lower().split()
        if term not in STOPWORDS
    ]