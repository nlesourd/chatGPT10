import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
import pyterrier as pt
from typing import List
import nltk
import re
nltk.download("stopwords")
STOPWORDS = set(nltk.corpus.stopwords.words("english"))


def msmarco_generate():
    """Generate msmarco collection."""

    dataset = pt.get_dataset("trec-deep-learning-passages")
    with pt.io.autoopen(dataset.get_corpus()[0], 'rt') as corpusfile:
        for l in corpusfile:
            docno, passage = l.split("\t")
            yield {'docno' : docno, 'text' : passage}

def load_inverted_index_trec(inverted_index_path: str):
    """Generate inverted index if it not exists otherwise load it.

    Args:
        inverted_index_path: Path for the folder of inverted index.
        collection_path: Path of TSV collection file.

    Returns:
        Index loaded with pyterrier type.
    """
    if not pt.started():
        pt.init(mem=8000)

    if not os.path.exists(inverted_index_path):
        iter_indexer = pt.IterDictIndexer(inverted_index_path)
        indexref = iter_indexer.index(msmarco_generate(), meta={'docno' : 20, 'text': 4096})
        index = pt.IndexFactory.of(indexref)
    else:
        indexref = pt.IndexRef.of(inverted_index_path)
        index = pt.IndexFactory.of(indexref)

    return index

def query_preprocessing(query: str, STOPWORDS_DEL: True) -> List[str]:
    """Preprocesses a string of text.

    Arguments:
        doc: A string of text.

    Returns:
        List of strings.
    """
    if STOPWORDS_DEL:
        query_pp = [term for term in re.sub(r"[^\w]|_", " ", query).lower().split()
                                            if term not in STOPWORDS]
    # if query_pp empty or just one word
    # use here the frequency of words !! -> course 3
    if not(STOPWORDS_DEL) or len(query_pp) <= 1:
        query_pp = [term for term in re.sub(r"[^\w]|_", " ", query).lower().split()]
    
    return ' '.join(query_pp)