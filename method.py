import pyterrier as pt
from pyterrier_t5 import MonoT5ReRanker
import pandas as pd
from preprocessing import query_preprocessing


class Baseline:
    
    def __init__(self, inverted_index) -> None:
        self.inverted_index_path = inverted_index
        bm25 = pt.BatchRetrieve(self.inverted_index_path, wmodel="BM25", 
                              controls={"wmodel": "default", "ranker.string": "default", "results" : 1000})
        bo1 = pt.rewrite.Bo1QueryExpansion(self.inverted_index_path)
        self.first_rank_model = bm25 >> bo1 >> bm25
        self.rerank_model = None

    def rank_query(self, query : str) -> pd.core.frame.DataFrame:
        query_ppc = query_preprocessing(query, STOPWORDS_DEL = False)
        print(query_ppc)
        return self.first_rank_model.search(query_ppc)
        
class AdvancedMethod:
    
    def __init__(self, inverted_index, nb_reranked=1000) -> None:
        self.nb_reranked = nb_reranked
        self.inverted_index_path = inverted_index
        bm25 = pt.BatchRetrieve(self.inverted_index_path, wmodel="BM25", 
                              controls={"wmodel": "default", "ranker.string": "default", "results" : 1000})
        bo1 = pt.rewrite.Bo1QueryExpansion(self.inverted_index_path)
        self.first_rank_model = bm25 >> bo1 >> bm25
        self.rerank_model = MonoT5ReRanker()
        self.queries = []

    #TODO
    def rank_query(self, query : str) -> pd.core.frame.DataFrame:
        query_ppc = query_preprocessing(query, STOPWORDS_DEL = False)
        print(query_ppc)
        self.queries.append(query_ppc)
        return self.first_rank_model.search(query_ppc)
