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
  

"""
def rank_query(self, query : str, method, query_before : str) -> pd.core.frame.DataFrame:
        # Ranking the docs relatively to the query
        results_rank = method.first_rank_model.search(query)
        if results_rank.shape[0]<1000:
            print(query)
            results_rank = method.first_rank_model.search(query_before + " " + query)
        # Re-Ranking the docs relatively to the query
        if method.rerank_model == None:
            pipeline = method.first_rank_model
            final_rank = pipeline.search(query)
        # Mono-T5 re-ranking
        elif isinstance(method.rerank_model, MonoT5ReRanker):
            # Add the text to the results
            l_texts = []
            for i in range(self.nb_reranked):
                doc_no = results_rank.loc[i]['docno']
                l_texts.append(es.get(index="index_texts", id=doc_no)['_source']['content'])
            results_rank['text'] = l_texts
            final_rank = method.rerank_model.transform(results_rank)
            # Sort
            final_rank.sort_values(by='score', ascending=False, inplace=True)
            print(final_rank)
        # Other re-ranking

        return final_rank
"""