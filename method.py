import pyterrier as pt
from pyterrier_t5 import MonoT5ReRanker
import pandas as pd
from preprocessing import query_preprocessing
import openai

# Set up your GPT-3 API key
openai.api_key = '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'

class Baseline:
    
    def __init__(self, inverted_index) -> None:
        self.inverted_index = inverted_index
        bm25 = pt.BatchRetrieve(self.inverted_index, wmodel="BM25", 
                              controls={"wmodel": "default", "ranker.string": "default", "results" : 1000})
        bo1 = pt.rewrite.Bo1QueryExpansion(self.inverted_index)
        self.first_rank_model = bm25 >> bo1 >> bm25
        self.rerank_model = None
        self.query_rewriting = False

    def rank_query(self, query : str) -> pd.core.frame.DataFrame:
        query_ppc = query_preprocessing(query, STOPWORDS_DEL = False)
        print(query_ppc)
        return self.first_rank_model.search(query_ppc)
        
class AdvancedMethod:
    
    def __init__(self, inverted_index, nb_reranked=1000, query_rewriting=False) -> None:
        self.nb_reranked = nb_reranked
        self.inverted_index = inverted_index
        self.first_rank_model = pt.BatchRetrieve(self.inverted_index, wmodel="BM25", 
                              controls={"wmodel": "default", "ranker.string": "default", "results" : 1000})
        self.rerank_model = pt.text.get_text(self.inverted_index, "text") >> MonoT5ReRanker()
        self.queries = []
        self.query_rewriting = query_rewriting

    #TODO
    def rank_query(self, query : str) -> pd.core.frame.DataFrame:
        self.queries.append(query)
        if self.query_rewriting:
            context = " ".join(self.queries)
            query = self.rewriting_query(context, query)
        query_ppc = query_preprocessing(query, STOPWORDS_DEL = False)
        print(query_ppc)
        results_first_rank = self.first_rank_model.search(query_ppc)
        results_rerank = self.rerank_model.transform(results_first_rank)
        return results_rerank        

    def rewriting_query(self, context, current_query):
        prompt = f"Context: {context},Current query: {current_query}"
        response = openai.Completion.create(
            engine="text-davinci-003",  # You can choose an appropriate engine
            prompt=prompt,
            max_tokens=50,  # Adjust the length of the generated response
            temperature=0.7  # Adjust the creativity of the generated response
        )
        query_rewrited = response['choices'][0]['text'].strip()
        return query_rewrited