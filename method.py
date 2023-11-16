import pyterrier as pt
from pyterrier_t5 import MonoT5ReRanker, DuoT5ReRanker
import pandas as pd
from preprocessing import query_preprocessing
import openai

# Set up your GPT-3 API key
openai.api_key = ''

class Baseline:
    
    def __init__(self, inverted_index) -> None:
        self.inverted_index = inverted_index
        bm25 = pt.BatchRetrieve(self.inverted_index, wmodel="BM25", 
                              controls={"wmodel": "default", "ranker.string": "default", "results" : 1000})
        bo1 = pt.rewrite.Bo1QueryExpansion(self.inverted_index)
        self.pipeline = bm25 >> bo1 >> bm25
        self.rerank_model = None
        self.query_rewriting = False

    def rank_query(self, query : str) -> pd.core.frame.DataFrame:
        query_ppc = query_preprocessing(query, STOPWORDS_DEL = False)
        print(query_ppc)
        return self.pipeline.search(query_ppc)
    


class AdvancedMethod:
    
    def __init__(self, inverted_index, nb_reranked_mono=1000, nb_reranked_duo=20 ,query_rewriting=False) -> None:

        self.inverted_index = inverted_index
        bm25 = pt.BatchRetrieve(self.inverted_index, wmodel="BM25", 
                              controls={"wmodel": "default", "ranker.string": "default", "results" : nb_reranked_mono})
        monoT5 = MonoT5ReRanker()
        duoT5 = DuoT5ReRanker()
        self.pipeline = bm25 >> pt.text.get_text(self.inverted_index, "text") >> monoT5 % nb_reranked_duo >> duoT5
        self.queries = []
        self.query_rewriting = query_rewriting

    def rank_query(self, query : str) -> pd.core.frame.DataFrame:
        self.queries.append(query)
        if self.query_rewriting:
            context = " ".join(self.queries)
            query = self.rewriting_query(context, query)
        query_ppc = query_preprocessing(query, STOPWORDS_DEL = False)
        print(query_ppc)
        results = self.pipeline.search(query_ppc)
        return results      

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