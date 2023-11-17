import pyterrier as pt
from pyterrier_t5 import MonoT5ReRanker, DuoT5ReRanker
import pandas as pd
from preprocessing import query_preprocessing
import openai

# Set up your GPT-3 API key, has to be filled with your own
openai.api_key = ''

class Baseline:
    
    def __init__(self, inverted_index) -> None:
        """Initialization of baseline.

        Args:
            inverted_index: inverted index needed for the BM25 process.
        """
        self.inverted_index = inverted_index
        bm25 = pt.BatchRetrieve(self.inverted_index, wmodel="BM25", 
                              controls={"wmodel": "default", "ranker.string": "default", "results" : 1000})
        bo1 = pt.rewrite.Bo1QueryExpansion(self.inverted_index)
        self.pipeline = bm25 >> bo1 >> bm25
        self.rerank_model = None
        self.query_rewriting = False

    def rank_query(self, query: str) -> pd.core.frame.DataFrame:
        """Rank query using the baseline pipeline.

        Args:
            query: sent by user to get 1000 most relevant documents.

        Returns:
            Results containing 1000 most relevant documents.
        """
        query_ppc = query_preprocessing(query, STOPWORDS_DEL = False)
        print(query_ppc)
        return self.pipeline.search(query_ppc)
    


class AdvancedMethod:
    
    def __init__(self, inverted_index, nb_reranked_mono=1000, nb_reranked_duo=10 ,query_rewriting=True) -> None:
        """Initialization of advanced method.

        Args:
            inverted_index: inverted index needed for the BM25 process.
            nb_reranked_mono: default 1000, number of documents reranked with monoT5
            nb_reranked_duo: default 10, number of documents reranked with duoT5
            query_rewriting: default True, if you want the query rewriting or not
        """
        self.inverted_index = inverted_index
        bm25 = pt.BatchRetrieve(self.inverted_index, wmodel="BM25", 
                              controls={"wmodel": "default", "ranker.string": "default", "results" : nb_reranked_mono})
        monoT5 = MonoT5ReRanker()
        duoT5 = DuoT5ReRanker()
        self.pipeline = bm25 >> pt.text.get_text(self.inverted_index, "text") >> monoT5 % nb_reranked_duo >> duoT5
        self.queries = []
        self.query_rewriting = query_rewriting

    def rank_query(self, query: str) -> pd.core.frame.DataFrame:
        """Rank query using the advanced method pipeline.

        Args:
            query: sent by user to get 1000 most relevant documents.

        Returns:
            Results containing 1000 most relevant documents.
        """
        self.queries.append(query)
        if self.query_rewriting:
            context = " ".join(self.queries)
            query = self.rewriting_query(context, query)
        query_ppc = query_preprocessing(query, STOPWORDS_DEL = False)
        print(query_ppc)
        results = self.pipeline.search(query_ppc)
        return results      

    def rewriting_query(self, context: str, current_query: str):
        """Query rewriting using chatGPT3.5 .

        Args:
            context: concatenation of previous queries.
            current query: last query submited by the user.

        Returns:
            New query rewrite using context and knoledge injection.
        """
        prompt = f"Context: {context},Current query: {current_query}"
        response = openai.Completion.create(
            engine="text-davinci-003",  # You can choose an appropriate engine
            prompt=prompt,
            max_tokens=50,  # Adjust the length of the generated response
            temperature=0.7  # Adjust the creativity of the generated response
        )
        query_rewrited = response['choices'][0]['text'].strip()
        return query_rewrited