import pandas as pd
import pyterrier as pt

if not pt.started():
    pt.init()

pd_indexer = pt.DFIndexer("./data/pd_index")

# Load data from TSV file
file_path = './data/collection_test.tsv'
df = pd.read_csv(file_path, delimiter='\t')
df.columns = ['docno','body']

index_ref = pd_indexer.index(df['body'], df)
index = pt.IndexFactory.of(index_ref)
print(index.getCollectionStatistics().toString())

"""
# Load linguistic model
nlp = spacy.load('en_core_web_sm')

def preprocess_text(text):
    doc = nlp(text)
    # Tokenization and lemmatization
    tokens = [token.lemma_ for token in doc]
    # Delete stop words
    tokens = [token for token in tokens if not nlp.vocab[token].is_stop]

    tokens = [token.text for token in tokens if not token.is_punct]
    return tokens

for index, row in df.iterrows():
    doc_id = row[0]
    body = preprocess_text(row[1])
print(body)
"""

