import pandas as pd

# Charger le fichier CSV dans un DataFrame
df = pd.read_csv('./data/queries_train.csv')

# Lire le fichier texte contenant les nouvelles queries
with open('./data/queries_rewrited_train.txt', 'r') as file:
    new_queries = file.readlines()

# Modifier le champ "query" dans le DataFrame avec les nouvelles queries
for i, new_query in enumerate(new_queries):
    df.at[i-1, 'query'] = new_query.strip()

# Sauvegarder le DataFrame modifié dans un nouveau fichier CSV
df.to_csv('./data/queries_rewrited_train.csv', index=False)
