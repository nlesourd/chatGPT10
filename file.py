import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('./data/queries_train.csv')

# Read the text file containing the new queries
with open('./data/queries_rewrited_train.txt', 'r') as file:
    new_queries = file.readlines()

# Modify the "query" field in the DataFrame with the new queries
for i, new_query in enumerate(new_queries):
    df.at[i-1, 'query'] = new_query.strip()

# Save the modified DataFrame in a new CSV file
df.to_csv('./data/queries_rewrited_train.csv', index=False)
