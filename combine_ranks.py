import csv
from collections import defaultdict

# Create a dictionnary to retrieve the ranks for each qid
def create_dic_qid_rank(path_file_input: str):
# Read the text file containing the new queries
    with open(path_file_input, 'r') as file:
        reader = csv.reader(file, delimiter='\t') 
        # Ignore the first line
        next(reader)
        dic_qid_rank = defaultdict(dict)
        rank = 0
        # Iteration on lines
        for l in reader:
            ligne = l[0].split(",")
            # Start of a new qid
            if rank%1001 == 0:
                rank = 1
                qid = ligne[0]
            dic_qid_rank[qid][ligne[1]] = rank
            rank += 1
    return dic_qid_rank

# Combine the ranks of two files with given weights for the ranks
def combine_ranks(path_file_input1: str, path_file_input2: str, weight1: int, 
                  weight2: int, path_file_output: str):
    
    # Retrieve the ranks of the two files
    dic_qid_rank1 = create_dic_qid_rank(path_file_input1)
    dic_qid_rank2 = create_dic_qid_rank(path_file_input2)

    for qid in dic_qid_rank1:
        # Retrieve all the id of documents
        l_docs = set(list(dic_qid_rank1[qid].keys()) + list(dic_qid_rank2[qid].keys()))
        print(len(l_docs))


combine_ranks('data\kaggle_advanced_method.csv', 'data\kaggle.csv', 0.5, 
                  0.5, 'data\combined_ranks.csv')