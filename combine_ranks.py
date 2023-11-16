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
# We assume thats doc_ids are the same for ranking 1 and 2
def combine_ranks(path_file_input1: str, path_file_input2: str, weight1: int, 
                  weight2: int, path_file_output: str):
    
    # Retrieve the ranks of the two files
    dic_qid_rank1 = create_dic_qid_rank(path_file_input1)
    dic_qid_rank2 = create_dic_qid_rank(path_file_input2)

    # Put qid and docid at the top of the file
    with open(path_file_output, 'w') as file:
        file.write("qid,docid\n")

    with open(path_file_output, 'a') as file:
        for qid in dic_qid_rank1:
            # Retrieve all the id of documents
            l_docs = list(dic_qid_rank1[qid].keys())
            # Create a list of tuples with rank and id
            list_rank_id = []
            for docid in l_docs:
                new_rank = weight1 * dic_qid_rank1[qid][docid] + weight2 * dic_qid_rank2[qid][docid]
                list_rank_id.append((new_rank, docid))
            # Sort the list by rank
            list_rank_id = sorted(list_rank_id, key=lambda x: x[0])
            # Add in the doc
            for rank, docid in list_rank_id:
                file.write(qid + "," + docid + "\n")

combine_ranks('data\kaggle_advanced_method_v2.csv', 'data\kaggle_advanced_method.csv', 0.6, 
                  0.4, 'data\combined_ranks.csv')