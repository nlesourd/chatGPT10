from typing import List, Tuple, Union

import csv


# Limit of size 
def load_data(path: str) -> Tuple[List[str], List[int]]:
    csv.field_size_limit(4096 * 4096)
    # Open the TSV file
    with open(path, 'r', newline='', encoding='utf-8') as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter='\t')
        print(tsvreader[10])
        # Ignore the first line
        # next(tsvreader)

        contents = []
        label = []
        i=0
        # Iteration on lines
        for i, line in enumerate(tsvreader):
            if i < 10:
                print(line)
    return contents, label

load_data("collection.tsv")