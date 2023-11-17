# chatGPT10

## Required libraries

All the libraries in the list below must be on your machine in order to run the two python files.

- os
- pyterrier
- pyterrier_t5
- pandas
- csv
- os
- typing
- nltk
- re
- openai

If you don't have one of these libraries, you can install it using the python pip package manager.

`pip install <librarie's name>`

Or using requirements.txt like this:

`pip install -r requirements.txt`

## Run the code

`python3 main.py`

## Test the results

Download TREC Eval and compile it

"""bash
git clone https://github.com/usnistgov/trec_eval.git
cd trec_eval
make
"""

Tests

- Evaluation of the (entire) qrels_train return
"""bash
cd ../trec_eval/
./trec_eval -c -m recall.1000 -m map -m recip_rank -m ndcg_cut.3 -l2 -M1000 ../chatGPT10/data/qrels_train.txt ../chatGPT10/data/results.txt
"""

- Evaluation of the first queries qrels_train return (it allows to evaluate the first queries that don't depend of other queries)
"""bash
cd ../trec_eval/
./trec_eval -c -m recall.1000 -m map -m recip_rank -m ndcg_cut.3 -l2 -M1000 ../chatGPT10/data/reduced_qrels/first_query_qrels.txt ../chatGPT10/data/results.txt
"""
