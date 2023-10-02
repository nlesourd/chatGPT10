# chatGPT10

## Required libraries

All the libraries in the list below must be on your machine in order to run the two python files.

- os
- pyterrier
- pandas

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

Test

"""bash
./trec_eval -q qrels_file results_file
"""