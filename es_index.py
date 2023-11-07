from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import csv


es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])
index_name = "index_texts"
shard_count = 1
replica_count = 0

request_body = {
    "settings": {
        "number_of_shards": shard_count,
        "number_of_replicas": replica_count
    },
    "mappings": {
        "properties": {
            "content": {
                "type": "text"
            }
        }
    }
}
# ignore the creation if already exists
es.indices.create(index=index_name, body=request_body, ignore=400) 

collection_path = "./data/collection.tsv"
with open(collection_path, 'r', newline='', encoding='utf-8') as tsvfile:
    tsvreader = csv.reader(tsvfile, delimiter='\t')
    # Iteration on lines
    actions = []
    for i, line in enumerate(tsvreader):
        docno = line[0]
        text = line[1]
        document_data = {"content": text}
        action = {
            "_op_type": "index",
            "_index": index_name,
            "_id": docno,         
            "_source": document_data
        }
        actions.append(action)

        if i%10000 == 0 and i > 0:
            print(i)
            success, failed = bulk(es, actions, raise_on_error=True)
            print(f"Indexed {success} documents, {failed} failed.")
            actions = []
        
        elif i>8840000:
            docno = line[0]
            text = line[1]
            document_data = {"content": text}
            action = {
                "_op_type": "index",
                "_index": "index_texts",
                "_id": docno,         
                "_source": document_data
            }
            actions.append(action)
            print(i)
            success, failed = bulk(es, actions, raise_on_error=True)
            print(f"Indexed {success} documents, {failed} failed.")
            actions = []