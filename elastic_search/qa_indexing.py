#!/usr/bin/env python
# Licensed to Elasticsearch B.V under one or more agreements.
# Elasticsearch B.V licenses this file to you under the Apache 2.0 License.
# See the LICENSE file in the project root for more information

"""Script that downloads a public dataset and streams it to an Elasticsearch cluster"""

import json
import os
from os.path import abspath, join, dirname, exists
import tqdm
import urllib3
from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk

WORK_DIR = os.path.abspath(os.getcwd())
DATASET_PATH = join(WORK_DIR, "data/qa/es_data_with_covid.json")


def create_index(client):
    """Creates an index in Elasticsearch if one isn't already there."""
    client.indices.create(
        index="qa",
        # body={
        #     "settings": {"number_of_shards": 1},
        #     "mappings": {
        #         "properties": {
        #             "name": {"type": "text"},
        #             "borough": {"type": "keyword"},
        #             "cuisine": {"type": "keyword"},
        #             "grade": {"type": "keyword"},
        #             "location": {"type": "geo_point"},
        #         }
        #     },
        # },
        ignore=400,
    )


def generate_actions(data):
    """Reads the file through csv.DictReader() and for each row
    yields a single document. This function is passed into the bulk()
    helper to create many documents in sequence.
    """
    for idx, row in enumerate(data):
        doc = {
            "id": str(idx),
            "question": row["question"],
            "question_word": row["question_word"],
            "answer": row["answer"],
            "answer_display": row["answer_display"].replace("\n", "<br>"),
            "summary": row["summary"],
        }
        yield doc


def main():
    print("Loading dataset...")
    with open(DATASET_PATH, 'r') as open_file:
        data = json.load(open_file)
    number_of_docs = len(data)

    client = Elasticsearch(timeout=30, max_retries=10, retry_on_timeout=True)
    print("Creating an index...")
    create_index(client)

    print("Indexing documents...")
    progress = tqdm.tqdm(unit="docs", total=number_of_docs)
    successes = 0
    for ok, action in streaming_bulk(
        client=client, index="qa", actions=generate_actions(data),
    ):
        progress.update(1)
        successes += ok
    print("Indexed %d/%d documents" % (successes, number_of_docs))


if __name__ == "__main__":
    main()