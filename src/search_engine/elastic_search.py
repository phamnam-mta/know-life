from typing import Text, Dict, List
from elasticsearch import AsyncElasticsearch
from src import ELASTICSEARCH_URL
from src.utils.constants import QA_INDEX, QA_QUERY_FIELDS

class ESKnowLife():
    def __init__(self, es_url=ELASTICSEARCH_URL, index=QA_INDEX) -> None:
        self.es =  AsyncElasticsearch(hosts=es_url)
        self.index = index

    def elastic_to_qa(self, raw_data):
        data = []
        if raw_data and len(raw_data["hits"]["hits"]) > 0:
            for h in raw_data["hits"]["hits"]:
                record = {
                    "highlight": h.get("highlight")
                }
                record.update(h["_source"])
                data.append(record)
        return data

    
    async def get_qa_pairs(self, question: Text, page_index=0, page_size=20):
        query_body = {
            "from": page_index,
            "size": page_size,
            "highlight": {
                "number_of_fragments": 0,
                "fields": {
                    "question": {},
                    "answer_display": {}
                }
            },
            "query": {
                "multi_match" : {
                "query": question,
                "fields": QA_QUERY_FIELDS
                }
            }
        }

        return await self.es.search(
            index=self.index,
            body=query_body
        )

    async def search(self, question: Text, page_index=0, page_size=20):
        es_data = await self.get_qa_pairs(question, page_index, page_size)
        pairs = self.elastic_to_qa(es_data)

        es_ranking = [{
                "id": p["id"],
                "question": p["question"],
                "answer": p["answer_display"],
                "highlight": {
                    "question": p["highlight"].get("question"),
                    "answer": p["highlight"].get("answer_display")
                }
        } for p in pairs]

        return es_ranking
