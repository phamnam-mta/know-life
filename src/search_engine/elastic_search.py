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
                data.append(h["_source"])
        return data

    
    async def get_qa_pairs(self, question: Text, page_index=0, page_size=20):
        query_body = {
            "from": page_index,
            "size": page_size,
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
