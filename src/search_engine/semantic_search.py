import logging
from typing import Text, List
from src.search_engine.elastic_search import ESKnowLife
from src.nlu import BERTRanker
from src import ELASTICSEARCH_URL
from src.utils.constants import (
    QA_INDEX,
    QA_MODEL_DIR,
    ResponseAttribute
)

logger = logging.getLogger(__name__)
class SemanticSearch():
    def __init__(self, model_path=QA_MODEL_DIR, es_url=ELASTICSEARCH_URL, index=QA_INDEX) -> None:
        self.retrieval = ESKnowLife(es_url, index)
        self.ranker = BERTRanker(model_path)
        logger.info("Ranking Model loaded")

    async def search(self, question: Text, to_return=ResponseAttribute.ALL, page_size=0, page_index=20):
        es_data = await self.retrieval.get_qa_pairs(question, page_index, page_size)
        pairs = self.retrieval.elastic_to_qa(es_data)

        candidates = [p["summary"] for p in pairs]
        ranking, scores = self.ranker.re_ranking(question, candidates)

        if to_return == ResponseAttribute.ANSWER:
            es_ranking = [{
                "id": p["id"],
                "answer": p["answer_display"]
            } for p in pairs]

            re_ranking = [{
                "id": pairs[i]["id"],
                "score": str(scores[i]),
                "answer": pairs[i]["answer_display"]
            } for i in ranking]
        else:
            es_ranking = [{
                "id": p["id"],
                "question": p["question"],
                "answer": p["answer_display"]
            } for p in pairs]

            re_ranking = [{
                "id": pairs[i]["id"],
                "score": str(scores[i]),
                "question": pairs[i]["question"],
                "answer": pairs[i]["answer_display"]
            } for i in ranking]

        return re_ranking, es_ranking

        