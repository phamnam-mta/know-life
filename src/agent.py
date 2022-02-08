import functools
from typing import (
    Any,
    Callable,
    Optional,
    Text,
    Union,
    List
)
from src.search_engine import EntitySearch, SemanticSearch, ESKnowLife
from src.medical_test_service.medical_test import MedicalTest
from src.utils.constants import (
    ResponseAttribute
)

def agent_must_be_ready(f: Callable[..., Any]) -> Callable[..., Any]:
    """Any Agent method decorated with this will raise if the agent is not ready."""

    @functools.wraps(f)
    def decorated(self, *args: Any, **kwargs: Any) -> Any:
        if not self.is_ready():
            raise Exception(
                "Agent needs to be prepared before usage. You need to set a "
                "processor and a tracker store."
            )
        return f(self, *args, **kwargs)

    return decorated

class Agent():
    def __init__(self,
            kb_model_dir: Text,
            kb_data_dir: Text,
            qa_model_dir: Text, 
            es_url: Text, 
            index: Text
    ) -> None:
        self.entity_search = EntitySearch(
            model_dir=kb_model_dir,
            data_dir=kb_data_dir,)
        self.semantic_search = SemanticSearch(qa_model_dir, es_url, index)
        self.elastic_search = ESKnowLife(es_url, index)
        self.medical_test = MedicalTest()

    @classmethod
    def load_agent(cls,
            kb_model_dir: Text,
            kb_data_dir: Text,
            qa_model_dir: Text, 
            es_url: Text, 
            index: Text
        ):
        agent = Agent(
            kb_model_dir=kb_model_dir,
            kb_data_dir=kb_data_dir,
            qa_model_dir=qa_model_dir, 
            es_url=es_url, 
            index=index)
        return agent

    def is_ready(self) -> bool:
        """Check if all necessary components are instantiated to use agent."""
        return self.entity_search is not None and self.semantic_search is not None and self.elastic_search is not None and self.medical_test is not None

    @agent_must_be_ready
    def search_by_entity(self, text: Text):
        resp = self.entity_search.query(text)        
        return resp

    @agent_must_be_ready
    async def search_by_semantic(self, text: Text, to_return= ResponseAttribute.ALL.value, page_size=0, page_index=20):
        re_ranking, es_ranking = await self.semantic_search.search(text, to_return=to_return, page_size=page_size, page_index=page_index)
        return re_ranking, es_ranking

    @agent_must_be_ready
    async def search_by_elastic(self, text: Text, page_size=0, page_index=20):
        es_ranking = await self.elastic_search.search(text, page_size=page_size, page_index=page_index)
        return es_ranking

    @agent_must_be_ready
    async def medical_test_suggestion(self, indicators: List):
        suggestions = self.medical_test.get_suggestions(indicators)
        for s in suggestions:
            s["relevant_questions"] = await self.elastic_search.search(s["name"], page_index=0, page_size=3)
        return suggestions
    