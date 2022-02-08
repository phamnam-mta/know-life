import os
from typing import List, Text
import warnings

from src.utils.io import *
from src.nlu import BERTEntityExtractor
from src.utils.fuzzy import is_relevant_string, get_fuzzy_score
from src.utils.constants import (
    KB_DEFAULT_MODEL_DIR,
    KB_DEFAULT_DATA_DIR,
    SYNONYM_KEY
)
from src.data_provider.neo4j_provider import Neo4jProvider

class EntitySearch():
    def __init__(self,
        model_dir=KB_DEFAULT_MODEL_DIR,
        data_dir=KB_DEFAULT_DATA_DIR):

        self.ner = BERTEntityExtractor(
            model_dir=model_dir, data_dir=data_dir)

        self.provider = Neo4jProvider()
        
    def query(self, question):
        '''
        Args:
            - quesion (str) : utterance
        Return:
            result (dict) { 
                "answers" (list) ,
                "ner_response" (list of dict),
                "answer_dislay" (list): 
            }
        '''
        entities = self.ner.inference(question)

        result = []
        index = 0
        
        for intent in entities['intent']:
            request = {
                'symptom' : entities['symp'],
                'disease' : entities['disease'],
                'intent' : intent
            }
            prettier_answer = self.provider.query(request)  # list of str
            highlight_terms = []
            score = 100
            result.append({
                "id": index,
                "score": score,
                "question": question,
                "answer_dislay" : prettier_answer,
                "highlight": {
                    "question": [self.get_highlight(question, highlight_terms)],
                    "answer_dislay": [self.get_highlight(prettier_answer, highlight_terms)],
                },
            })
            index += 1

        return result

    def get_highlight(self, text: Text, terms: List):
        highlight = text
        for t in terms:
            t = f" {t} "
            highlight = highlight.replace(t, f"<em>{t}</em>")
        return highlight

    def query_single_entity(self, entity, relation):
        '''
        Return:
            - result (list)
            - kb_answer (list)
        '''
        results = []
        scores = []
        kb_answer = []

        result = ""

        for sample in self.database:
            if SYNONYM_KEY in sample:
                for synonym in sample[SYNONYM_KEY]:
                    is_relevant, score = is_relevant_string(synonym,entity,method=['exact','fuzzy','include'],return_score=True)
                    if is_relevant:
                        break
            else:
                is_relevant, score = is_relevant_string(sample['disease'],entity,method=['exact','fuzzy','include'],return_score=True)
            if is_relevant:
                for att in sample['attributes']:
                    if att['attribute'] == relation:
                        # short content
                        # try:
                        #     result = att['short_content']
                        # except:
                        #     result = att['content']

                        result = att['content']

                        results.append(result)
                        scores.append(score)
                        kb_answer.append([sample['disease'],att['attribute'],score])

        # Re-ranking
        result, kb_answer = self.reranking(kb_answer,results,entity)
        
        return result, kb_answer

