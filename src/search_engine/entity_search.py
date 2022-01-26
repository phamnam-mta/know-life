import os
from typing import List, Text
import warnings

from src.utils.io import *
from src.utils.kb_utils import *
from src.nlu import BERTEntityExtractor
from src.utils.kb_utils import is_relevant_string, get_fuzzy_score
from src.utils.constants import (
    MAX_ANSWER_LENGTH,
    KB_DEFAULT_MODEL_DIR,
    KB_DEFAULT_DATA_DIR,
    KB_DATABASE_PATH,
    KB_RELATION_PATH,
    ENTITY,
    SYNONYM_KEY
)
from src.neo4j.inferencer import Inferencer

class EntitySearch():
    def __init__(self,
        database_path=KB_DATABASE_PATH, 
        relation_path=KB_RELATION_PATH, 
        model_dir=KB_DEFAULT_MODEL_DIR,
        data_dir=KB_DEFAULT_DATA_DIR):

        self.database = read_json(database_path)
        # self.relations = read_txt(relation_path)

        self.ner = BERTEntityExtractor(
            model_dir=model_dir, data_dir=data_dir)

        self.neo4j_inferencer = Inferencer()
        
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
        # bug here
        # entity_relation = self.to_entity_relation(entities)

        result = []
        index = 0
        
        for intent in entities['intent']:
            request = {
                'symptom' : entities['symptom'],
                'disease' : entities['disease'],
                'intent' : intent
            }
            prettier_answer = self.neo4j_inferencer.query(request)  # list of str
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

        # for k, v in entity_relation.items():
        #     _, prettier_answer, score = self.get_entity_by_relation(k, v) # list
        #     highlight_terms = [vl["value"] for vl in v if vl["value"]]
        #     highlight_terms.append(k)
        #     result.append({
        #         "id": index,
        #         "score": score,
        #         "question": question,
        #         "answer_dislay" : prettier_answer,
        #         "highlight": {
        #             "question": [self.get_highlight(question, highlight_terms)],
        #             "answer_dislay": [self.get_highlight(prettier_answer, highlight_terms)],
        #         },
        #     })
        #     index += 1

        return result

    def get_highlight(self, text: Text, terms: List):
        highlight = text
        for t in terms:
            t = f" {t} "
            highlight = highlight.replace(t, f"<em>{t}</em>")
        return highlight

    def to_entity_relation(self, entities: List):
        entity_relation = {}
        diseases = [e["value"] for e in entities if e["key"] == ENTITY]
        if diseases:
            for e in entities:
                if e["key"] != ENTITY:
                    for d in diseases:
                        if not entity_relation.get(d):
                            entity_relation[d] = [e]
                        else:
                            entity_relation[d].append(e)
        return entity_relation



    def get_prettier_answer(self, answer, relation):
        ''' Format/Prettier answer
        Args:
            - answer (list of str)
        Result:
            - result (str)
        '''
        result = []
        if answer != []:
            result = f"<br>{relation.title()}:<br>" + "<br>".join(answer)

        return result

    def get_entity_by_relation(self, entity, relation):
        ''' Get answer by given (entity,relation)
            Get early first
        Args:
            - entity (str) : 
            - relation (list of relation) :  
        Return:
            - result (str)
            - prettier_answer (str)
            - kb_response (list)
        '''
        answers = []
        prettier_answer = []
        scores = []

        for rel in relation:
            val, score = self.query_single_entity(entity, rel["key"])
            answer = "\n".join(val)
            answer_display = self.get_prettier_answer(val, rel["value"])
            answers.append(answer)
            scores.append(score)
            
            if answer_display != []:
                prettier_answer.append(answer_display)

        answers = '.'.join(answers)
        prettier_answer = ''.join(prettier_answer)

        return answers, prettier_answer, scores

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
    
    def reranking(self,kb_answer,results,entity):
        ''' Reranking the extracted answers from KB
            Get first
        Args: 
            - kb_answer (list) : [('U não', 'treatment', 100), ('U màng não', 'treatment', 100)]
            - results (list) : 
            - entity (str)
        Return:
            - result ()
        '''

        max_score = 0
        reranked_kb_result = []
        result = []

        if len(results) >= 1:
            for (triplet_sample, response_text) in zip(kb_answer,results):
                disease = triplet_sample[0]
                fuzz_score = get_fuzzy_score(disease,entity)
                triplet_sample[-1] = fuzz_score

                if fuzz_score > max_score:
                    max_score = fuzz_score
                    result = response_text
                    # append at the beginning
                    reranked_kb_result = [triplet_sample,*reranked_kb_result] 
                else:
                    reranked_kb_result.append(triplet_sample)
        else:
            result = results
            reranked_kb_result = kb_answer

        return result, reranked_kb_result