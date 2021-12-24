import os
from typing import List, Text
import warnings

from src.utils.io import *
from src.utils.kb_utils import *
from src.nlu import BERTEntityExtractor
from src.utils.kb_utils import is_relevant_string
from src.utils.constants import (
    MAX_ANSWER_LENGTH,
    KB_DEFAULT_MODEL_DIR,
    KB_DEFAULT_DATA_DIR,
    KB_DATABASE_PATH,
    KB_RELATION_PATH,
    ENTITY,
    SYNONYM_KEY
)


class EntitySearch():
    def __init__(self, 
        database_path=KB_DATABASE_PATH, 
        relation_path=KB_RELATION_PATH, 
        model_dir=KB_DEFAULT_MODEL_DIR,
        data_dir=KB_DEFAULT_DATA_DIR):

        self.database = read_json(database_path)
        self.relations = read_txt(relation_path)

        self.ner = BERTEntityExtractor(
            model_dir=model_dir, data_dir=data_dir)

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
        entity_relation = self.to_entity_relation(entities)
        
        result = []
        index = 0
        for k, v in entity_relation.items():
            _, prettier_answer, score = self.get_entity_by_relation(k, v) # list
            highlight_terms = [vl["value"] for vl in v if vl["value"]]
            highlight_terms.append(k)
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
            prettier_answer.append(answer_display)
            scores.append(score)

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
                        kb_answer.append((sample['disease'],att['attribute'],score))
                        
        # Re-ranking
        if len(results) >= 1:
            THRESHOLD = 80
            
            score_index = [i[0] for i in sorted(enumerate(scores), key=lambda x:-x[1])]
            
            result = [results[i] for i in score_index]

            if scores[score_index[0]] >= THRESHOLD:
                result = result[0]
        
        kb_answer = sorted(kb_answer, key=lambda x: -x[2])

        return result, kb_answer