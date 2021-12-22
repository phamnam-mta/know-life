import os
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
        answers = []
        answer_dislay = []
        kb_answer = []
        
        ner_response = self.ner.inference(question)

        for k, v in ner_response.items():
            answer, prettier_answer, kb_response = self.get_entity_by_relation(k, v) # list
            answers.append(answer)
            kb_answer.append(kb_response)
            answer_dislay.append(prettier_answer)

        result = {
            "answers": answers,
            "ner_response": ner_response,
            "answer_dislay" : answer_dislay,
            "kb_response" : kb_answer
        }

        return result

    def get_prettier_answer(self, answer, max_answer_length=MAX_ANSWER_LENGTH):
        ''' Format/Prettier answer
        Args:
            - answer (list of str)
        Result:
            - result (str)
        '''
        result = []

        # if end-half contains ":"
        # if ':' in answer[0][len(answer[0])//2:]:
        #     result = answer[0] + ' ' + '<br>'.join(answer[1:]) 
        # else:

        if answer != []:
            result = '<br>'.join(answer)

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
        result = []
        prettier_answer = []
        kb_response = []

        for rel in relation:
            val, kb_answer = self.query_single_entity(entity, rel)
            pret_answer = self.get_prettier_answer(val)
            prettier_answer.append(pret_answer)
            result.extend(val)
            kb_response.append(kb_answer)

        prettier_answer = '.'.join(prettier_answer)
        result = '.'.join(result)

        return result , prettier_answer, kb_response

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
            is_relevant, score =  is_relevant_string(sample['disease'],entity,method=['exact','fuzzy','include'],return_score=True)
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