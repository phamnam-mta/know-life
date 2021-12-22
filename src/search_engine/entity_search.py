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

        # self.ner = BERTEntityExtractor(
        #     model_dir=model_dir, data_dir=data_dir)

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
        
        ner_response = self.ner.inference(question)

        for k, v in ner_response.items():
            answer = self.get_entity_by_relation(k, v) # list
            answers.append(answer)

            prettier_answer = self.get_prettier_answer(answer)
            answer_dislay.append(prettier_answer)

        result = {
            "answers": answers,
            "ner_response": ner_response,
            "answer_dislay" : answer_dislay
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
        result = '<br>'.join(answer)

        return result

    def get_entity_by_relation(self, entity, relation):
        ''' Get answer by given (entity,relation)
            Get early first
        Args:
            - entity (str) : 
            - relation (list of relation) :  
        Return:
            - result (list of str)
        '''
        result = []
        is_correct_relation = True

        # if relation not in self.relations:
        #     warnings.warn("Relations does not fit relation datatabase. Activate fuzzy match for relation .")
        #     is_correct_relation = False

        for rel in relation:
            val = self.query_single_entity(entity, rel)
            val = self.get_prettier_answer(val)

            result.append(val)

        return result

    def query_single_entity(self, entity, relation):
        results = []
        scores = []

        result = ""
        
        for sample in self.database:
            is_relevant, score =  is_relevant_string(sample['disease'],entity,method=['exact','fuzzy'],return_score=True)
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
        # Re-ranking
        if len(results) >= 1:
            max_score_index = scores.index(max(scores))

            result = results[max_score_index]

        return result