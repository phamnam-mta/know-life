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

        '''
        answers = []

        # - ner_response (dict) : {
        #     'disease' : ['relations']
        # }
        ner_response = self.ner.inference(question)
        #print(ner_response)

        # is_safe == False TODO
        # for resp in ner_response:
        #     for k,v in resp.items():
        #         answer = self.get_entity_by_relation(k,v)
        #         answers.append(answer)
        for k, v in ner_response.items():
            answer = self.get_entity_by_relation(k, v)
            answers.append(answer)
        # print(f'answers : {answers}')

        result = {
            "answers": answers,
            "ner_response": ner_response
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
        length_cnt = 0
        accept_redundant_length = max_answer_length // 3
        for sentence in answer:
            length_cnt += len(sentence)
            if length_cnt > max_answer_length + accept_redundant_length:
                if result == []:
                    result.append(sentence)
                break
            result.append(sentence)

        result = ' . '.join(result)

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

        # Re-ranking based on disease: TODO

        return result

    def query_single_entity(self, entity, relation):
        result = ""

        for sample in self.database:
            if is_relevant_string(sample['disease'],entity,method=['exact','fuzzy','include']):
                for att in sample['attributes']:
                    if att['attribute'] == relation:
                        # short content
                        try:
                            result = att['short_content']
                        except:
                            result = att['content']

                        return result
        return result

