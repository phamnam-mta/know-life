import os
import warnings
from tqdm import tqdm

# from model.utils import *
from src.utils import *
from src.ner_model.inference import Inference

class InferenceEngine:
    def __init__(self, database_dir="../data", model_dir='./ckpt', max_answer_length=300):

        self.database = read_json(os.path.join(database_dir,'data.json'))
        self.relations = read_txt(os.path.join(database_dir,'relations.txt'))

        self.max_answer_length = max_answer_length
        self.accept_redundant_length = max_answer_length // 3

        self.ner = Inference(path_args=model_dir)

    def query(self, question):
        '''
        Args:
            - quesion (str) : utterance

        '''
        result = []
        
        # - ner_response (dict) : {
        #     'disease' : ['relations']
        # }
        ner_response = self.ner.inference(question)
        # print(ner_response)

        # is_safe == False TODO
        # for resp in ner_response:
        #     for k,v in resp.items():
        #         answer = self.get_entity_by_relation(k,v)
        #         result.append(answer)
        for k,v in ner_response.items():
            answer = self.get_entity_by_relation(k,v)
            result.append(answer)
        # print(f'results : {answer}')

        return result
    
    def get_prettier_answer(self,answer):
        ''' Format/Prettier answer
        Args:
            - answer (list of str)
        Result:
            - result (str)
        '''
        result = []
        length_cnt = 0
        for sentence in answer:
            length_cnt += len(sentence)
            if length_cnt > self.max_answer_length + self.accept_redundant_length:
                if result == []:
                    result.append(sentence)
                break
            result.append(sentence)

        result = ' . '.join(result)

        return result

    def get_entity_by_relation(self,entity,relation):
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

        if relation not in self.relations:
            warnings.warn("Relations does not fit relation datatabase. Activate fuzzy match for relation .")
            is_correct_relation = False

        for rel in relation:
            val = self.query_single_entity(entity,rel)
            val = self.get_prettier_answer(val)

            result.append(val)
                
        # Re-ranking based on disease: TODO
        
        return result
    
    def query_single_entity(self,entity,relation):
        result = ""

        for sample in self.database:
            if sample['disease'] == entity:
                for att in sample['attributes']:
                    if att['attribute'] == relation:
                        # short content
                        try:
                            result = att['short_content']
                        except:
                            result = att['content']

                        return result
        return result            


if __name__ == '__main__':
    inference_engine = InferenceEngine()
    entites = ['thalas','máu nhiễm mỡ','rối loạn tiền đình']
    relations = ['description','symptoms','cause']
    for ent, rel in zip(entites,relations):
        answer = inference_engine.search([ent],[rel])
        print(f'Entity : {ent} , Relation : {rel}')
        print(answer)
        print('=================')
