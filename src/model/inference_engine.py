import os
import warnings
from tqdm import tqdm

# from model.utils import *
from utils import *

class InferenceEngine:
    def __init__(self, database_dir="../data", max_answer_length=300):

        self.database = read_json(os.path.join(database_dir,'data.json'))
        self.relations = read_txt(os.path.join(database_dir,'relations.txt'))

        self.max_answer_length = max_answer_length
        self.accept_redundant_length = max_answer_length // 3

    def search(self,entities,relations):
        ''' 
        Args:
            - entities (list of str) :
            - relations (list of str) :
            assert len(entities) == len(relations)
        Return:
            - result (list) : top-k answer
        '''
        result = []
        is_safe = True

        if len(entities) == len(relations):
            warnings.warn("Entities and Relations does not have same length !")
            is_safe = False

        # is_safe == False TODO
        for ent,rel in zip(entities,relations):
            answer = self.get_entity_by_relation(ent,rel)
            answer = self.get_prettier_answer(answer)
            result.append(answer)

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
                break
            result.append(sentence)

        result = ' . '.join(result)

        return result

    def get_entity_by_relation(self,entity,relation):
        ''' Get answer by given (entity,relation)
            Get early first
        Args:
            - entity (str) : 
            - relation (str) : 
        Return:
            - result (list of str)
        '''
        result = []
        is_correct_relation = True

        if relation not in self.relations:
            warnings.warn("Relations does not fit relation datatabase. Activate fuzzy match for relation .")
            is_correct_relation = False

        bag_of_result = []
        for sample in tqdm(self.database):
            # compare fuzzy
            disease_synonyms = sample['synonyms'] # list
            for ds in disease_synonyms:
                if is_relevant_string(entity,ds,method=['exact','fuzzy']):
                    if is_correct_relation:
                        result = sample[relation]
                        return result
                    else:
                        for rel , val in sample.items():
                            if rel in ['disease','key']:
                                continue
                                
                            if is_relevant_string(rel,relation,method=['include']):
                                result = val
                                bag_of_result.append((ds,rel,val))
                    break
                
        # Re-ranking based on disease: TODO
        
        if len(bag_of_result) > 1:
            result = bag_of_result[0][-1] # val
            print(bag_of_result[0][0],bag_of_result[0][1])
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
