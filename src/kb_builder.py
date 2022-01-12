from constants import DISEASE_DATA_URL,SYMPTOM_DATA_URL
from utils import load_json, write_json
from kb_builder import KnowledgeBaseBuilde

class KnowledgeBaseBuilde:
    def __init__(self,disease_url=DISEASE_DATA_URL,symptom_url=SYMPTOM_DATA_URL):
        self.disease_data = load_json(DISEASE_DATA_URL)
        self.symptom_data = load_json(SYMPTOM_DATA_URL)
    
    def build_symptom_relation_database(self):
        ''' From disease database, build:
        1. Symptom database
        2. Relation database
        '''

        for sample in self.disease_data:
            # get symptom


    def merge_database(self):
        pass

    def get_statistics(self):
        pass