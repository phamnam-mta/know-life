import string

from src.utils.constants import (
    VERIFY_INTENT,
    DIAGONIS_INTENT,
    INFO_INTENT,
    INTENT_MAPPER
)

class Normalizer:
    def __init__(self):
        pass
    
    def __call__(self,neo4joutput,intent):
        ''' Normalize Neo4jProvider output into string
        Args:
            - neo4joutput (list of str/dict) :
            - intent (str) 
        Return:
            - result (str)
        '''
        result = ''

        if intent in INFO_INTENT:
            result = self.normalize_info_intent(neo4joutput)

        if intent in DIAGONIS_INTENT:
            result = self.normalize_diagnosis_intent(neo4joutput)

        if intent in VERIFY_INTENT:
            result = self.normalize_verify_intent(neo4joutput)

        return result 

    def normalize_info_intent(self, neo4joutput):
        '''
        Args:
            - neo4joutput (list of str)
        '''
        result = '<br>-----------------------------------------------------------<br>'.join(neo4joutput)
        return result

    def normalize_diagnosis_intent(self, neo4joutput):
        '''
        Args:
            - neo4joutput (list of dict)
        '''
        print(neo4joutput)
        exit()
        template = '''
        Dựa theo các triệu chứng: ,
        Bạn có thể mắc các bệnh sau đây : 
        '''
        return result
    
    def normalize_verify_intent(self, neo4joutput):
        '''
        Args:
            - neo4joutput (list of dict)
        '''
        return result