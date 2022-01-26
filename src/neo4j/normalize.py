import string

from src.utils.constants import (
    VERIFY_INTENT,
    DIAGNOSIS_INTENT,
    INFO_INTENT,
    INTENT_MAPPER
)

class Normalizer:
    def __init__(self):
        pass
    
    def __call__(self,neo4joutput,symptom,disease,intent):
        ''' Normalize Neo4jProvider output into string
        Args:
            - neo4joutput (list of str/dict) :
            - symptom (list of str)
            - disease (list of str)
            - intent (str) 
        Return:
            - result (str)
        '''
        result = ''

        if intent in INFO_INTENT:
            result = self.normalize_info_intent(neo4joutput)

        if intent in DIAGNOSIS_INTENT:
            result = self.normalize_diagnosis_intent(neo4joutput,symptom)

        if intent in VERIFY_INTENT:
            result = self.normalize_verify_intent(neo4joutput,symptom,disease)

        return result 

    def normalize_info_intent(self, neo4joutput):
        '''
        Args:
            - neo4joutput (list of str)
        '''
        result = '<br>-----------------------------------------------------------<br>'.join(neo4joutput)
        return result

    def normalize_diagnosis_intent(self, neo4joutput, symptom):
        '''
        Args:
            - neo4joutput (list of dict)
            - symptom (list of str)
        '''
        potential_disease = []
        for obj in neo4joutput:
            potential_disease.append(f"- {obj['name']} , tỉ lệ mắc : {obj['ratio']}")
        result = f'''Dựa theo các triệu chứng: {','.join(symptom)},
                Bạn có thể mắc các bệnh sau đây : {"<br>".join(potential_disease)}
                '''
        return result
    
    def normalize_verify_intent(self, neo4joutput, symptom, disease):
        '''
        Args:
            - neo4joutput (list of dict)
            - symptom (list of str)
            - disease (list of str)
        '''
        related_symptom = []
        not_related_symptom = []
        
        for list_of_obj in neo4joutput:
            for obj in list_of_obj:
                if obj['relation'] == True:
                    related_symptom.append(obj['symptom'])
                else:
                    not_related_symptom.append(obj['symptom'])
        result = f'''
                Các triệu chứng liên quan {disease}: {','.join(related_symptom)}
                Các triệu chứng KHÔNG liên quan {disease}: {','.join(not_related_symptom)}
                '''
        return result