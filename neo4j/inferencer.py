''' From NLU output to Cypher code

intent: 
- overview, cause, symp, risk_factor, treatment, prevention, severity
- diag
- verify
'''
import itertools

from constants import *
from py2neo import Graph,Node
THRESHOLD = 0.55

class Inferencer:
    def __init__(self):
<<<<<<< HEAD
        self.graph = Graph("bolt://localhost:7687", auth=("neo4j", "knowlife"))
=======
        self.graph = Graph("bolt://localhost:7687", auth=("neo4j", "password"))
>>>>>>> d9d9aea93645617d261aea715e35b97d4e3cde64

    def query(self,request):
        '''
        Args:
            - request {
                - symptom (list)
                - disease (list)
                - intent (str)
            }
        Return:
            - result (list of str)
        '''
        intent = request['intent']
        symptom = request['symptom']
        disease = request['disease']

        if intent in VERIFY_INTENT:
            result = self.get_answer_verify(symptom,disease)
            
<<<<<<< HEAD
        if intent in DIAGNOSIS_INTENT:
=======
        if intent in DIAGONIS_INTENT:
>>>>>>> d9d9aea93645617d261aea715e35b97d4e3cde64
            result = self.get_answer_diag(symptom)
            
        if intent in INFO_INTENT:
            result = self.get_answer_info(disease,intent)

        return result

    def get_answer_verify(self,symptom,disease):
        result = []
        THRESHOLD = 0.55
        for s_ in symptom:
            for d_ in disease:
                query = f"""
                MATCH (d:Disease)
                WHERE apoc.text.sorensenDiceSimilarity(d.name, "{d_}") >=  {THRESHOLD}
                MATCH (s:Symptom) 
                WHERE apoc.text.sorensenDiceSimilarity(s.name, "{s_}") >=  {THRESHOLD}
                RETURN d.name as disease, s.name as symptom, EXISTS( (d)-[:HAS_SYMPTOM]->(s) ) as relation
                """

            ans = self.graph.run(query)
          
            ans = ans.data() # list of dict
            result.append(ans)

        return result

    def get_num_rels(self,disease):
        ''' Get disease - number of relations 
        '''
        query = f'''
        MATCH (d:Disease)-[HAS_SYMPTOM]->(s:Symptom) 
        WITH d,count(s) as rels, collect(s) as symptoms
        WHERE rels > 1 and d.name in ["{disease}"]
        RETURN d.name as disease, rels 
        '''
        result = self.graph.run(query)

        return result.data()[0]['rels']

    def get_answer_diag(self,symptom):
        '''
        '''
        # uppercase 1st letter
        symptom = [item.capitalize() for item in symptom]

        query = f'''
        match (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
        where s.name IN {symptom}
        return d.name as name, COUNT(d) as num_symptom
        ORDER BY COUNT(d) DESC LIMIT 15
        '''
        ans = self.graph.run(query)
        result = ans.data()
        for res in result:
            res['disease_rels'] = self.get_num_rels(res['name'])
            res['ratio'] = res['num_symptom'] / res['disease_rels']
        
        return result

    def get_answer_info(self,entity,intent):
        result = []
        for ent in entity:
            query = f"""
            MATCH (a:Disease)
            WHERE apoc.text.sorensenDiceSimilarity(a.name, "{ent}") >=  {THRESHOLD}
            RETURN a.{intent} as result
            """

            ans = self.graph.run(query).data()
            
            for a in ans:
                result.append(a['result'])

        return result

if __name__ == '__main__':
    inferencer = Inferencer()

    request = {
        'symptom': ['phân có máu','sốt','chóng mặt','buồn nôn','đau ngực'],
        'disease': ['trĩ ngoại'],
        'intent' : 'verify'
    }
    answer = inferencer.query(request)
    print(answer)
    '''
    [{'name': 'Than', 'num_symptom': 3, 'disease_rels': 13, 'ratio': 0.23076923076923078}, 
    {'name': 'Lỵ trực trùng', 'num_symptom': 3, 'disease_rels': 11, 'ratio': 0.2727272727272727}, 
    {'name': 'Tiêu chảy cấp', 'num_symptom': 3, 'disease_rels': 12, 'ratio': 0.25}, 
    {'name': 'Hội chứng Mittelschmerz', 'num_symptom': 3, 'disease_rels': 10, 'ratio': 0.3}, 
    {'name': 'Ung thư lá lách', 'num_symptom': 2, 'disease_rels': 11, 'ratio': 0.18181818181818182}, 
    {'name': 'Máu nhiễm mỡ', 'num_symptom': 2, 'disease_rels': 9, 'ratio': 0.2222222222222222}, 
    {'name': 'Viêm đại tràng', 'num_symptom': 2, 'disease_rels': 9, 'ratio': 0.2222222222222222}, 
    {'name': 'Barrett thực quản', 'num_symptom': 2, 'disease_rels': 5, 'ratio': 0.4}, 
    {'name': 'Amip ăn não', 'num_symptom': 2, 'disease_rels': 12, 'ratio': 0.16666666666666666}, 
    {'name': 'Tăng thông khí', 'num_symptom': 2, 'disease_rels': 8, 'ratio': 0.25}, 
    {'name': 'Thấp tim ở trẻ em', 'num_symptom': 2, 'disease_rels': 4, 'ratio': 0.5}, 
    {'name': 'Loét thực quản', 'num_symptom': 2, 'disease_rels': 7, 'ratio': 0.2857142857142857}, {
    'name': 'Lao ruột', 'num_symptom': 2, 'disease_rels': 12, 'ratio': 0.16666666666666666}, 
    '''