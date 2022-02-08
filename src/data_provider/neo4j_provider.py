''' From NLU output to Cypher code

intent: 
- overview, cause, symp, risk_factor, treatment, prevention, severity
- diag
- verify
'''
import itertools
from typing import Text

from src.data_provider import NEO4J_AUTH, NEO4J_URL
from src.utils.constants import (
    NEO4J_THRESHOLD,
    VERIFY_INTENT,
    DIAGNOSIS_INTENT,
    INFO_INTENT,
    INTENT_MAPPER
)
from src.data_provider.normalize import Normalizer
from py2neo import Graph, Node


class Neo4jProvider():
    def __init__(self, uri: Text = None, user: Text = None, password: Text = None):
        if not uri:
            uri = NEO4J_URL
            user = NEO4J_AUTH.split("/")[0]
            password = NEO4J_AUTH.split("/")[1]
        self.graph = Graph(uri, auth=(user, password))
        self.normalizer = Normalizer()

    def query(self, request):
        ''' Cypher code to get data from graph database
        Args:
            - request {
                - symptom (list)
                - disease (list)
                - intent (str)
            }
        Return:
            - result (str)
        '''
        intent = request['intent']
        symptom = request['symptom']
        disease = request['disease']
        disease = self.normalizer.normalize_disease(disease)
        
        result = []

        if intent in VERIFY_INTENT:
            neo4j_intent = INTENT_MAPPER[intent]
            result = self.get_answer_verify(symptom, disease)

        if intent in DIAGNOSIS_INTENT:
            # neo4j_intent = INTENT_MAPPER[intent]
            result = self.get_answer_diag(symptom)
            result = self.reranking_diag(result)

        if intent in INFO_INTENT:
            neo4j_intent = INTENT_MAPPER[intent]
            result = self.get_answer_info(disease, neo4j_intent)

        # filter None element in list
        result = list(filter(None, result))
        result = self.normalizer(result, symptom, disease, intent)

        return result

    def get_answer_verify(self, symptom, disease):
        result = []
        for s_ in symptom:
            for d_ in disease:
                query = f"""
                MATCH (d:Disease)
                WHERE apoc.text.sorensenDiceSimilarity(d.name, "{d_}") >=  {NEO4J_THRESHOLD}
                MATCH (s:Symptom) 
                WHERE apoc.text.sorensenDiceSimilarity(s.name, "{s_}") >=  {NEO4J_THRESHOLD}
                RETURN d.name as disease, s.name as symptom, EXISTS( (d)-[:HAS_SYMPTOM]->(s) ) as relation
                """

            ans = self.graph.run(query)

            ans = ans.data()  # list of dict
            result.append(ans)

        return result

    def get_num_rels(self, disease):
        ''' Get disease - number of relations 
        '''
        query = f'''
        MATCH (d:Disease)-[HAS_SYMPTOM]->(s:Symptom) 
        WITH d,count(s) as rels, collect(s) as symptoms
        WHERE rels > 1 and d.name in ["{disease}"]
        RETURN d.name as disease, rels 
        '''
        result = self.graph.run(query).data()
        return result

    def get_answer_diag(self, symptom):
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
            disease_rels = self.get_num_rels(res['name'])
            if disease_rels:
                res['disease_rels'] = disease_rels[0].get("rels")
                res['ratio'] = res['num_symptom'] / res['disease_rels']
            else:
                res['ratio'] = 0.001

        return result

    def get_answer_info(self, entity, intent):
        result = []
        for ent in entity:
            query = f"""
            MATCH (a:Disease)
            WHERE apoc.text.sorensenDiceSimilarity(a.name, "{ent.replace('bệnh', '')}") >=  {NEO4J_THRESHOLD}
            RETURN a.{intent} as result
            """

            ans = self.graph.run(query).data()

            for a in ans:
                result.append(a['result'])

        return result

    def reranking_diag(self, response, topk=10):
        ''' Reranking/Normalize the output 
        Args:
            - response (list of dict) : [{'name': 'Chấn thương sọ não', 'num_symptom': 3, 'disease_rels': 20, 'ratio': 0.15}]
                - name (str) : disease name
                - num_symptom (str) : overlap symptom
                - disease_rels (int) : # symptoms that disease totally has
                - ratio (float) : num_symptom/disease_rels
        Return:
            - result (list of dict)
        '''
        # sorted from highest to lowest
        result = sorted(response, key=lambda d: -d['ratio'])
        result = result[:topk]

        return result


if __name__ == '__main__':
    user = NEO4J_AUTH.split("/")[0]
    password = NEO4J_AUTH.split("/")[1]
    p = Neo4jProvider(NEO4J_URL, user, password)

    request = {
        'symptom': ['phân có máu', 'sốt', 'chóng mặt', 'buồn nôn', 'đau ngực'],
        'disease': ['trĩ ngoại'],
        'intent': 'diagnosis'
    }
    answer = p.query(request)
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
