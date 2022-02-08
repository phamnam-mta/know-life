import os
import pandas as pd
import logging
import json
from py2neo import Graph,Node
from typing import Text, List

WORK_DIR = os.path.abspath(os.getcwd())
DISEASE_PATH = os.path.join(WORK_DIR, "data/kb/diseases.csv")
SPECIALTY_PATH = os.path.join(WORK_DIR, "data/kb/specialties.csv")
SYMPTOM_PATH = os.path.join(WORK_DIR, "data/kb/symptoms.csv")
DISEASE_SYMPTOM_PATH = os.path.join(WORK_DIR, "data/kb/diseases_has_symptoms.csv")
DISEASE_SPECIALTY_PATH = os.path.join(WORK_DIR, "data/kb/diseases_health_specialties.csv")

class KnowledgeGraph():
    def __init__(self,
                uri: Text = None,
                user: Text = None,
                password: Text = None,
                disease_path: Text = DISEASE_PATH,
                symptom_path: Text = SYMPTOM_PATH,
                specialty_path: Text = SPECIALTY_PATH,
                disease_symptom_path: Text = DISEASE_SYMPTOM_PATH,
                disease_specialty_path: Text = DISEASE_SPECIALTY_PATH):
        
        # node
        self.disease = pd.read_csv(disease_path)
        self.symptom = pd.read_csv(symptom_path)
        self.specialty = pd.read_csv(specialty_path)
        
        # relationship
        self.disease_symptom_rel = pd.read_csv(disease_symptom_path)
        self.disease_specialty_rel = pd.read_csv(disease_specialty_path)
        
        self.graph = Graph(uri, auth=(user, password))

    def read_nodes(self):
        DISEASE = []
        SYMPTOM = []
        SPECIALTY = []
        
        DISEASE_SYMPTOM = []
        DISEASE_SPECIALTY = []
        
        # disease - node
        for index, row in self.disease.iterrows():
            DISEASE.append({
                "id":row['id'],
                "name":row['name'],
                "overview":row['overview'],
                "cause":row['cause'],
                "symptom":row['symptom'],
                "risk_factor":row['risk_factor'],
                "treatment":row['treatment'],
                "diagnosis":row['diagnosis'],
                "prevention":row['prevention'],
                "severity":row['severity'],
                "synonym":row['synonym']
            })
        
        # symptom - node
        for index, row in self.symptom.iterrows():
            SYMPTOM.append({
                "id":row['id'],
                "name":row['name'],
                "overview":row['overview'],
            })
            
        # specialty - node
        for index, row in self.specialty.iterrows():
            SPECIALTY.append({
                "id":row['id'],
                "name":row['name'],
                "description":row['description'],
            })
        
        # disease-symptom rels
        for index, row in self.disease_symptom_rel.iterrows():
            DISEASE_SYMPTOM.append({
                "disease_id":row['disease_id'],
                "symptom_id":row['symptom_id']
            })
        
        # disease-specialty rels
        for index, row in self.disease_specialty_rel.iterrows():
            DISEASE_SPECIALTY.append({
                "disease_id":row['disease_id'],
                "specialty_id":row['specialty_id']
            })      
            
        return DISEASE, SYMPTOM, SPECIALTY, DISEASE_SYMPTOM, DISEASE_SPECIALTY
    
    def remove_nodes(self):
        print("Remove all nodes,  relationships")
        cypher = 'MATCH (n) DETACH DELETE n'
        self.graph.run(cypher)
        
    def create_node(self):
        self.remove_nodes()

        DISEASE, SYMPTOM, SPECIALTY, DISEASE_SYMPTOM, DISEASE_SPECIALTY = self.read_nodes()
        
        # disease
        print("Create disease node")
        for node in DISEASE:
            cypher_ = "CREATE (d:Disease $props) RETURN d"
            self.graph.run(cypher_,props=node)
        
        #symptom
        print("Create symptom node")
        for node in SYMPTOM:
            cypher_ = "CREATE (d:Symptom $props) RETURN d"
            self.graph.run(cypher_,props=node)
        
        # specialty
        print("Create specialty node")
        for node in SPECIALTY:
            cypher_ = "CREATE (d:Specialty $props) RETURN d"
            self.graph.run(cypher_,props=node)
        
        # disease-symptom rels
        print("Create disease-symptom relation")
        for rel in DISEASE_SYMPTOM:
            disease_id = rel['disease_id']
            symptom_id = rel['symptom_id']
            cypher_ = f'''
            MATCH (d:Disease), (s:Symptom)
            WHERE d.id = {disease_id} AND s.id = {symptom_id}
            CREATE (d)-[r:HAS_SYMPTOM]->(s)
            RETURN type(r)
            '''
            self.graph.run(cypher_)

        # disease-specialty rels
        print("Create disease-specialty relation")
        for rel in DISEASE_SPECIALTY:
            disease_id = rel['disease_id']
            symptom_id = rel['specialty_id']
            cypher_ = f'''
            MATCH (d:Disease), (s:Specialty)
            WHERE d.id = {disease_id} AND s.id = {symptom_id}
            CREATE (d)-[r:HAS_SPECIALTY]->(s)
            RETURN type(r)
            '''
            self.graph.run(cypher_)

        return

    def create_rels(self):
        for node in self.disease_symptom_rel:
            disease_name = node['from']['name']
            symptoms = []
            for s in node['to']:
                symptoms.append(s['name'])
            for symptom_name in symptoms:
                query = '''
                MATCH (d:Disease {name:$disease_name})
                MATCH (s:Symptom {name:$symptom_name})
                CREATE (d)-[:HAS_SYMPTOM]->(s)
                RETURN d,s
                '''
                self.graph.run(query,parameters={"disease_name":disease_name,"symptom_name":symptom_name})
        return

if __name__ == "__main__":
    NEO4J_URL = os.getenv("NEO4J_URL", None)
    NEO4J_AUTH = os.getenv("NEO4J_AUTH", None)
    user = NEO4J_AUTH.split("/")[0]
    password = NEO4J_AUTH.split("/")[1]
    
    kg = KnowledgeGraph(NEO4J_URL, user, password)

    kg.create_node()
    #kg.create_rels()