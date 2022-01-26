import logging
import json
import os

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
from py2neo import Graph
from fuzzywuzzy import fuzz
from knowledge_graph import KnowledgeGraph

def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

if __name__ == "__main__":
    NEO4J_URL = os.getenv("NEO4J_URL", None)
    NEO4J_AUTH = os.getenv("NEO4J_AUTH", None)
    user = NEO4J_AUTH.split("/")[0]
    password = NEO4J_AUTH.split("/")[1]
    
    app = KnowledgeGraph(NEO4J_URL, user, password)

    response = app.query('ung gan',['cause','symptom'])
    
    for att_res in response:
        print(att_res)

    app.close()