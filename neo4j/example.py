import logging
import json

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
    # See https://neo4j.com/developer/aura-connect-driver/ for Aura specific connection URL.
    scheme = "bolt"  # Connecting to Aura, use the "neo4j+s" URI scheme
    host_name = "localhost"
    port = 7687
    url = "{scheme}://{host_name}:{port}".format(scheme=scheme, host_name=host_name, port=port)
    user = "neo4j"
    password = "password"
    
    app = KnowledgeGraph(url, user, password)

    DATA = read_json('../data/data.json')

    # create disease
    # app.build_database(DATA)

    response = app.query('ung gan',['cause','symptom'])
    
    for att_res in response:
        print(att_res)

    app.close()