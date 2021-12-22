import logging
import json

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
from py2neo import Graph
from cypher import CypherTranslator
from fuzzywuzzy import fuzz

def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

class App:

    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.cypher_translator = CypherTranslator()

    def close(self):
        # Don't forget to close the driver connection when you are finished with it
        self.driver.close()

    def create_friendship(self, person1_name, person2_name):
        with self.driver.session() as session:
            # Write transactions allow the driver to handle retries and transient errors
            result = session.write_transaction(
                self._create_and_return_friendship, person1_name, person2_name)
            for record in result:
                print("Created friendship between: {p1}, {p2}".format(
                    p1=record['p1'], p2=record['p2']))

    @staticmethod
    def _create_and_return_friendship(tx, person1_name, person2_name):

        # To learn more about the Cypher syntax,
        # see https://neo4j.com/docs/cypher-manual/current/

        # The Reference Card is also a good resource for keywords,
        # see https://neo4j.com/docs/cypher-refcard/current/

        query = (
            "CREATE (p1:Person { name: $person1_name }) "
            "CREATE (p2:Person { name: $person2_name }) "
            "CREATE (p1)-[:KNOWS]->(p2) "
            "RETURN p1, p2"
        )
        result = tx.run(query, person1_name=person1_name, person2_name=person2_name)
        try:
            return [{"p1": record["p1"]["name"], "p2": record["p2"]["name"]}
                    for record in result]
        # Capture any errors along with the query and data for traceability
        except ServiceUnavailable as exception:
            logging.error("{query} raised an error: \n {exception}".format(
                query=query, exception=exception))
            raise

    def find_person(self, person_name):
        with self.driver.session() as session:
            result = session.read_transaction(self._find_and_return_person, person_name)
            for record in result:
                print("Found person: {record}".format(record=record))

    @staticmethod
    def _find_and_return_person(tx, person_name):
        query = (
            "MATCH (p:Person)"
            "WHERE p.name = $person_name "
            "RETURN p.name AS name"
        )
        result = tx.run(query, person_name=person_name)
        return [record["name"] for record in result]

    @staticmethod
    def _create_instance(tx,sample):

        # get all attributes
        attributes = []
        for att in sample["attributes"]:
            attributes.append(att['attribute'])

        query_str = "CREATE (d:Disease { name: $sample['disease'], url:$sample['url'], faq:$sample['faq']}) \n"
        
        for i,att in enumerate(attributes):
            q = f"SET d.{att} = {sample['attributes'][i]['content']} \n"
            query_str += q
        query = (
            f"""
            {query_str}
            RETURN d
            """
        )
        result = tx.run(query, sample=sample)

        try:
            return [{"d": record["d"]["name"]}
                    for record in result]
        # Capture any errors along with the query and data for traceability
        except ServiceUnavailable as exception:
            logging.error("{query} raised an error: \n {exception}".format(
                query=query, exception=exception))
            raise    
    
    def build_database(self,data):
        for sample in data:
            with self.driver.session() as session:
                # Write transactions allow the driver to handle retries and transient errors
                result = session.write_transaction(
                    self._create_instance, sample)
                for record in result:
                    print(f"Created disease: {record['d']}")
    
    @staticmethod
    def _find_and_return_disease(tx, disease_name):
        query = (
            "MATCH (p:Disease)"
            "WHERE p.name = $disease_name "
            "RETURN p.name AS name"
        )
        result = tx.run(query, disease_name=disease_name)
        return [record["name"] for record in result]


    @staticmethod
    def _query(tx, disease_name, attribute):
        THRESHOLD = 0.55

        query = f"""
        MATCH (a:Disease)
        WHERE apoc.text.sorensenDiceSimilarity(a.name, "{disease_name}") >=  {THRESHOLD}
        RETURN a.{attribute} as result
        """
        print(query)
        result = tx.run(query, disease_name=disease_name,attribute=attribute)
        return [record["result"] for record in result]

    def query(self,disease_name, attributes, mode='simple'):        
        result = []
        if mode == 'simple':
            for att in attributes:
                with self.driver.session() as session:
                    result = session.write_transaction(self._query,disease_name,att)   
                    
                    result.append(result)
        return result

if __name__ == "__main__":
    # See https://neo4j.com/developer/aura-connect-driver/ for Aura specific connection URL.
    scheme = "bolt"  # Connecting to Aura, use the "neo4j+s" URI scheme
    host_name = "localhost"
    port = 7687
    url = "{scheme}://{host_name}:{port}".format(scheme=scheme, host_name=host_name, port=port)
    print(url)
    user = "neo4j"
    password = "password"
    app = App(url, user, password)
    DATA = read_json('../data/data.json')

    # create disease
    # app.build_database(DATA)

    response = app.query('ung gan',['cause','symptom'])
    
    for att_res in response:
        print(att_res)

    app.close()