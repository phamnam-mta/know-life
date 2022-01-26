# elasticsearch
QA_INDEX='qa'
QA_QUERY_FIELDS = [ "question", "answer", "answer_display" ]
KB_INDEX='kb'

# entity search
MAX_ANSWER_LENGTH = 300
KB_DEFAULT_MODEL_DIR = "models/bert_extractor"
KB_DEFAULT_DATA_DIR = "data/kb"
NOT_FOUND_ENTITY = "KB chưa cập nhật quan hệ này"
TEST_DIR = "data/kb/testcases.csv"
ENTITY = "disease"
SYNONYM_KEY = "synonym"

# semantic search
from enum import Enum
class ResponseAttribute(Enum):
    ALL = 1
    ANSWER = 2
class SearchMethod(Enum):
    ES = "elastic"
    SM = "semantic"
    EE = "entity"
QA_MODEL_DIR = "models/ranking"

# server
DEFAULT_REQUEST_TIMEOUT = 60 * 5  # 5 minutes
DEFAULT_RESPONSE_TIMEOUT = 60 * 60  # 1 hour
DEFAULT_SERVER_PORT = 5000
DEFAULT_ENCODING= "utf-8"
TCP_PROTOCOL = "TCP"
DEFAULT_SANIC_WORKERS = 1
ENV_SANIC_WORKERS = "SANIC_WORKERS"
ENV_SANIC_BACKLOG = "SANIC_BACKLOG"
DEFAULT_LOG_LEVEL_LIBRARIES = "ERROR"
ENV_LOG_LEVEL_LIBRARIES = "LOG_LEVEL_LIBRARIES"
DEFAULT_SERVER_INTERFACE = "0.0.0.0"

# neo4j
NEO4J_THRESHOLD = 0.8
DIAGNOSIS_INTENT = ['diagnosis']
VERIFY_INTENT = ['verify']
INFO_INTENT = ['queryentity','queryattribute.attr_cause','queryattribute.attr_symp','queryattribute.attr_rf',\
    'queryattribute.attr_treat','queryattribute.attr_prev','queryattribute.attr_sev','advise']

INTENT_MAPPER= {
    'diagnosis':"diagnosis",
    'verify':"verify",
    'queryentity':"overview",
    'queryattribute.attr_cause':"cause",
    'queryattribute.attr_symp':"symptom",
    'queryattribute.attr_treat':"treatment",
    'queryattribute.attr_prev':"prevention",
    'queryattribute.attr_sev':"severity",
    'advise':"advise",
}