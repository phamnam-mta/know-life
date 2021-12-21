# elasticsearch
QA_INDEX='qa'
QA_QUERY_FIELDS = [ "question", "answer" ]
KB_INDEX='kb'

# entity search
MAX_ANSWER_LENGTH = 300
KB_DEFAULT_MODEL_DIR = "models/bert_extractor"
KB_DEFAULT_DATA_DIR = "data/kb"
KB_DATABASE_PATH = "data/kb/data.json"
KB_RELATION_PATH = "data/kb/relations.txt"


# semantic search
from enum import Enum
class ResponseAttribute(Enum):
    ALL = 1
    ANSWER = 2

QA_MODEL_DIR = "models/ranking"