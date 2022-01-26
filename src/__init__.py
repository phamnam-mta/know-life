import os
import logging

logging.getLogger(__name__).addHandler(logging.StreamHandler())
logging.getLogger(__name__).setLevel(logging.INFO)
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", None)