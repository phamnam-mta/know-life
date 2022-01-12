ATTRIBUTES_CHECKLIST = [
    "synonym",
    "overview",
    "cause",
    "symptom",
    "riskfactor",
    "prevention",
    "diagnosis",
    "treatment",
    "infection",
    "department_key"
]
TOP_K_MISSING = 10
DISEASE_DATA_URL = '../data/kb/raw/vinmec_data.json'
SYMPTOM_DATA_WIKI_URL = '../data/kb/processed/symtom_data_wiki.json' # from wiki
SYMPTOM_DATA_URL = '../data/kb/processed/symtom_data.json' # from msd manuals
WORD2VEC_URL = '../data/w2v/word2vec_vi_syllables_100dims.txt'