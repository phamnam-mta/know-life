import json
import string
import logging

def remove_ref_tag(str):
    str = re.sub("(?<=\[)(.*?)(?=\])", "", str)
    str = str.replace('[]',' ')
    return str

def load_json(path):
    ''' Read a list of (dict)
    '''
    logging.info(f"Read {path} ...")
    with open(path) as f:
        data = json.load(f)
    return data

def write_json(data,path):
    ''' Write a list of (dict)
    '''
    logging.info(f"Save to {path}")
    with open(path, 'w',encoding='utf-8') as f:
        json.dump(data, f,ensure_ascii=False)