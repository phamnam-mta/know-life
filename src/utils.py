import json
import string
import logging
from fuzzywuzzy import fuzz

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

def is_relevant_string( str1,
                        str2,
                        remove_accent=False,
                        method=['exact','fuzzy','include'],
                        score=80,
                        return_score=False):
    '''
    1. Extract Match 
    2. Compare 2 str by fuzzy
    3. Included in str
    '''
    if remove_accent:
        str1 = remove_accents(str1)
        str2 = remove_accents(str2)

    # 1. extract match
    if 'exact' in method:
        if str1.lower() == str2.lower():
            if return_score:
                return True,100
            return True

    # 2. fuzzy match
    if 'fuzzy' in method:
        ratio = fuzz.ratio(str1.lower(),str2.lower())
        if ratio  > score:
            if return_score:
                return True, fuzz.ratio(str1.lower(),str2.lower())
            return True
    
    # 3. include match
    if 'include' in method:
        str1 = str1.lower().split(' ')
        str2 = str2.lower().split(' ')
        overlap_str = list(set(str2) & set(str1))
        if len(overlap_str) >= len(str1) // 2 + 1:
            if return_score:
                return True, 100
            return True
    
    return False, 0

def get_fuzzy_score(str1,str2):
    score = fuzz.ratio(str1.lower(),str2.lower())
    return score

def remove_accents(s):
    s = re.sub(r'[àáạảãâầấậẩẫăằắặẳẵ]', 'a', s)
    s = re.sub(r'[ÀÁẠẢÃĂẰẮẶẲẴÂẦẤẬẨẪ]', 'A', s)
    s = re.sub(r'[èéẹẻẽêềếệểễ]', 'e', s)
    s = re.sub(r'[ÈÉẸẺẼÊỀẾỆỂỄ]', 'E', s)
    s = re.sub(r'[òóọỏõôồốộổỗơờớợởỡ]', 'o', s)
    s = re.sub(r'[ÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠ]', 'O', s)
    s = re.sub(r'[ìíịỉĩ]', 'i', s)
    s = re.sub(r'[ÌÍỊỈĨ]', 'I', s)
    s = re.sub(r'[ùúụủũưừứựửữ]', 'u', s)
    s = re.sub(r'[ƯỪỨỰỬỮÙÚỤỦŨ]', 'U', s)
    s = re.sub(r'[ỳýỵỷỹ]', 'y', s)
    s = re.sub(r'[ỲÝỴỶỸ]', 'Y', s)
    s = re.sub(r'[Đ]', 'D', s)
    s = re.sub(r'[đ]', 'd', s)
    return s