import re
import json

from fuzzywuzzy import fuzz

def read_txt(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(line.replace('\n',''))
    return data

def read_json(path):
    with open(path, 'r', encoding='utf8') as f:
        data = json.load(f)
    return data

def write_json(data,path):
    # print(f"Save data into {path}")
    with open(path, 'w') as f:
        json.dump(data, f, ensure_ascii=False)

def is_relevant_string( str1,
                        str2,
                        remove_accent=False,
                        method=['exact','fuzzy','include'],
                        score=60):
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
            return True

    # 2. fuzzy match
    if 'fuzzy' in method:
        ratio = fuzz.ratio(str1.lower(),str2.lower())
        if ratio  > score:
            return True
    
    # 3. include match
    if 'include' in method:
        if str1 in str2 or str2 in str1:
            return True
    
    return False

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