import re
import json

from fuzzywuzzy import fuzz

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
        if str1 in str2 or str2 in str1:
            if return_score:
                return True, 100
            return True
    
    return False, 0

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