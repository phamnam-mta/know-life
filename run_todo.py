import pandas as pd
import json
from fuzzywuzzy import fuzz

SYNONYM_KEY = 'synonym'

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

def get_closest_disease(entity,index,data):
    for sample in data:
        if SYNONYM_KEY in sample:
            for synonym in sample[SYNONYM_KEY]:
                is_relevant, score = is_relevant_string(synonym,entity,method=['exact','fuzzy','include'],return_score=True)
                if is_relevant:
                    break
        else:
            is_relevant, score = is_relevant_string(sample['disease'],entity,method=['exact','fuzzy','include'],return_score=True)

        if is_relevant:
            sample['index'] = index
            return sample
    
    result =  {
        "disease" : entity,
        'index' : index
    }

    return result

with open('./data/vinmec_data.json') as f:
    DISEASE = json.load(f)
    
def solve(data,pic_name,save_path):
    data_ = data.loc[data['PIC']==pic_name]['Disease'].tolist()
    index_ = data.loc[data['PIC']==pic_name]['Index'].tolist()
    result = []
    for index,d in zip(index_,data_):
        result.append(get_closest_disease(d,index,DISEASE))

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

data = pd.read_csv('data/pic.csv',',')

solve(data,'Minh','todo/minh_data.json')
solve(data,'Nam','todo/name_data.json')
solve(data,' vũ','todo/vu_data.json')
