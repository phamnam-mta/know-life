import json
import logging
from dict_hash import sha256

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

# data =  load_json('data/kb/symptom_data.json')

# print(len(data))
# result = []
# for sample in data:
#     obj = {}
#     name = list(sample.keys())[0]
#     obj['name'] = list(sample.keys())[0]
#     for k,v in sample[name].items():
#         print(k)
#         if k == 'define':
#             obj['overview'] = v
#         else:
#             obj[k] = v
#     obj['key'] = sha256(obj)
#     result.append(obj)

# write_json(result,'data/kb/symptom_data_msd.json')

# merge symptom databases
data_msd = load_json('data/kb/symptom_data_msd.json')
data_wiki = load_json('data/kb/symptom_data_wiki.json')
data_msd = data_msd + data_wiki
write_json(data_msd,'data/kb/symptom_data.json')

# data = load_json('data/kb/symptom_data_wiki.json')
# result = []
# for sample in data:
#     obj = {}
#     obj['url'] = sample['url']    
#     obj['name'] = sample['name']    
#     obj['key'] = sample['key']    
#     attributes = sample['attributes']
#     for att in attributes:
#         if att['attribute'].lower() not in ['tham khảo', 'bảng chọn điều hướng"']:
#             obj[att['attribute']] = att['content'].replace('\n','').replace("\"",'"')
#     result.append(obj)
# write_json(result,'data/kb/symptom_data_wiki.json')


# format into correct schema
# data = load_json('data/kb/symptom_data.json')

# for sample in data:
#     if 'url' in sample:
#         for k,v in sample.items():
#             if type(v) == list:
#                 try:
#                     v = ' '.join(v)
#                 except:
#                     print(sample)
#                     exit()

# write_json(data_msd,'data/kb/symptom_data.json')