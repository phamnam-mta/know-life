import json
import pandas as pd
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from constants import ATTRIBUTES_CHECKLIST, TOP_K_MISSING

def get_disease(data):
    result = []
    for sample in data:
        result.append(sample['disease'])
    return result

def count_complete_disease(data):
    print(f'Processing data : {len(data)} disease')
    
    result = {
        }

    for att in ATTRIBUTES_CHECKLIST:
        result[att] = []

    for sample in data:
        if 'synonym' in list(sample.keys()):
            result['synonym'].append(1)
        else:
            result['synonym'].append(0)
        if 'department_key' in list(sample.keys()):
            result['department_key'].append(1)
        else:
            result['department_key'].append(0)

        attributes = sample['attributes']
        attributes = [item['attribute'] for item in attributes]

        for k, v in result.items():
            if k in ['synonym','department_key']:
                continue
            if k in attributes:
                result[k].append(1)
            else:
                result[k].append(0)
    
    for k, v in result.items():
        print(f'{k} : {sum(result[k])} - {round(sum(result[k])/len(data),2)}')
    
    return result

def count_missing_disease(list_of_disease,count_disease):
    print('Analyse missing')

    missing_value = [0] * len(list_of_disease)    
    
    for k,v in count_disease.items():
        for i, val in enumerate(v):
            if val == 0:
                missing_value[i] += 1
    print('========================================')
    # missing X attribute
    for index in range(TOP_K_MISSING):
        degree = len(ATTRIBUTES_CHECKLIST) - index
        # sort missing value
        print(f'# miss-{degree}-attr disease: {len([item for item in missing_value if item == degree])} - {round(len([item for item in missing_value if item == degree])/len(list_of_disease),4)}')
        # number of missing-value
        # get index
        # get disease by index

    print('========================================')
    # attribute -> missing disease 
    for value , att in zip(count_disease.values(),ATTRIBUTES_CHECKLIST):
        print(f'attribute {att} has : {len([item for item in value if item==0])} missed diseases')

    missing_disease = []
    y_axis_labels = []
    for k,v in count_disease.items():
        missing_disease.append(v)
        y_axis_labels.append(k)

    df_cm = pd.DataFrame(missing_disease)
    plt.figure(figsize=(15,20))
    sn.set(font_scale=0.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 1.5},xticklabels=list_of_disease, yticklabels=y_axis_labels) # font size
    plt.xlabel('Disease')
    # plt.ylabel('Attributes')
    plt.show()
    return count_disease

def get_statistics():
    with open('../data/kb/vinmec_data.json') as f:
        data = json.load(f)

    list_of_disease = get_disease(data)

    # complete disease
    count_disease = count_complete_disease(data)

    # missing disease
    miss_disease = count_missing_disease(list_of_disease,count_disease)

def build_confusion_matrix():
    with open('../data/kb/data.json') as f:
        data = json.load(f)

    dep_num = 0

    columns = [
        "overview",
        "cause",
        "symptom",
        "riskfactor",
        "prevention",
        "diagnosis",
        "treatment",
        "infection",
    ]

    df = []
    disease = []
    for sample in data:

        disease.append(sample['disease'])

        attributes = sample["attributes"]
        attributes = [att['attribute'] for att in attributes]

        tmp_col = []

        if "synonym" in list(sample.keys()):
            tmp_col.append('yes')
        else:
            tmp_col.append('no')

        for col in columns:
            if col in attributes:
                tmp_col.append('yes')
            else:
                tmp_col.append('no')
        if "department_key" in list(sample.keys()):
            tmp_col.append('yes')
        else:
            tmp_col.append('no')

        df.append(tmp_col)

    # concate 2 list
    result = []
    for di, col in zip(disease,df):
        result.append([di] + col)

    columns = ["disease"] + ["synonym"] + columns + ["department_key"]
    df = pd.DataFrame(result,columns=columns)

    df.to_csv('stats.csv',',')

if __name__ == '__main__':
    # build_confusion_matrix()
    get_statistics()