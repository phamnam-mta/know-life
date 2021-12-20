import json

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