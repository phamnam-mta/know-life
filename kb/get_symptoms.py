'''
Get symptoms from wikipedia
'''

import json
import requests
import bs4
from bs4 import BeautifulSoup, NavigableString, Tag
import re
from dict_hash import sha256
from tqdm import tqdm

def read_json(path):
    with open(path) as f:
        DATA = json.load(f) 
    return DATA

class SymptomBuilder:
    def __init__(self, database_url):
        self.database = read_json(database_url)

    def resolution_symptoms(self):
        pass

    def eda_symptoms(self):
        pass

    def build_symptoms(self):
        pass

    def get_stats(self):
        pass

def remove_ref_tag(str):
    str = re.sub("(?<=\[)(.*?)(?=\])", "", str)
    str = str.replace('[]',' ')
    return str

def get_attribute_dict(s):
    ''' From disease (s) to result (dict)
    '''
    # URL
    # s = 'khó chịu'
    symptom = '_'.join(s.split(' '))
    URL = f"https://vi.wikipedia.org/wiki/{symptom}"

    # sending the request
    response = requests.get(URL)
    
    # parsing the response
    soup = bs4.BeautifulSoup(response.text, 'html')

    try:
        # getting infobox
        infobox = soup.find('table', {'class': 'infobox'})
        td_text = infobox.find_all("td")
        th_text = infobox.find_all("th")

        th_text = [item.text for item in th_text]
        th_text = th_text[1:]

        td_text = [item.text for item in td_text]
        td_text = td_text[:-1]
    except:
        pass

    main_attribute = soup.find_all('h2')
    main_attribute = [item.text for item in main_attribute if 'mục lục' not in item.text.lower()]
    main_attribute = ["overview"] + main_attribute

    for i,att in enumerate(main_attribute):
        x = re.search(".+?(?=\[)", att)
        if x != None:
            main_attribute[i] = x.group(0)
        if 'xem thêm' in att.lower():
            main_attribute = main_attribute[:i]
            break


    # get defintion text
    content = []
    try:
        result = soup.find("div", {"class":"mw-parser-output"})
        result = result.find_all('p')
        content = [remove_ref_tag(item.text) for item in result if s in item.text.lower()]
        content = [' '.join(content)]
    except:
        print(f"Yet, Wiki does not support for: {s}")
        return [], URL
    # Get text between h2 tag
    for header in soup.find_all('h2'):
        nextNode = header
        if 'xem thêm' in nextNode.text.lower():
            break
        tmp_content = []
        while True:
            nextNode = nextNode.nextSibling
            if nextNode is None:
                break
            if isinstance(nextNode, Tag):
                if nextNode.name == "h2":
                    break
                string = nextNode.get_text().strip()
                if string.strip() != '\n' and string.strip() != '':
                    tmp_content.append(remove_ref_tag(string.strip()))
        if tmp_content != []:
            content.append('\n'.join(tmp_content))

    result = []
    for title,c_ in zip(main_attribute,content):
        result.append({
            "attribute" : title,
            "content" : c_
        })
        
    try:
        for title,context in zip(th_text,td_text):
            result.append({
                "attribute" : title,
                "content" : context
            })
    except:
        print(f"Yet, Wiki does not support infobox for: {s}")

    return result, URL


with open('../data/kb/vinmec_data.json') as f:
    DATA = json.load(f) 

SYMPTOM = []
for sample in tqdm(DATA):
    attributes = sample['attributes']
    for att in attributes:
        if att['attribute'] == 'symptom' and 'entities' in att:
            for symptom in att['entities']:
                attributes, url = get_attribute_dict(symptom['name'])
                obj = {
                    'url' : url,
                    'symptom' : symptom['name'],
                    'attributes' : attributes
                }
                key = sha256(obj)
                symptom["key"] = key
                obj["key"] = key
                SYMPTOM.append(obj)
print(len(SYMPTOM))

with open(f'../data/kb/symptoms_data_wiki.json', 'w',encoding='utf-8') as f:
    json.dump(SYMPTOM, f,ensure_ascii=False) 


with open(f'../data/kb/vinmec_data2.json', 'w',encoding='utf-8') as f:
    json.dump(DATA, f,ensure_ascii=False) 