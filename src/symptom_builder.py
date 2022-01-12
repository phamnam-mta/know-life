import json
import requests
import bs4
import re

from tqdm import tqdm
from dict_hash import sha256
from bs4 import BeautifulSoup, NavigableString, Tag

class SymptomBuilder:
    def __init__(self, database_url):
        self.database = read_json(database_url)

    def resolution_symptoms(self):
        pass

    def build_symptoms(self):
        pass

    def get_statistics(self):
        # number of symptoms
        # number of synonyms
        # avg synonym per symptom
        # number of complete(full-info) symptom
        # number of incomplete symptom
        pass

    def get_symptom_attribute(self,symptom,source='wiki'):
        ''' Get symptom's attributes from Open-KB (e.g. : Wiki, YAGO,...)
        '''
        if source == 'wiki':
            # dict
            result = get_symptom_from_wiki(symptom)
        
        return result
    
    def get_symptom_from_wiki(self,symptom):
        ''' From disease (s) to result (dict) or None
        '''
        symptom = '_'.join(symptom.split(' '))
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
            return None

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

        result = {}
        for title,c_ in zip(main_attribute,content):
            result[title] = c_
        result['url'] = URL

        try:
            for title,context in zip(th_text,td_text):
                result[title] = content
        except:
            print(f"Yet, Wiki does not support infobox for: {s}")

        return result
