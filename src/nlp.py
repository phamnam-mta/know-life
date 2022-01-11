import pandas as pd
import numpy as np

from numpy import dot
from numpy.linalg import norm
from gensim.models.keyedvectors import KeyedVectors
from underthesea import pos_tag

from utils import get_fuzzy_score
from constants import WORD2VEC_URL

class NLPToolKit:
    def __init__(self,model_path=WORD2VEC_URL):
        self.word2vec = self.load_w2v(model_path)

    @staticmethod
    def get_similarity_score(x,y):
        cosine = dot(x, y)/(norm(x)*norm(y))
        return cosine 

    def load_w2v(self,path):
        try:
            print(f'Loading Word2Vec model from {path}')
            model = KeyedVectors.load_word2vec_format(path, binary=False)
        except:
            print(f'Path {path} of Word2Vec model not found')
            return None
        return model

    def get_synonym(self,word,method='dict',top_k=1,p=0.1):
        '''
        Args:
            - word (str)
            - method ['dict','w2v']
            - p (float) : % replace of whole sentence
        Return:
            - result (list) : List of synonym
        '''
        result = []
        # syllable
        if len(word.split(' ')) == 1:
            if method == 'w2v':
                # [('tiên', 0.9147775173187256), ('mới', 0.9007983207702637)]
                synponym = self.word2vec.wv.most_similar(word, topn=top_k)  
                result = [item[0] for item in synponym]

            if method == 'dict':
                # TODO
                pass
        else:
            if method == 'w2v':
                cur_p = 0
                pos_tag_out = pos_tag(word)
                
                for pos in pos_tag_out:
                    token = pos[0]
                    tag = pos[1]
                
                    del_p = cur_p / len(pos_tag_out)

                    if tag in ['A','V','Vp'] and del_p < p:
                        synonym = self.get_synonym(token,method='w2v',top_k=top_k)
                        
                        if synonym != []:
                            for s in synonym:
                                if cur_p / len(pos_tag_out) > p:
                                    break
                                new_synonym = word.replace(token,s)
                                result.append(new_synonym)
                                cur_p += 1
                            
            if method == 'dict':
                # TODO
                pass
            pass
        return result 

    def from_entity_to_vector(self, word):
        ''' Return np.ndarry
        '''
        # syllable
        if len(word.split(' ')) == 1:
            vector = self.word2vec.wv[word]
            # print(nlp.word2vec.wv.most_similar('đau', topn=10))

        # syllable
        if len(word.split(' ')) == 1:
            vector = self.word2vec.wv[word]
        else:
            vector = []
            for token in word.split(' '):
                vector.append(self.word2vec.wv[token])

            # average pooling
            vector = np.mean(vector, axis=0)    
        return vector
    
    def map_entity_to_list(self,entity,list_of_entities,method='fuzzy'):
        ''' Resolution an entity
        e.g. "đau đầu" -> "nhức đầu" (in database)
        Args:
            - method : ['fuzzy','w2v']
        Return:
            - result (str) 
            - score (float) 
        '''
        result = entity
        max_score = 0
        for word in list_of_entities:
            if method == 'fuzzy':
                score = get_fuzzy_score(entity,word)
            
            if method == 'w2v':
                entity_vector = self.from_entity_to_vector(entity)
                word_vector = self.from_entity_to_vector(word)
                score = self.get_similarity_score(entity_vector,word_vector)
            
            if score > max_score:
                max_score = score
                result = word
            
        return result, max_score
        
if __name__ == '__main__':
    nlp = NLPToolKit()
    print(nlp.map_entity_to_list('nhức đầu',['đau đầu','chóng mặt','sốt','ho','buồn','co giật chân tay'],method='w2v'))
    print(nlp.get_synonym('nhức đầu',method='w2v',p=0.3))
    print(nlp.get_synonym('chóng mặt',method='w2v',p=0.3))
