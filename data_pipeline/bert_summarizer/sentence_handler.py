from typing import List

from src.utils.normalizer import text_normalize
from src.utils.tokenizer import sentences_seg

class SentenceHandler(object):

    def sentence_processor(self, sentences,
                           min_length: int = 4,
                           max_length: int = 128) -> List[str]:
        """
        Processes a given spacy document and turns them into sentences.
        :param doc: The document to use from spacy.
        :param min_length: The minimum token length a sentence should be to be considered.
        :param max_length: The maximum token length a sentence should be to be considered(long more will be truncated).
        :return: Sentences.
        """
        to_return = []

        for s in sentences:
            num_token = len(s.split())
            
            if num_token > max_length:
                num_split = num_token//max_length
                if num_token%max_length > 0:
                    num_split += 1
                sent_size = num_token//num_split
                for i in range(num_split):
                    start = i*sent_size
                    end = start + sent_size
                    if i == num_split - 1:
                        end = num_token
                    to_return.append(" ".join(s.split()[start:end]))
            elif num_token > min_length:
                to_return.append(s)

        return to_return

    def process(self, body: str,
                min_length: int = 4,
                max_length: int = 128) -> List[str]:
        """
        Processes the content sentences.
        :param body: The raw string body to process
        :param min_length: Minimum token length that the sentences must be
        :param max_length: Max length token that the sentences mus fall under(long more will be truncated)
        :return: Returns a list of sentences.
        """
        sentences = sentences_seg(text_normalize(body))
        return self.sentence_processor(sentences, min_length, max_length)

    def __call__(self, body: str,
                 min_length: int = 4,
                 max_length: int = 128) -> List[str]:
        """
        Processes the content sentences.
        :param body: The raw string body to process
        :param min_length: Minimum token length that the sentences must be
        :param max_length: Max token length that the sentences mus fall under(long more will be truncated)
        :return: Returns a list of sentences.
        """
        return self.process(body, min_length, max_length)