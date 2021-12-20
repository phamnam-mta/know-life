import numpy as np
from typing import List, Text
from sentence_transformers.cross_encoder import CrossEncoder
from src.utils.normalizer import text_normalize
from src.utils.tokenizer import words_seg

class BERTRanker():
    def __init__(self, model_path=None) -> None:

        self.model = CrossEncoder(model_path)

    def re_ranking(self, question: Text, candidates: List[Text]) -> List:
        question = " ".join(words_seg(text_normalize(question)))
        model_input = [[question, doc] for doc in candidates]
        pred_scores = self.model.predict(model_input, convert_to_numpy=True, show_progress_bar=True)
        pred_scores_argsort = np.argsort(-pred_scores)

        return pred_scores_argsort, pred_scores
