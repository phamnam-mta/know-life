import json
import os
from tqdm import tqdm
from sentence_handler import SentenceHandler
from src.utils.tokenizer import words_seg
from src.utils.normalizer import text_normalize

WORK_DIR = os.path.abspath(os.getcwd())
sentence_handler = SentenceHandler()

if __name__ == "__main__":
    print("Loading dataset...")
    with open(os.path.join(WORK_DIR, "data/qa/kb/kb_qa.json"), "r") as file:
        data = json.load(file)
    
    print("Segment sentences...")
    for d in tqdm(data):
        try:
            d["question_word"] = " ".join(words_seg(text_normalize(d["question"])))
            d["sentences"] = sentence_handler.process(d["answer"]) 
        except Exception as e:
            print(e)
            print(d)
    with open(os.path.join(WORK_DIR, "data/qa/kb/kb_qa_sent.json"), "w") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)