import os
import json
from tqdm import tqdm
from transformers import *
from sentence_handler import SentenceHandler
from summarizer import Summarizer

WORK_DIR = os.path.abspath(os.getcwd())

def main():
    print("Loading pretrain...")
    custom_config = AutoConfig.from_pretrained('vinai/phobert-base')
    custom_config.output_hidden_states=True
    custom_tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
    custom_model = AutoModel.from_pretrained('vinai/phobert-base', config=custom_config)
    model = Summarizer(custom_model=custom_model, custom_tokenizer=custom_tokenizer, sentence_handler=SentenceHandler())

    print("Loading dataset...")
    with open(os.path.join(WORK_DIR, "data/qa/QA.json"), "r") as file:
        data = json.load(file)
    
    print("Summarizing...")
    for d in tqdm(data):
        try:
            d["summary"] = model(d["answer"], num_sentences=5)
        except Exception as e:
            print(d)
            print(e)

    with open(os.path.join(WORK_DIR, "data/qa/QA_summary.json"), "w") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()