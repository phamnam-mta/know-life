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

    with open("./data/QA_sent.json") as file:
        data = json.load(file)

    for d in tqdm(data):
        try:
            if len(d["sentences"]) > 5:
                sents, _ = model.cluster_runner(d["sentences"], num_sentences=5)
                d["summary"] = sents
            else:
                d["summary"] = d["sentences"]
        except Exception as e:
            print(d)
            print(e)
            
    with open("./data/QA_summary.json", "w") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()