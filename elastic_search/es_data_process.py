import json
import os
import pandas as pd
from tqdm import tqdm

WORK_DIR = os.path.abspath(os.getcwd())

def main():
    print("Loading dataset...")
    with open(os.path.join(WORK_DIR, "data/qa/kb/kb_qa_summary.json"), "r") as file:
        data = json.load(file)

    data_clean = []
    for d in tqdm(data):
        if d["question_word"] and len(d["answer"].split()) > 12 and d["summary"]:
            d["summary"] = " ".join(d["summary"])
            if d not in data_clean:
                data_clean.append(d)
    print(len(data_clean))
    with open(os.path.join(WORK_DIR, "data/qa/kb/es_kb_data.json"), "w") as file:
        json.dump(data_clean, file, ensure_ascii=False)

if __name__ == "__main__":
    main()