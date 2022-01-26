import json
import os
import random
import pandas as pd
from tqdm import tqdm

WORK_DIR = os.path.abspath(os.getcwd())

def main():
    print("Loading dataset...")
    with open(os.path.join(WORK_DIR, "data/qa/triples_elastic.json"), "r") as file:
        data = json.load(file)
        random.shuffle(data)
    
    split_index = int(len(data)*0.9)
    train = {"sent1":[], "sent2": [], "score": []}
    for d in tqdm(data[:split_index]):
        pos = d["positive"]
        neg = d["negative"]

        train["sent1"].append(pos["question"])
        train["sent2"].append(pos["summary"])
        train["score"].append(1)

        for n in neg:
            train["sent1"].append(pos["question"])
            train["sent2"].append(n["summary"])
            train["score"].append(0)
    
    with open(os.path.join(WORK_DIR, "data/qa/train.json"), "w") as file:
        json.dump(train, file, ensure_ascii=False)

    df = pd.DataFrame(train)
    df.to_csv("data/qa/train.csv", index=False)

    test = []
    for d in tqdm(data[split_index:]):
        pos = d["positive"]
        neg = d["negative"]

        test.append({'query': pos["question"], 
                    'positive': [pos["summary"]], 
                    'negative': [n["summary"] for n in neg]})
    
    with open(os.path.join(WORK_DIR, "data/qa/dev.json"), "w") as file:
        json.dump(test, file, ensure_ascii=False)

if __name__ == "__main__":
    main()