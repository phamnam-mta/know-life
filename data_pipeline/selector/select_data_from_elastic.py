import os
import json
import asyncio
from tqdm import tqdm
from src.search_engine.elastic_search import ESKnowLife

WORK_DIR = os.path.abspath(os.getcwd())
NEG_PAGE_SIZE = 5
NEG_PAGE_INDEX = 20
es = ESKnowLife()

async def main():
    print("Loading dataset...")
    with open(os.path.join(WORK_DIR, "data/qa/es_data.json"), "r") as file:
        data = json.load(file)

    data_from_elastic = []
    for d in tqdm(data):
        es_data = await es.get_qa_pairs(d["question"], NEG_PAGE_INDEX, NEG_PAGE_SIZE)
        neg = es.elastic_to_qa(es_data)
        data_from_elastic.append({
            "positive": {
                "question": d["question_word"],
                "summary": d["summary"]
            },
            "negative": [{
                "question": n["question_word"],
                "summary": n["summary"],
            } 
            for n in neg]
        })

    with open(os.path.join(WORK_DIR, "data/qa/triples_elastic.json"), "w") as file:
        json.dump(data_from_elastic, file, ensure_ascii=False)

if __name__ == "__main__":
    asyncio.run(main())