import json
import os
import re
from tqdm import tqdm
from src.utils.tokenizer import words_seg
from src.utils.normalizer import text_normalize

WORK_DIR = os.path.abspath(os.getcwd())
DATASET_PATH = os.path.join(WORK_DIR, "data/qa/es_data.json")

def main():
    print("Loading dataset...")
    with open(os.path.join(WORK_DIR, "data/qa/covid.txt"), "r") as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines if line.strip()]

    qa_covid = []
    answer = []
    question = ""
    question_word = ""
    for idx, line in enumerate(lines):
        if re.search("^\d+.", line):
            if answer:
                _answer_display = "<br>".join(answer)
                _answer = " ".join(answer)
                _summary = " ".join(words_seg(text_normalize(_answer)))
                qa_covid.append({
                    "question": question,
                    "question_word": question_word,
                    "answer": _answer,
                    "answer_display": _answer_display,
                    "summary": _summary
                })
            answer = []
            question = line
            question_word = " ".join(words_seg(text_normalize(line)))
        else:
            answer.append(line)
        if idx == len(lines) - 1 and answer:
            _answer_display = "<br>".join(answer)
            _answer = " ".join(answer)
            _summary = " ".join(words_seg(text_normalize(_answer)))
            qa_covid.append({
                "question": question,
                "question_word": question_word,
                "answer": _answer,
                "answer_display": _answer_display,
                "summary": _summary
            })
            answer = []

    with open(DATASET_PATH, 'r') as open_file:
        data = json.load(open_file)

    data.extend(qa_covid)

    with open(os.path.join(WORK_DIR, "data/qa/es_data_with_covid.json"), "w") as file:
        json.dump(data, file, ensure_ascii=False)

if __name__ == "__main__":
    main()