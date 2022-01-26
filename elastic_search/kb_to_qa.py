import json
import os

TEMPLATE = {
    "overview": "bệnh {disease} là bệnh gì",
    "cause": "nguyên nhân của bệnh {disease}",
    "symptom": "triệu chứng của bệnh {disease}",
    "risk_factor": "Đối tượng nguy cơ mắc bệnh {disease}",
    "treatment": "Các biện pháp điều trị bệnh {disease}",
    "diagnosis": "các biện pháp chẩn đoán bệnh {disease}",
    "prevention": "cách phòng ngừa bệnh {disease}",
    "severity": "bệnh {disease} có nguy hiểm không"
}
WORK_DIR = os.path.abspath(os.getcwd())

with open(os.path.join(WORK_DIR, "data/kb/disease.json"), "r") as file:
    data = json.load(file)
print("data length: ", len(data))

qa_pair = []

keys = TEMPLATE.keys()
for d in data:
    for k in keys:
        if d.get(k):
            qa_pair.append({
                "question": TEMPLATE[k].format(disease=d.get("name")),
                "answer_display": d.get(k),
                "answer": d.get(k).replace("<br>", "")
            })
print(len(qa_pair))
with open(os.path.join(WORK_DIR, "data/qa/kb_qa.json"), "w") as file:
    json.dump(qa_pair, file, ensure_ascii=False)