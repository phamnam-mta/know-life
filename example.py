from src.model.inference_engine import InferenceEngine

if __name__ == '__main__':
    inference_engine = InferenceEngine(
        database_dir="./data", 
        max_answer_length=300
    )

    query = {
        'Xơ vữa động mạch ngoại biên' : ['overview', 'cause']
    }

    answer = inference_engine.query([query])

    for ans in answer:
        for rel_answer in ans:
            print(rel_answer)