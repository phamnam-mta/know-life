from src.inference_engine import InferenceEngine

if __name__ == '__main__':
    inference_engine = InferenceEngine(
        database_dir="./data", 
        model_dir='./ckpt',
        max_answer_length=300
    )

    query = 'ung thư điều trị như nào?'

    answer = inference_engine.query(query)

    for ans in answer:
        for rel_answer in ans:
            print(rel_answer)