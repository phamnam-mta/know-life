from src.search_engine import EntitySearch

if __name__ == '__main__':
    inference_engine = EntitySearch()

    query = 'bệnh ung thư điều trị như nào?'

    answer = inference_engine.query(query)

    for ans in answer:
        for rel_answer in ans:
            print(rel_answer)
