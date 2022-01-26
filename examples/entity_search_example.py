from src.search_engine import EntitySearch

if __name__ == '__main__':
    inference_engine = EntitySearch()

    query = 'bệnh ung thư điều trị như nào?'

    res = inference_engine.query(query)

    print(res)
