import asyncio
from src.search_engine.semantic_search import SemanticSearch

async def main():
    inference_engine = SemanticSearch()

    query = 'bệnh ung thư điều trị như nào?'

    re_ranking, es_ranking = await inference_engine.search(query, page_size=5)

    for a in es_ranking:
        print("question: {}\nanswer:{}".format(a["question"], a["answer"]))
        print("----------")

    print("\n\n")
    print("-----Re-ranking-----")
    for a in re_ranking:
        print("score: {}\nquestion: {}\nanswer: {}".format(a["score"], a["question"], a["answer"]))
        print("----------")

if __name__ == "__main__":
    asyncio.run(main())