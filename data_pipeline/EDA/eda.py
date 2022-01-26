import pandas as pd
import re
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer, models
from tqdm import tqdm

def check_valid(text):
    count = 0
    for t in text.split():
        if not re.match("^[a-zA-Z0-9đĐ_]", t):
            count += 1
    return False if count >= len(text.split())//2 else True

titles = pd.read_csv("/Users/namph/Documents/know-life/data/eda/title_word.csv")
titles = titles.loc[titles["titles"].apply(lambda x: check_valid(x))]["titles"].values.tolist()
print(len(titles))

model_name = '/Users/namph/Documents/know-life/models/ranking'
word_embedding_model = models.Transformer(model_name)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='cls')
sentence_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

topic_model = BERTopic(embedding_model=sentence_model)
embeddings = sentence_model.encode(titles, show_progress_bar=True)
topics, probs = topic_model.fit_transform(titles, embeddings)
print(topic_model.get_topic_info())
