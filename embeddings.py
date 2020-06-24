from transformers import BertModel, BertTokenizer
import torch
from torch import nn
from scipy.spatial.distance import cosine
import numpy as np
from wmd import WMD
import nltk
from nltk.corpus import stopwords
from collections import Counter
from parse_articles import get_articles

def create_embedding(sentence):
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids)
    last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
    return last_hidden_states

tokenizer = BertTokenizer.from_pretrained('KB/bert-base-swedish-cased-ner')
model = BertModel.from_pretrained('KB/bert-base-swedish-cased-ner')
cos = nn.CosineSimilarity()

articles = get_articles('data/small.json')
stopwords = stopwords.words('swedish')
documents = {}
for article in articles:
    title = article['title']
    text = article['content_text'].split()
    words = Counter([t for t in text if t not in stopwords])
    sorted_words = sorted(words)
    print([words[t] for t in sorted_words])
    documents[title] = (title, np.array([words[t] for t in sorted_words], dtype=np.float32))


class BERTEmbeddings(object):
    def __getitem__(self, item):
        return create_embedding(item)

calc = WMD(BERTEmbeddings, documents)
print(calc.nearest_neighbors('Järvsö IF:s veteraner tog 20 medaljer på SM i Malmö'))

# data_frames = create_data_frames()
# print('Data frames created')
# categories_df = link_entities_to_categories(data_frames[0], data_frames[3])
# print('Categories data frame created')


# categories_df['embedding'] = categories_df['category'].apply(lambda x: create_embedding(x))
# print('Embeddings created')
# copy_df = categories_df.copy()

# cnt = 0
# for i in categories_df.index:
#     emb_i = categories_df['embedding'][i]
#     for j in copy_df.index[i+1:]:
#         emb_j = categories_df['embedding'][j]
#         sim = cos(emb_i, emb_j)
#         if sim.item() > 0.995:
#             cnt += 1
#             print(i, j)
#             print('Merged categories', cnt, sim.item(), categories_df['category'][i], categories_df['category'][j])
    