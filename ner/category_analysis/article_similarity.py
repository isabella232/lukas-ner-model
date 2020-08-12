from transformers import BertTokenizer
from torch import nn

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

from .utils.parse_articles import get_articles
from .category_similarity import create_embedding

"""
Quite messy code that did not produce any astonishing results, but kept here for future reference.
"""


def TFIDF_article_similarity():
    """Calculate similarities between articles using TFIDF."""
    stopwords = stopwords.words("swedish")
    articles = get_articles("data/articles_10k.json")

    corpus = [article["content_text"] for article in articles]

    vect = TfidfVectorizer(min_df=1, stop_words=stopwords)
    tfidf = vect.fit_transform(corpus)
    pairwise_similarity = tfidf * tfidf.T

    return pairwise_similarity.toarray()


def BERT_article_similarity():
    """Calculate similarities between articles using BERT word embeddings and cosine similarity."""
    articles = get_articles("data/articles_small.json")
    stopwords = stopwords.words("swedish")
    tokenizer = BertTokenizer.from_pretrained("KB/bert-base-swedish-cased-ner")
    cos = nn.CosineSimilarity(dim=0)

    embeddings = []

    for article in articles:
        temp = []
        sentences = article["content_text"].replace("\n\n", ".").split(".")
        for sentence in sentences:
            if not sentence.strip():
                continue
            words = sentence.split()
            ok_ind_w = [i for i in range(len(words)) if words[i] not in stopwords]
            token_lens = [len(tokenizer.encode(word)) - 2 for word in words]
            ok_ind_t = []
            for i in ok_ind_w:
                prev = sum(token_lens[0:i]) if i > 0 else 0
                ok_ind_t += [prev + i for i in range(0, token_lens[i])]
            ok_ind_t = [i + 1 for i in ok_ind_t]
            try:
                embedding = create_embedding(sentence.strip() + ".")
                embedding = embedding[:, ok_ind_t, :]
                temp += [embedding]
            except IndexError:  # 1541 max length sentence
                print(sentence)
                continue
        embeddings += [temp]
        print("-" * 100)
    print(len(embeddings), "article embeddings created!")

    all_sims = []
    for i, art_i in enumerate(embeddings):
        art_sims = []
        for j, art_j in enumerate(embeddings):
            print("Now comparing article", i, "with article", j, "…")
            if i == j:
                continue
            sen_sims = []
            for sen_i in art_i:
                tok_sims = []
                for sen_j in art_j:
                    for tok_i in range(0, sen_i.size()[1]):
                        for tok_j in range(0, sen_j.size()[1]):
                            sim = cos(sen_i[:, tok_i, :], sen_j[:, tok_j, :]).item()
                            tok_sims += [sim]
                sen_sims += [max(tok_sims)]
            art_sims += [(sum(sen_sims) / len(sen_sims)) ** 2]
        all_sims += [art_sims]
