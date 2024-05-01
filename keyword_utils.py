import numpy as np
import scipy.sparse
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from textblob import Word
from tqdm import trange


def keywords_extractor(
    texts: list[str],
    vectorizer: CountVectorizer,
    model: SentenceTransformer,
    k: int = 5,
) -> list[list[str]]:
    """Extract k keywords for each text in a list of texts."""
    matrix = vectorizer.fit_transform(texts)
    words = vectorizer.get_feature_names_out()

    singularizer = lambda word: Word(word).singularize()
    singularizer = np.vectorize(singularizer)
    words = singularizer(words)

    embedding_text = model.encode(texts, show_progress_bar=True)
    embedding_words = model.encode(words, show_progress_bar=True)

    keywords = []
    for i in range(len(texts)):
        indexes = matrix[i].nonzero()[1]
        similarity = embedding_words[indexes] @ embedding_text[i].T
        similarity = similarity.flatten()

        if len(words[indexes]) <= k:
            keywords.append(set(words[indexes].tolist()))
        else:
            k_best_indexes = np.argpartition(-similarity, k)[:k]
            keywords.append(set(words[indexes][k_best_indexes].tolist()))

    return keywords


def keywords_to_adjacency(keywords: list[set[str]]) -> scipy.sparse.coo_matrix:
    """Get the number of similar keywords in each set of keywords."""
    n = len(keywords)
    coord_x = []
    coord_y = []
    value = []

    for i in trange(n):
        for j in range(i + 1, n):
            buff = len(keywords[i].intersection(keywords[j]))
            if buff > 0:
                coord_x += [i]
                coord_y += [j]
                value += [buff]

    print(len(value), len(coord_x), len(coord_y))

    adjacency = scipy.sparse.coo_matrix(
        (value + value, (coord_x + coord_y, coord_y + coord_x)), shape=(n, n)
    )
    return adjacency
