from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models import Word2Vec
import numpy as np
from gensim.models import KeyedVectors

def get_tfidf_features(texts, max_features=1000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

def get_ngram_features(texts, n=2):
    vectorizer = CountVectorizer(ngram_range=(1,n))
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

def get_word2vec_features(texts, size=100):
    tokenized  = [t.split() for t in texts]
    #model = Word2Vec(tokenized, vector_size = size, window=5, min_count=2)
    model = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
    X = np.array([np.mean([model.wv[w] for w in words if w in model.wv] or [np.zeros(size)], axis=0) for words in tokenized])
    return X, model