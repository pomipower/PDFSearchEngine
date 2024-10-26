import nltk
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('punkt_tab')

sentences = [
    "This is a sentence."
    "This is another sentence."
    "And yet another sentence."
]


tokenised_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]
print(tokenised_sentences)
model = Word2Vec(tokenised_sentences, vector_size = 100, window = 5, min_count=1, workers=4)

word_vector = model.wv['sentence']
print(word_vector.tolist())