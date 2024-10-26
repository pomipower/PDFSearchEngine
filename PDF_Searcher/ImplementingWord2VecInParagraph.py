#Implementing the word to vector code  to a paragraph
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')

paragraph = """The cat jumped over the fence in the backyard. I went to the store to buy some groceries. I read this sentence yesterday, it was great yesterday. Learning new languages can be challenging but fun. The weather today is perfect for a walk in the park. He enjoys reading books about history and science. This is an example of a short sentence. They decided to take a trip to the mountains next weekend. My favorite hobby is painting landscapes during my free time.this seems to be a great sentence. The coffee shop on the corner serves the best lattes in town."""

# Splitting paragraph as per sentences, tokenising paragraph per sentence in a 2D array
sentences = paragraph.split('.')
sentences.pop()
tokenised_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]

#Creating Word2Vec model on tokenised sentences
model = Word2Vec(tokenised_sentences, vector_size=100, window=5, min_count=1, workers=4)
#model = Word2Vec.load('C:\Users\Ashok\Downloads\bert-master\bert-master')

sentence_vectors = []

#Getting average vector for each sentence
for tokenised_sentence in tokenised_sentences:
    if tokenised_sentence:
        vector = np.mean([model.wv[word] for word in tokenised_sentence if word in model.wv], axis=0)
        sentence_vectors.append(vector)

print(sentence_vectors)
print("Length of sentence vectors is:" + f"{len(sentence_vectors)}")
print(sentences)
print("Length of sentences is:" + f"{len(sentences)}")
# Making HashMap, where keys are the sentences and the value is the vector array

sentences_vec_map = {}
"""for i, sentence in enumerate(sentences):
    sentences_map[sentence] = tuple(sentence_vectors[i])"""
# Taking average vector of a sentence and making average_sentence_vec array

for i, vectors in enumerate(sentence_vectors):
    sentences_vec_map[tuple(vectors)] = sentences[i]


def get_sentence_vector(sentence, model):
    tokenized_sentence = word_tokenize(sentence.lower())
    vector = np.mean([model.wv[word] for word in tokenized_sentence if word in model.wv], axis=0)
    return vector
#Comparing two vectors function


query = input("\nEnter query: ")

query_vector = get_sentence_vector(query, model)
print("your query vector is:"+ str(query_vector))

similarity_map = {}
for keys in sentences_vec_map.keys():
    similarity = cosine_similarity([query_vector], [keys])
    """string_val = sentences_vec_map[keys]
    sentences_vec_map[tuple(similarity)] = string_val"""
    similarity_map[similarity[0][0]] = sentences_vec_map[keys] # some problem here?
print("This is the raw similarities of each sentence compared to other sentences: " + str(similarity_map.keys()))
print("This is the sorted similarities, which is more helpful, since when user gives a query, \nwe wan't the most "
      "useful result: " + str(sorted(similarity_map.keys(), reverse=True)))

needed_str = max(similarity_map.keys())
print("The search result is: " + similarity_map[needed_str])

