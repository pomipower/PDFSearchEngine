import nltk
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import numpy as np
nltk.download('punkt')
from sklearn.metrics.pairwise import cosine_similarity

# Function to get the vector representation of query sentence
def get_sentence_vector(sentence, model):
    tokenized_sentence = word_tokenize(sentence.lower())
    vector = np.mean([model.wv[word] for word in tokenized_sentence if word in model.wv], axis=0)
    return vector

def compareVectors(vector1, vector2):
    return cosine_similarity([vector1], [vector2])

#this vector represents this sentence: The coffee shop on the corner serves the best lattes in town
sentence_vector = np.array([-4.9887580e-04, -2.3809415e-03, 9.3093142e-04, 3.2042291e-03,
                            -3.9801751e-03, -2.3811718e-03, 2.8289987e-03, 5.2091782e-03,
                            -1.2621776e-03, -1.4867376e-03, 1.5740270e-03, 7.6798932e-04,
                            -2.5865994e-03, 1.5035601e-03, -9.8180654e-04, -9.9678617e-04,
                            -2.3248042e-04, 4.5057709e-04, -5.8167684e-03, -4.6154051e-04,
                            3.9682128e-03, 1.4168300e-03, 4.6709725e-03, -2.8698859e-04,
                            1.9069893e-03, -2.0986146e-03, 6.4534172e-05, -7.2406564e-04,
                            -3.3013182e-04, -2.9578470e-04, -3.2942330e-03, 7.3410803e-04,
                            5.5335560e-03, -4.2682257e-03, 7.2600431e-04, 1.1061140e-03,
                            3.2600426e-04, -1.0786004e-03, -2.2579357e-04, -1.8144391e-03,
                            -1.1615712e-03, 1.5849322e-03, -6.7443057e-04, -7.8192010e-04,
                            2.0659615e-03, 6.2511908e-04, -1.1150591e-03, 1.5321866e-03,
                            2.0217218e-03, 1.7297626e-03, 1.4492963e-04, -4.8707603e-04,
                            -1.5127991e-03, -1.4024858e-03, 6.2129949e-04, -1.0777771e-03,
                            1.5729053e-04, -1.4390807e-03, 1.2015473e-03, 2.6403188e-03,
                            1.2549215e-03, 7.0479023e-04, -8.3382335e-04, -2.8657403e-03,
                            -1.5740782e-03, 6.1587547e-05, -9.2952303e-04, 4.5807795e-03,
                            -3.2533417e-03, 1.6812244e-03, 1.0984658e-03, 3.1764540e-04,
                            8.2107336e-04, -1.6330895e-03, 1.0177026e-03, -1.0938380e-03,
                            2.8938539e-03, 7.0211478e-05, -9.1677048e-04, -5.3821411e-04,
                            -2.4886031e-04, 2.6549010e-03, -5.6753092e-04, 8.8703242e-04,
                            -3.9318274e-03, -9.4876671e-04, 2.9068387e-03, -2.5314544e-03,
                            -2.0813430e-03, 1.1953107e-03, 2.3338113e-04, -2.1814138e-03,
                            1.7479014e-03, 5.9207465e-04, 4.5559015e-03, 2.5675923e-03,
                            -2.3775117e-03, -1.5038979e-03, -2.2600952e-03, 6.0099224e-04],
                           dtype=np.float32)

# Taking user input for the query
query = input("\nEnter query: ")

query_vector = get_sentence_vector(query, model)
print("Your query's vector is: " + str(query_vector))

similarity = cosine_similarity([query_vector], [sentence_vector])

print(similarity[0][0])
