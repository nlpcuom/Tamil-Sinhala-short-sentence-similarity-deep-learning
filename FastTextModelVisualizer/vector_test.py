from __future__ import print_function
from gensim.models import KeyedVectors

# Creating the model
## Takes a lot of time depending on the vector file size 
en_model = KeyedVectors.load_word2vec_format('C:\\Users\\Nilaxan\\Documents\\GitHub\\pre-trained-word-embedding-models\\fastText\\facebook\\cc.ta.300.vec')

# Getting the tokens 
words = []
for word in en_model.vocab:
    words.append(word)

# Printing out number of tokens available
print("Number of Tokens: {}".format(len(words)))

# Printing out the dimension of a word vector 
print("Dimension of a word vector: {}".format(
    len(en_model[words[0]])
))

# Print out the vector of a word 
print("Vector components of a word: {}".format(
    en_model[words[0]]
))

# Number of Tokens: 2000000
# Dimension of a word vector: 300

# Pick a word 
find_similar_to = 'சிங்கம்'

# Finding out similar words [default= top 10]
for similar_word in en_model.similar_by_word(find_similar_to):
    print("Word: {0}, Similarity: {1:.2f}".format(
        similar_word[0], similar_word[1]
    ))

# Output 
# Word: சிங்கம்2, Similarity: 0.84
# Word: சிங்கம்3, Similarity: 0.83
# Word: சிங்கம்யா, Similarity: 0.81
# Word: சிங்கமடா, Similarity: 0.80
# Word: சிங்கமேவ, Similarity: 0.80
# Word: சிங்கமென, Similarity: 0.79
# Word: சிங்கம்ல, Similarity: 0.79
# Word: சிங்கம்Apr, Similarity: 0.79
# Word: சிங்கம்லே, Similarity: 0.79
# Word: சிங்கமுலு, Similarity: 0.78

# Test words 
word_add = ['சிங்கம்', 'புலி']
word_sub = ['குதிரை']

# Word vector addition and subtraction 
for resultant_word in en_model.most_similar(
    positive=word_add, negative=word_sub
):
    print("Word : {0} , Similarity: {1:.2f}".format(
        resultant_word[0], resultant_word[1]
    ))

# Output 

# Word : சிங்கப்புலி , Similarity: 0.63
# Word : சிங்கம்3 , Similarity: 0.62
# Word : சிங்கம்2 , Similarity: 0.60
# Word : சிங்கம்புலி , Similarity: 0.59
# Word : சிங்கம்யா , Similarity: 0.58
# Word : சிங்கமடா , Similarity: 0.58
# Word : சிங்கம்லே , Similarity: 0.57
# Word : புலி12 , Similarity: 0.57
# Word : புலி11 , Similarity: 0.57
# Word : சிங்கம்ல , Similarity: 0.57