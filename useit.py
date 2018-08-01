import json
import numpy as np
import keras
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
from keras.models import model_from_json

tokenizer = tokenizer(num_words=3000)

labels = [ 'negative','positive']

with open('dictionary.json ', 'r') as dictionary_file:
    dictionary = json.load(dictionary_file)

def text_to_index_array(text):
    words = kpt.text_to_word_sequence
    wordIncdices = []
    for word in words:
        if word in dictionary:
            wordIncdices.append(dictionary[word])
        else:
            print("%s is not in dictionary corpus" %(word))

    return wordIncdices


json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()
model  = model_from_json(loaded_model_json)
model.load_weights('model.h5')


while 1:
    eval_sentence = raw_input("Input a twit to be evaluated ")

    if len(eval_sentence) ==0:
        break

    test_arr = text_to_index_array(eval_sentence)
    input = tokenizer.sequences_to_matrix([test_arr],mode ='binary')
    pred = model.predict(input)
    print("%s sentiment %f%%confidence" %(labels[np.argmax(pred)],pred[0][np.argmax(pred)]*100))