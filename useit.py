import json
import os
import numpy as np
import keras
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
from keras.models import model_from_json

tokenizer = Tokenizer(num_words=3000)

labels = [ 'negative','positive']

datapath = os.path.dirname(os.path.abspath(__file__))

with open('dictionary.json', 'r') as dictionary_file:
    dictionary = json.load(dictionary_file)

def text_to_index_array(text):
    words = kpt.text_to_word_sequence(text)
    wordIncdices = []
    for word in words:
        if word in dictionary:
            wordIncdices.append(dictionary[word])
        else:
            print("%s is not in dictionary corpus " %(word))
    return wordIncdices

json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()
model  = model_from_json(loaded_model_json)
model.load_weights('twit_model.h5')

while True:
    eval_sentence = input('Input a twit to be evaluated  :  ')
    if len(eval_sentence) == 0:
        break

    test_arr = text_to_index_array(eval_sentence)
    _input_ = tokenizer.sequences_to_matrix([test_arr],mode ='binary')
    pred = model.predict(_input_)
    print("%s sentiment %0.2f%% confidence \n" %(labels[np.argmax(pred)],pred[0][np.argmax(pred)]*100))
