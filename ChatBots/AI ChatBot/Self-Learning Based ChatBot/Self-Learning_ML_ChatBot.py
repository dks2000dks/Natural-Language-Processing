"""
The following code is incomplete.
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import keras
from sklearn.model_selection import train_test_split
import operator
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras import layers , activations , models , preprocessing , utils
import nltk as nlp
import re
from nltk.tokenize import word_tokenize 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import string
from gtts import gTTS
from io import BytesIO
import os
import yaml
import tensorflow as tf
from playsound import playsound
import warnings
warnings.filterwarnings('ignore')
print ("------------------------Imported Libraries--------------------")
print()

# importing .yml file
with open('ai.yml', 'r', errors ='ignore') as stream:
	data = yaml.safe_load(stream)

# Extracting Questions and Answers
Questions = []
Answers = []
for l in data['conversations']:
	Questions.append(l[0])
	Answers.append(l[1])
print ("No.of Conversations in Dataset ",len(Questions))

# Pre-Processing Data Functions
Non_Eng_Letters_Regex = re.compile('[^a-zA-Z ]')

# Remove all English Non-Letters
def cleanEnglishVocab(List):
	Output = []
	for line in List:
		line = line.replace('-', ' ').replace(',', ' ').upper()
		line = Non_Eng_Letters_Regex.sub('', line)
		Output.append(line)
	return Output

Questions = cleanEnglishVocab(Questions)
Answers = cleanEnglishVocab(Answers)   

# Vocabulary Size
tokenizer = preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(Questions + Answers)
VOCAB_SIZE = len(tokenizer.word_index)+1
print( 'Size of Vocabulary : {}'.format(VOCAB_SIZE))

Vocabulary = []
for word in tokenizer.word_index:
	Vocabulary.append(word)
	
# Tokenising Data
def Tokenise_Data(Dataset,Vocabulary):
	tokenizer = preprocessing.text.Tokenizer()
	tokenizer.fit_on_texts(Vocabulary) 
	Tokenized_Lines = tokenizer.texts_to_sequences(Dataset)
	return Tokenized_Lines
	
Tokenised_Questions_Data = Tokenise_Data(Questions,Vocabulary)
Tokenised_Answers_Data = Tokenise_Data(Answers,Vocabulary)

print (Tokenised_Answers_Data)

# Properties of Data
Max_Input_Length = len(max(Tokenised_Questions_Data,key=len))
print ("Maximum Question Length ",Max_Input_Length)

Max_Output_Length = len(max(Tokenised_Answers_Data,key=len))
print ("Maximum Answer Length ",Max_Output_Length)

# Padding Data
Padded_Questions = preprocessing.sequence.pad_sequences(Tokenised_Questions_Data , maxlen=Max_Input_Length , padding='post')
Padded_Answers = preprocessing.sequence.pad_sequences(Tokenised_Answers_Data , maxlen=Max_Output_Length , padding='post')

Encoder_Input = np.array(Padded_Questions)
Decoder_Input = np.array(Padded_Answers)
Decoder_Output = np.array(utils.to_categorical(Padded_Answers, VOCAB_SIZE))
print (Encoder_Input.shape)
print (Decoder_Output.shape)


# Creating an Encoder-Decoder Model
def EncoderDecoder_Model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
	model = Sequential()
	model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
	model.add(LSTM(n_units))
	model.add(RepeatVector(tar_timesteps))
	model.add(LSTM(n_units, return_sequences=True))
	model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
	return model

'''
def EncoderDecoder_Model1(VOCAB_SIZE):
	encoder_inputs = tf.keras.layers.Input(shape=( None , ))
	encoder_embedding = tf.keras.layers.Embedding( VOCAB_SIZE, 200 , mask_zero=True ) (encoder_inputs)
	encoder_outputs , state_h , state_c = tf.keras.layers.LSTM( 200 , return_state=True )( encoder_embedding )
	encoder_states = [ state_h , state_c ]

	decoder_inputs = tf.keras.layers.Input(shape=( None ,  ))
	decoder_embedding = tf.keras.layers.Embedding( VOCAB_SIZE, 200 , mask_zero=True) (decoder_inputs)
	decoder_lstm = tf.keras.layers.LSTM( 200 , return_state=True , return_sequences=True )
	decoder_outputs , _ , _ = decoder_lstm ( decoder_embedding , initial_state=encoder_states )
	decoder_dense = tf.keras.layers.Dense( VOCAB_SIZE , activation='softmax') 
	output = decoder_dense ( decoder_outputs )

	model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output )
	model.compile(optimizer='adam', loss='categorical_crossentropy')
	return model
'''

# +1 is due to pad character at the end as full stop
model = EncoderDecoder_Model(VOCAB_SIZE, VOCAB_SIZE, Max_Input_Length, Max_Output_Length, 256)
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.summary()
model.fit([Encoder_Input,Decoder_Input], Decoder_Output, batch_size=5, epochs=20)
