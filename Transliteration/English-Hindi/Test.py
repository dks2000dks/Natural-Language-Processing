import pandas as pd
import numpy as np
import string
import seaborn as sns
import matplotlib.pyplot as plt
import keras
from sklearn.model_selection import train_test_split
import operator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import nltk as nlp
import re
from nltk.tokenize import word_tokenize 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import xml.etree.ElementTree as ET
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential,load_model
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint
from nltk.translate.bleu_score import corpus_bleu


"""
Languages Data
"""
# Setting English Characters
Eng_Alphabets = string.ascii_uppercase
Pad_Char = '-PAD-'

# Creating a Index Map for English Characters
English_IndexMap = {Pad_Char: 0}
for index, alpha in enumerate(Eng_Alphabets):
    English_IndexMap[alpha] = index+1

# Setting Hindi Characters
Hindi_Alphabets = [chr(alpha) for alpha in range(2304, 2432)]
Pad_Char = '-PAD-'
Hindi_Alphabet_size = len(Hindi_Alphabets)

# Creating a Index Map for Hindi Characters
Hindi_IndexMap = {Pad_Char: 0}
for index, alpha in enumerate(Hindi_Alphabets):
    Hindi_IndexMap[alpha] = index+1

print ('......................Languages Details.......................')
Hindi_Chars = list(Hindi_IndexMap.keys())
print ("Hindi Characters",Hindi_Chars)
print ()
print ("Hindi Characters Mapping",Hindi_IndexMap)
Length_Hindi_Chars = len(Hindi_Chars)
print()


English_Chars = list(English_IndexMap.keys())
print ("English Characters",English_Chars)
print ()
print ("English Characters Mapping",English_IndexMap)
Length_English_Chars = len(English_Chars)
print ()

"""
Processing Data
"""
# Pre-Processing Data Functions
Non_Eng_Letters_Regex = re.compile('[^a-zA-Z ]')

# Remove all English Non-Letters
def cleanEnglishVocab(line):
    line = line.replace('-', ' ').replace(',', ' ').upper()
    line = Non_Eng_Letters_Regex.sub('', line)
    return line.split()

# Remove all Hindi Non-Letters
def cleanHindiVocab(line):
    line = line.replace('-', ' ').replace(',', ' ')
    cleaned_line = ''
    for char in line:
        if char in Hindi_IndexMap or char == ' ':
            cleaned_line += char
    return cleaned_line.split()

def WordOneHotRep(Word,MapIndex):
	'''
	One Hot Representation of Words
	'''
	OneHotRep = np.zeros((len(Word)+1,1,len(MapIndex)))
	for Letter_Index,Letter in enumerate(Word):
		RepPos = MapIndex[Letter]
		OneHotRep[Letter_Index][0][RepPos] = 1
	Pad_Pos = MapIndex[Pad_Char]
	OneHotRep[Letter_Index+1][0][Pad_Pos] = 1
	return Word,OneHotRep

def IndexRepresentation(Word,MapIndex):
	'''
	Index Representation of Words
	'''
	IndexRep = np.zeros((len(Word)+1,1))
	for Letter_Index,Letter in enumerate(Word):
		RepPos = MapIndex[Letter]
		IndexRep[Letter_Index][0] = RepPos
	Pad_Pos = MapIndex[Pad_Char]
	IndexRep[Letter_Index+1][0] = Pad_Pos
	
	IndexRep = IndexRep.flatten().astype(int)
	return Word,list(IndexRep)

"""
Encoder Module
"""
def Encode_Words(Data,MapIndex,Max):
	'''
	Takes Tokenizer, a dictionary(not datatype) in which all words in the dataset are present.
	It assigns the corresponding index number as indication of that word.
	'''
	Encoded_Data = np.zeros((len(Data),Max+1))
	r = 0
	for d in Data:
		data = IndexRepresentation(d,MapIndex)[1]
		i = len(data)
		for i in range(i,Max+1):
			data.append(0)
			i = i+1
		Encoded_Data[r] = np.array(data)
		r = r+1
		
	return Encoded_Data.astype(int)

"""
Encoding Function for One Hot Encoded Representation of Target Sequence
"""
def Encode_Output(sequences, vocab_size):
	ylist = []
	for sequence in sequences:
		encoded = to_categorical(sequence, num_classes=vocab_size)
		ylist.append(encoded)
	y = np.array(ylist)
	y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
	return y

model = load_model("EnglishHindi.h5")

def Predict_Sequence(model, List, source,Loss=True):
	prediction = model.predict(source, verbose=0)[0]
	integers = [np.argmax(vector) for vector in prediction]
	target = []
	for i in integers:
		word = List[i]
		if Loss == True:
			if word is None:
				break
			target.append(word)
		else:
			if word is None:
				break
			elif word != '-PAD-':
				target.append(word)
				
	return ' '.join(target)

# Evaluate the BLEU Skill of the Model
def Evaluate_Model(model, List, sources, lang_source, targets):
	Actual, Predicted = [], []
	for i, source in enumerate(sources):
		# Translate Encoded Source Text
		source = source.reshape((1, source.shape[0]))
		Translation = Predict_Sequence(model, List, source,Loss=True)
		X, Y = lang_source[i], targets[i]
		if i%10 == 0:
			print('i=[%i], Input=[%s], Actual=[%s], Predicted=[%s]' % (i, X, Y, Translation))
		Actual.append([Y.split()])
		Predicted.append(Translation.split())
	# Calculate BLEU Score
	print('BLEU-1: %f' % corpus_bleu(Actual, Predicted, weights=(1.0, 0, 0, 0)))
	print('BLEU-2: %f' % corpus_bleu(Actual, Predicted, weights=(0.5, 0.5, 0, 0)))
	print('BLEU-3: %f' % corpus_bleu(Actual, Predicted, weights=(0.3, 0.3, 0.3, 0)))
	print('BLEU-4: %f' % corpus_bleu(Actual, Predicted, weights=(0.25, 0.25, 0.25, 0.25)))

	
""""Testing the Model"""
Max_Hindi_Word = 10
Test = input('Enter a Hindi Word ')
T = cleanHindiVocab(Test)[0]
Test_T = cleanHindiVocab(Test)
W,R = IndexRepresentation(T,Hindi_IndexMap)
print (W)
print (R)
En = Encode_Words(Test_T,Hindi_IndexMap,Max_Hindi_Word)
Prediction = Predict_Sequence(model,English_Chars,En,Loss=False)
print ("Predicted Translation for the Word %s is %s" % (W,Prediction))


Test = input('Enter a Hindi Sentence ')
T = cleanHindiVocab(Test)
Output = ""
Output_Words = []

for t in T:
	Input = []
	Input.append(t)
	
	W,R = IndexRepresentation(t,Hindi_IndexMap)
	En = Encode_Words(Input,Hindi_IndexMap,Max_Hindi_Word)
	Prediction = Predict_Sequence(model,English_Chars,En,Loss=False)
	Output_Words.append(Prediction)
	
Prediction = Output.join(Output_Words)
print ("Predicted Translation for the Word %s is %s" % (Test,Prediction))
