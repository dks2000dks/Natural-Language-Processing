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
from keras.models import Sequential
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
    


"""
Creating a Data Loader
"""
class TransLiterationDataLoader():
	
	def __init__(self, filename):
		self.Eng_Words, self.Hindi_Words = self.readXmlDataset(filename, cleanHindiVocab)
		self.shuffle_indices = list(range(len(self.Eng_Words)))
		np.random.shuffle(self.shuffle_indices)
		self.shuffle_start_index = 0
    
    # Length of Dataset    
	def __len__(self):
		return len(self.Eng_Words)
    
	# Specific Item of Dataset    
	def __getitem__(self, idx):
		return self.Eng_Words[idx], self.Hindi_Words[idx]
    
	# Reading Dataset    
	def readXmlDataset(self, filename, lang_vocab_cleaner):
		transliterationCorpus = ET.parse(filename).getroot()
		English_Words = []
		Hindi_Words = []
		
		# Cleaning Data
		for line in transliterationCorpus:
			wordlist1 = cleanEnglishVocab(line[0].text)
			wordlist2 = lang_vocab_cleaner(line[1].text)

            # Skip Noisy Data i.e is inconsistant Data
			if len(wordlist1) != len(wordlist2):
				print('Skipping: ', line[0].text, ' - ', line[1].text)
				continue
			
			# Each English Words and Corresponding Hindi Words is added to list
			for word in wordlist1:
				English_Words.append(word)
			for word in wordlist2:
				Hindi_Words.append(word)

		return English_Words, Hindi_Words
    
    # Getting Random Index of Dataset    
	def get_random_sample(self):
		return self.__getitem__(np.random.randint(len(self.Eng_Words)))
    
	def Get_Batch_from_Array(self, batch_size, array):
		end = self.shuffle_start_index + batch_size
		batch = []
		if end >= len(array):
			batch = [array[i] for i in self.shuffle_indices[0:end%len(array)]]
			end = len(array)
		return batch + [array[i] for i in self.shuffle_indices[self.shuffle_start_index : end]]
    
	def Get_Batch(self, batch_size, postprocess = True):
		Eng_Batch = self.Get_Batch_from_Array(batch_size, self.Eng_Words)
		Hindi_Batch = self.Get_Batch_from_Array(batch_size, self.Hindi_Words)
		self.shuffle_start_index += batch_size + 1
        
        # Reshuffle if 1 epoch is complete
		if self.shuffle_start_index >= len(self.Eng_Words):
			np.random.shuffle(self.shuffle_indices)
			self.shuffle_start_index = 0
            
		return Eng_Batch, Hindi_Batch
		
	def Get_Dataset(self,percentage,Shuffle):
		end = int(percentage*(len(self.Eng_Words))/100) - 1
		if Shuffle == True:
				English_Dataset = [] + [self.Eng_Words[i] for i in self.shuffle_indices[self.shuffle_start_index : end]]
				Hindi_Dataset = [] + [self.Hindi_Words[i] for i in self.shuffle_indices[self.shuffle_start_index : end]]
		else:
			English_Dataset = self.Eng_Words
			Hindi_Dataset = self.Hindi_Words
			
		return English_Dataset,Hindi_Dataset 


"""
Importing Data
"""
TrainDataloader = TransLiterationDataLoader('NEWS2012-Training-EnHi-13937.xml')
TestDataloader = TransLiterationDataLoader('NEWS2012-Ref-EnHi-1000.xml')
print ()
print ("______________________Data Loader Completed. Data is Loaded___________________")
print ()

# Fun Tasks

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


W,R = IndexRepresentation('KRISHNASRIKAR',English_IndexMap)
print (W)
print (R)


# Extracting Dataset from Data Loader
TrainEnglish,TrainHindi = TrainDataloader.Get_Dataset(40,Shuffle=False)
TestEnglish,TestHindi = TestDataloader.Get_Dataset(40,Shuffle=False)

"""
# Printing Example Data
print (TrainEnglish[0],TrainHindi[0])
"""


# Setting Dataset, Train Dataset and Test Dataset
Train_Shape = len(TrainEnglish)
Test_Shape = len(TestEnglish)
print ('.................Training and Test Details....................')
print ("Shape of Training Dataset",Train_Shape)
print ("Shape of Test Dataset",Test_Shape)
print ()

# Combining Datasets to form a Total Dataset
English_Data = TrainEnglish + TestEnglish
Hindi_Data = TrainHindi + TestHindi
"""
# English_Data == Entire English Dataset
# Hindi_Data == Entire Hindi Dataset

# TrainEnglish = English Dataset for Training
# TestEnglish = English Dataset for Testing

# TrainHindi = Hindi Dataset for Training
# TestHindi = Hindi Dataset for Testing
"""

"""
# Printing Example Data
print (English_Data[1000],Hindi_Data[1000])
"""

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


def Vocabulary_Estimator(Data):
	D = list(np.unique(np.array(Data)))
	Vocab_Length = len(D)
	Max_Vocab_Length = len(max(D,key=len))
	
	return Vocab_Length,Max_Vocab_Length
	
print ('...................English Data...............................')
English_Vocab = Vocabulary_Estimator(English_Data)[0] + 1
Max_English_Word = Vocabulary_Estimator(English_Data)[1]
print ("Maximum English Word Length",Max_English_Word)
print ("English Vocabulary",English_Vocab)
print ()

print ('..................Hindi Data..................................')
Hindi_Vocab = Vocabulary_Estimator(Hindi_Data)[0] + 1
Max_Hindi_Word = Vocabulary_Estimator(Hindi_Data)[1]
print ("Maximum Hindi Word Length",Max_Hindi_Word)
print ("Hindi Vocabulary",Hindi_Vocab)
print ()	

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
	
TrainX = Encode_Words(TrainHindi,Hindi_IndexMap,Max_Hindi_Word)
TestX = Encode_Words(TestHindi,Hindi_IndexMap,Max_Hindi_Word)	

TrainY = Encode_Words(TrainEnglish,English_IndexMap,Max_English_Word)
TestY = Encode_Words(TestEnglish,English_IndexMap,Max_English_Word)
TrainY = Encode_Output(TrainY, Length_English_Chars)
TestY = Encode_Output(TestY, Length_English_Chars)

	
# Unmodified Functions
"""	
def Create_Tokenizer(Data):
	# Creating a Tokenizer to Split Data.
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(Data)
	return tokenizer
	
def max_length(lines):
	return max(len(line.split()) for line in lines)

print ('...................English Data...............................')
English_Tokenizer = Create_Tokenizer(English_Data)
English_Vocab = len(English_Tokenizer.word_index) + 1
Max_English_Word = max_length(English_Data)
print ("Maximum English Word Length",Max_English_Word)
print ("English Vocabulary",English_Vocab)
Vocabulary_Estimator(English_Data)
print ()

print ('..................Hindi Data..................................')
Hindi_Tokenizer = Create_Tokenizer(Hindi_Data)
Hindi_Vocab = len(Hindi_Tokenizer.word_index) + 1
Max_Hindi_Word = max_length(Hindi_Data)
print ("Maximum Hindi Word Length",Max_Hindi_Word)
print ("Hindi Vocabulary",Hindi_Vocab)
print ()


'''
Encoding Function
'''
def Encode_Sequences(tokenizer, length, lines):
	'''
	Takes Tokenizer, a dictionary(not datatype) in which all words in the dataset are present.
	It assigns the corresponding index number as indication of that word.
	'''
	# integer encode sequences
	X = tokenizer.texts_to_sequences(lines)
	# pad sequences with 0 values
	X = pad_sequences(X, maxlen=length, padding='post')
	return X


'''
Encoding Function for One Hot Encoded Representation of Target Sequence
'''
def Encode_Output(sequences, vocab_size):
	ylist = list()
	for sequence in sequences:
		encoded = to_categorical(sequence, num_classes=vocab_size)
		ylist.append(encoded)
	y = np.array(ylist)
	y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
	return y


# Prepare Data for Training and Testing
TrainX = Encode_Sequences(Hindi_Tokenizer, Max_Hindi_Word, TrainHindi)
TrainY = Encode_Sequences(English_Tokenizer, Max_English_Word, TrainEnglish)
TrainY = Encode_Output(TrainY, English_Vocab)

TestX = Encode_Sequences(Hindi_Tokenizer, Max_Hindi_Word, TestHindi)
TestY = Encode_Sequences(English_Tokenizer, Max_English_Word, TestEnglish)
TestY = Encode_Output(TestY, English_Vocab)
"""


# Creating an Encoder-Decoder Model
def EncoderDecoder_Model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
	model = Sequential()
	model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
	model.add(LSTM(n_units))
	model.add(RepeatVector(tar_timesteps))
	model.add(LSTM(n_units, return_sequences=True))
	model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
	return model


# +1 is due to pad character at the end as full stop
model = EncoderDecoder_Model(Hindi_Vocab, Length_English_Chars, Max_Hindi_Word+1, Max_English_Word+1, 256)
model.compile(optimizer='adam', loss='categorical_crossentropy')
print(model.summary())

plot_model(model, to_file='Model.png', show_shapes=True)
model.fit(TrainX, TrainY, epochs=10, batch_size=64, validation_data=(TestX, TestY), verbose=2)

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
	

print('------------------------------Train----------------------------')
Evaluate_Model(model, English_Chars, TrainX[0::100], TrainHindi[0::100], TrainEnglish[0::100])
print ()

print('------------------------------Test-----------------------------')
Evaluate_Model(model, English_Chars, TestX[0::10], TestHindi[0::10], TestEnglish[0::10])
print ()
print ("--------------------------------------------------------------")
print ()

	
""""Testing the Model"""
Test = input('Enter a Hindi Word ')
T = cleanHindiVocab(Test)[0]
Test_T = cleanHindiVocab(Test)
W,R = IndexRepresentation(T,Hindi_IndexMap)
print (W)
print (R)
En = Encode_Words(Test_T,Hindi_IndexMap,Max_Hindi_Word)
Prediction = Predict_Sequence(model,English_Chars,En,Loss=False)
print ("Predicted Translation for the Word %s is %s" % (W,Prediction))
