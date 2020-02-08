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
import xmldataset as Dataset
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

# Setting English Characters
Eng_Alphabets = string.ascii_uppercase
Pad_Char = '-PAD-'

English_IndexMap = {Pad_Char: 0}
for index, alpha in enumerate(Eng_Alphabets):
    English_IndexMap[alpha] = index+1

# Setting Hindi Characters
Hindi_Alphabets = [chr(alpha) for alpha in range(2304, 2432)]
Pad_Char = '-PAD-'
Hindi_Alphabet_size = len(Hindi_Alphabets)

Hindi_IndexMap = {Pad_Char: 0}
for index, alpha in enumerate(Hindi_Alphabets):
    Hindi_IndexMap[alpha] = index+1


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
    


# Creating a Data Loader
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

		for line in transliterationCorpus:
			wordlist1 = cleanEnglishVocab(line[0].text)
			wordlist2 = lang_vocab_cleaner(line[1].text)

            # Skip noisy data
			if len(wordlist1) != len(wordlist2):
				print('Skipping: ', line[0].text, ' - ', line[1].text)
				continue

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
		
	def Get_Dataset(self,percentage):
		end = int(percentage*(len(self.Eng_Words))/100) - 1
		English_Dataset = [] + [self.Eng_Words[i] for i in self.shuffle_indices[self.shuffle_start_index : end]]
		Hindi_Dataset = [] + [self.Hindi_Words[i] for i in self.shuffle_indices[self.shuffle_start_index : end]]
		
		return English_Dataset,Hindi_Dataset 

TrainDataloader = TransLiterationDataLoader('NEWS2012-Training-EnHi-13937.xml')
TestDataloader = TransLiterationDataLoader('NEWS2012-Ref-EnHi-1000.xml')
print ()
print ("______________________Data Loader Completed___________________")
print ()

def WordOneHotRep(Word,MapIndex):
	"""
	One Hot Representation of Words
	"""
	OneHotRep = np.zeros((len(Word)+1,1,len(MapIndex)))
	for Letter_Index,Letter in enumerate(Word):
		RepPos = MapIndex[Letter]
		OneHotRep[Letter_Index][0][RepPos] = 1
	Pad_Pos = MapIndex[Pad_Char]
	OneHotRep[Letter_Index+1][0][Pad_Pos] = 1
	return Word,OneHotRep

def IndexRep(Word,MapIndex):
	"""
	Index Representation of Words
	"""
	IndexRep = np.zeros((len(Word)+1,1))
	for Letter_Index,Letter in enumerate(Word):
		RepPos = MapIndex[Letter]
		IndexRep[Letter_Index][0] = RepPos
	Pad_Pos = MapIndex[Pad_Char]
	IndexRep[Letter_Index+1][0] = Pad_Pos
	return Word,IndexRep

"""	
W,R = IndexRep('KRISHNASRIKAR',English_IndexMap)
print (W)
print (R)
"""

# Extracting Dataset
TrainEnglish,TrainHindi = TrainDataloader.Get_Dataset(100)
TestEnglish,TestHindi = TestDataloader.Get_Dataset(100)

Train_Shape = len(TrainEnglish)
Test_Shape = len(TestEnglish)
print ('.................Training and Test Details....................')
print ("Shape of Training Dataset",Train_Shape)
print ("Shape of Test Dataset",Test_Shape)
print ()

# Combining Datasets
English_Data = TrainEnglish + TestEnglish
Hindi_Data = TrainHindi + TestHindi

# Creating a Tokenizer to Split Data.
def Create_Tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

print ('...................English Data...............................')
English_Tokenizer = Create_Tokenizer(English_Data)
English_Vocab = len(English_Tokenizer.word_index) + 1
Max_English_Word = len(max(English_Data,key=len))
print ("Maximum English Word Length",Max_English_Word)
print ("English Vocabulary",English_Vocab)
print ()

print ('..................Hindi Data..................................')
Hindi_Tokenizer = Create_Tokenizer(Hindi_Data)
Hindi_Vocab = len(Hindi_Tokenizer.word_index) + 1
Max_Hindi_Word = len(max(Hindi_Data,key=len))
print ("Maximum Hindi Word Length",Max_Hindi_Word)
print ("Hindi Vocabulary",Hindi_Vocab)
print ()

def Encode_Sequences(tokenizer, length, lines):
	# integer encode sequences
	X = tokenizer.texts_to_sequences(lines)
	# pad sequences with 0 values
	X = pad_sequences(X, maxlen=length, padding='post')
	return X

# One Hot Encoded Representation of Target Sequence
def Encode_Output(sequences, vocab_size):
	ylist = list()
	for sequence in sequences:
		encoded = to_categorical(sequence, num_classes=vocab_size)
		ylist.append(encoded)
	y = np.array(ylist)
	y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
	return y

"""	
# Prepare Data for Training and Testing
TrainX = Encode_Sequences(Hindi_Tokenizer, Max_Hindi_Word, TrainHindi)
TrainY = Encode_Sequences(English_Tokenizer, Max_English_Word, TrainEnglish)
TrainY = Encode_Output(TrainY, English_Vocab)

TestX = Encode_Sequences(Hindi_Tokenizer, Max_Hindi_Word, TestHindi)
TestY = Encode_Sequences(English_Tokenizer, Max_English_Word, TestEnglish)
TestY = Encode_Output(TestY, English_Vocab)
"""
