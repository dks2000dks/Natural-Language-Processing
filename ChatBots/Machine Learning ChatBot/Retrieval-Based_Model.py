import pandas as pd
import numpy as np
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
from sklearn.metrics.pairwise import cosine_similarity
import random
import string
from gtts import gTTS
from io import BytesIO
import os
from playsound import playsound
import warnings
warnings.filterwarnings('ignore')
print ("------------------------Imported Libraries--------------------")
print()

# Importing Text File
f = open('ML_Data.txt', 'r', errors ='ignore')
data = f.read()

# Converting Data to lowercase
data = data.lower()

# Listening to Data
'''
tts = gTTS(text=data[0:200], lang='en')
tts.save("Bot.mp3")
playsound('Bot.mp3')
'''

# Preprocessing Data Functions
def Remove_Punctuation(text):
    '''A Function for Removing Punctuation'''
    import string
    # Replacing the Punctuations with no space, which in effect deletes the punctuation marks 
    translator = str.maketrans('', '', string.punctuation)
    # Return the text stripped of punctuation marks
    return text.translate(translator)
    
def Lemmatizer(data):
	''' Using Lemmatizer '''
	data = Remove_Punctuation(data)
	lemmer = nlp.stem.WordNetLemmatizer()
	return [lemmer.lemmatize(d) for d in data]
    

# Tokenising Data into Sentences and Words
List_Sentences = nlp.sent_tokenize(data)
List_Words = nlp.word_tokenize(data)
print ("No.of Sentences = ",len(List_Sentences))
print ("No.of Words = ",len(List_Words))
print()



# Keyword Matching for Greetings. Using the same concept as ELIZA
Greetings_Inputs = ['hello', 'hi', 'hey', 'greetings', 'whats up']
Greeetings_Outputs = ['hi', 'greetings', 'hi there welcome', 'hello', 'hey', 'whats up']

def Greetings(Sentence):
	for word in Sentence.split():
		if word.lower() in Greetings_Inputs:
			return random.choice(Greeetings_Outputs)
			
# Generating Response
def Bot_Response(Input):
	'''
	A Function response which searches the user’s utterance for one or more known keywords and returns one of several possible responses.
	If it doesn’t find the input matching any of the keywords, it returns a response:” I am sorry! I don’t understand you”
	'''
	Bot_Response = ''
	
	User_Input = List_Sentences
	User_Input.append(Input)
	TfidfVec = TfidfVectorizer(tokenizer=Lemmatizer, stop_words='english')
	
	tfidf = TfidfVec.fit_transform(User_Input)
	
	vals = cosine_similarity(tfidf[-1], tfidf)
	idx=vals.argsort()[0][-2]
	flat = vals.flatten()
	flat.sort()
	req_tfidf = flat[-2]
    
	if(req_tfidf==0):
		Bot_Response = Bot_response + "I am sorry! I don't understand you"		
		return Bot_Response
	else:
		Bot_Response = Bot_Response + List_Sentences[idx]
		return Bot_Response
        

# Start Conversation
print ("-------------------------------------Start Conversation--------------------------------")  
flag=True

Introduction = "My name is Chitti. I am a protype of Retrieval-Based ChatBot designed by Krishna Srikar Durbha. I will answer your queries about Machine Learning. If you want to exit, type Bye!"
print('Chitti: ' + Introduction)

tts = gTTS(text=Introduction, lang='en')
tts.save("Bot.mp3")
playsound('Bot.mp3')

while(flag==True):
	user_response = input('User: ')
	user_response=user_response.lower()
	if(user_response!='bye'):
		
		if(user_response=='thanks' or user_response=='thank you' ):
			flag=False
            
			Response = "You are Welcome"
			print('Chitti: ' + Response)
			tts = gTTS(text=Response, lang='en')
			tts.save("Bot.mp3")
			playsound('Bot.mp3')
     
            
		else:
			if(Greetings(user_response)!=None):
				Wishes = Greetings(user_response)
				print("Chitti: "+ Wishes)
                
				tts = gTTS(text=Wishes, lang='en')
				tts.save("Bot.mp3")
				playsound('Bot.mp3')
                
                
			else:
				print("Chitti: ", end="")
				Response = Bot_Response(user_response)
				print(Response)
                
				tts = gTTS(text=Response, lang='en')
				tts.save("Bot.mp3")
				playsound('Bot.mp3')
				
				
	else:
		flag=False
		Response = "Bye! Have a Good Day."
		print('Chitti: ' + Response)
		tts = gTTS(text=Response, lang='en')
		tts.save("Bot.mp3")
		playsound('Bot.mp3')
