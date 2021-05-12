import numpy as np
import matplotlib.pyplot as plt
import nltk
from textblob import TextBlob
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')

class Details():
	def __init__(self,Data):
		self.Data = Data

	def Properties(self):
		S = " ".join(self.Data)
		AllWords = S.split()
		UniqueWords = np.unique(AllWords)
		
		MaxLength = 0
		for l in self.Data:
			MaxLength = max(MaxLength, len(l.split()))
			
		return MaxLength, UniqueWords

class PreProcesser():
    def __init__(self):
        None
        
    def toLower(self,Data):
        Output = []
        for s in Data:
            l = s.split()
            Output.append(" ".join(list(map(lambda x: x.lower(), l))))
        return Output
        
    def removePunctuation(self,Data):
        Output = []
        for s in Data:
            l = s.split()
            Output.append(" ".join(list(map(lambda x: x.replace('[^\w\s]',''), l))))
        return Output
        
    def removeStopWords(self,Data):
        Output = []
        for s in Data:
            l = s.split()
            s = " ".join(list(map(lambda x: x if x not in stop else "", l)))
            l = " ".join(s.split())
            Output.append(l)
        return Output
        
    def mostCommonWords(self,Data,N):
        Total = " ".join(Data)
        Words = Total.split()
        UniqueWords, Freqs = np.unique(Words, return_counts=True)
        Map = dict(zip(UniqueWords, Freqs))
        Map = {k: v for k, v in sorted(Map.items(), key=lambda item: -item[1])}
        
        Vocab = list(Map.keys())[:N]
        Output = []
        for s in Data:
            l = s.split()
            l = " ".join(list(map(lambda x: x if x in Vocab else "", l)))
            l = " ".join(l.split())
            Output.append(l)
        return Output
        
    def correctSpelling(self,Data):
        """
        Not we efficient
        """
        Output = []
        for s in Data:
            l = s.split()
            s = " ".join(list(map(lambda x: x if x not in stop else "", l)))
            l = " ".join(s.split())
            Output.append(l)
        return Output
        
    def performLemmatization(self,Data):
        lemma = nltk.WordNetLemmatizer()
        
        Output = []
        for s in Data:
            l = s.split()
            s = " ".join(list(map(lambda x: lemma.lemmatize(x), l)))
            l = " ".join(s.split())
            Output.append(l)
        return Output
        
        
    def Preprocess(self,Data):
        ProcessedData = Data
        
        # Removing Puntuations
        ProcessedData = self.removePunctuation(ProcessedData)
        
        # Removing StopWords
        ProcessedData = self.removeStopWords(ProcessedData)
        
        # Converting text into lower characters
        ProcessedData = self.toLower(ProcessedData)
        
        # Lemmatization
        ProcessedData = self.performLemmatization(ProcessedData)
        
        # Taking only Most Common Words
        ProcessedData = self.mostCommonWords(ProcessedData,10000)
        
        return ProcessedData
        
    def fit(self, Data):
        Total = " ".join(Data)
        Words = Total.split()
        
        self.UniqueWords = np.unique(Words)
        self.Word2Index = dict(zip(self.UniqueWords, np.arange(1, self.UniqueWords.shape[0] + 1)))
        self.Word2Index[""] = 0
        self.Index2Word = {v:k for k, v in self.Word2Index.items()}
        
        return self.Word2Index, self.Index2Word
        
    def MapTokens(self, Data, MaxLength):
        N = len(Data)
            
        Output = np.zeros((N,MaxLength))
        for i in range(len(Data)):
            l = Data[i].split()
            v = list(map(lambda x: self.Word2Index[x] if x in self.UniqueWords else 0, l))
            Output[i,:len(v)] = v
            
        return Output
