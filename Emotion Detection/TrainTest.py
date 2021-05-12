# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import nltk
from textblob import TextBlob
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as tr
from torchsummary import summary
from torch.utils.data import DataLoader
device = tr.device("cuda" if tr.cuda.is_available() else "cpu")
print ("Device:", device)
print ()
import warnings
warnings.filterwarnings("ignore")


# Importing Functions from Files
from Functions import Details, PreProcesser
from Model import TextClassificationModel, NLPDataset
print ()

# Importing Dataset
def Extract(File):
	X_Data = []
	y_Data = []
	f = open(File)
	Lines = f.readlines()
	for line in Lines:
		X,y = line.split(";")
		y = y.strip()
		X_Data.append(X)
		y_Data.append(y)
	return X_Data, y_Data
	
X_Train, y_Train = Extract("Dataset/train.txt")
X_Val, y_Val = Extract("Dataset/val.txt")
X_Test, y_Test = Extract("Dataset/test.txt")
print ("----------- Train, Valid and Test Data -----------")
print ("Train Data Sizes:", len(X_Train))
print ("Valid Data Sizes:", len(X_Val))
print ("Test Data Sizes:", len(X_Test))
print ()


# Details of Data
Describe = Details(X_Train)
MaxLength, UniquePhrases = Describe.Properties()
print ("----------- Properties Data before Processing -----------")
print ("MaxLength:", MaxLength)
print ("No.of Unique Phrases:", len(UniquePhrases))
print ()


# Preprocess Data
P = PreProcesser()
X_Train = P.Preprocess(X_Train)
X_Val = P.Preprocess(X_Val)
X_Test = P.Preprocess(X_Test)
print ("----------- Properties Data after Processing -----------")
Describe = Details(X_Train)
MaxLength, UniquePhrases = Describe.Properties()
print ("MaxLength:", MaxLength)
print ("No.of Unique Phrases:", len(UniquePhrases))
print ()


# Mapping Data
Word2Index, Index2Word = P.fit(X_Train)
X_Train = P.MapTokens(X_Train,MaxLength).astype(int)
X_Val = P.MapTokens(X_Val,MaxLength).astype(int)
X_Test = P.MapTokens(X_Test,MaxLength).astype(int)


# Data
Classes = np.unique(y_Train)
NumClasses = len(Classes)
ClassMap = dict(zip(Classes,np.arange(NumClasses)))
print ("ClassMap:", ClassMap)
print ()

y_Train = list(map(lambda x: ClassMap[x], y_Train))
y_Val = list(map(lambda x: ClassMap[x], y_Val))
y_Test = list(map(lambda x: ClassMap[x], y_Test))

y_Train = np.eye(NumClasses)[np.array(y_Train)]
y_Val = np.eye(NumClasses)[y_Val]
y_Test = np.eye(NumClasses)[y_Test]

print ("Datasets Shapes after Processing:")
print ("Train Data Shape:", X_Train.shape,y_Train.shape)
print ("Validation Data Shape:", X_Val.shape,y_Val.shape)
print ("Test Data Shape:", X_Test.shape,y_Test.shape)
print ()


# Model
EmbeddingDims = 256
RNN_Units = 256
ClassificationModel = TextClassificationModel(len(UniquePhrases)+1, EmbeddingDims, RNN_Units, NumClasses, device).to(device)
print (ClassificationModel)


# DataLoader
batch_size = 32
TrainDataLoader = DataLoader(NLPDataset(X_Train,y_Train),batch_size,shuffle=True)
ValDataLoader = DataLoader(NLPDataset(X_Val,y_Val),batch_size,shuffle=True)
TestDataLoader = DataLoader(NLPDataset(X_Test,y_Test),batch_size,shuffle=True)


# Training
Epochs = 10
LossFunction = nn.CrossEntropyLoss()
Optimizer = optim.Adam(ClassificationModel.parameters())
for epoch in range(Epochs):
    Loss = []
    for i,data in enumerate(TrainDataLoader,0):
        inputs,labels = data[0].to(device), data[1].to(device)
        Optimizer.zero_grad()

        outputs = ClassificationModel(inputs)
        loss = tr.sum(tr.multiply(-tr.log10(outputs),labels))

        loss.backward()
        Optimizer.step()

        Loss.append(loss.item())
    print ("Epoch-" + str(epoch) + " "*7 + "Loss: " + str(np.mean(Loss)))