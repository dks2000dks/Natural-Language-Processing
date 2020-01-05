import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import keras
from sklearn.model_selection import train_test_split


df = pd.read_csv('SPAM Text Message Data.csv',sep=',')															#Importing the .csv for data analysis
print (df.head())																			#Prints first 5 rows
print ("--------------------------------------------------------------------------")
print (df.shape)																			#Dimensions of the data
print (df.columns.values)																	#Names of columns in the data
print (df.info())																			#Describes data types of each column
print ("--------------------------------------------------------------------------")
print (df.describe())																		#Gives statitics of the data
print ("--------------------------------------------------------------------------")

X = df['Message'].values
y = df['Category'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

print ("Shape of Training Set is",X_train.shape)
print ("Shape of Test Set is",X_test.shape)
print ("--------------------------------------------------------------------------")

print ("Classes",np.unique(y)) 
print ("No.of unique words",len(np.unique(np.hstack(X))))
print ("--------------------------------------------------------------------------")

X_words = np.unique(np.hstack(X))
y_words = np.unique(y)

# print (X_words)
# print (y_words)

y_word_count = {}

for i in y_words:
	y_word_count[i] = 0
for j in y:
	y_word_count[j] += 1
	
print (y_word_count)
plt_ = sns.barplot(list(y_word_count.keys()), list(y_word_count.values()))
plt_.set_xticklabels(plt_.get_xticklabels(), rotation=90)
plt.show()
print ("--------------------------------------------------------------------------")

wordlength = [len(x) for x in X]
print ("Mean Word Length",np.mean(wordlength))
print ("Standard Deviation of Word Length",np.std(wordlength))
print ("--------------------------------------------------------------------------")

plt.boxplot(wordlength)
plt.show()

