import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

#in the tweets data, the last column is available for a handful of tweets and does not serve as useful information -> remove it with arange(0,4)
tweets_train = np.genfromtxt("tweets-train-data.csv", delimiter=',', dtype=None, usecols=np.arange(0,1), encoding=None)
targets_train = np.genfromtxt("tweets-train-targets.csv", delimiter='\n', dtype=None, encoding=None)

tweets_test = np.genfromtxt("tweets-test-data.csv", delimiter=',', dtype=None, usecols=np.arange(0,1), encoding=None)
targets_test = np.genfromtxt("tweets-test-targets.csv", delimiter='\n', dtype=None, encoding=None)

#transform string labels into numeric labels for classification
lblEncoder = LabelEncoder()
targets_train = lblEncoder.fit_transform(targets_train) 
targets_test = lblEncoder.fit_transform(targets_test)

wordCounter = CountVectorizer()

text_wordCnt_train=wordCounter.fit_transform(tweets_train)
text_wordCnt_test=wordCounter.fit_transform(tweets_test)

print(text_wordCnt_train.shape)
print(text_wordCnt_train[0])

model = MultinomialNB()
model = model.fit(text_wordCnt_train, targets_train)

print(model.score(text_wordCnt_train, targets_train))
