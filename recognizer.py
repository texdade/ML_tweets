import numpy as np
import matplotlib.pyplot as plt
import csv

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

tweets_train = np.genfromtxt("tweets-train-data.csv", delimiter=',', dtype=None, usecols=np.arange(0,1), encoding=None)
targets_train = np.genfromtxt("tweets-train-targets.csv", delimiter='\n', dtype=None, encoding=None)

tweets_test = np.genfromtxt("tweets-test-data.csv", delimiter=',', dtype=None, usecols=np.arange(0,1), encoding=None)
targets_test = np.genfromtxt("tweets-test-targets.csv", delimiter='\n', dtype=None, encoding=None)

#transform string labels into numeric labels for classification
lblEncoder = LabelEncoder()
targets_train = lblEncoder.fit_transform(targets_train) 
targets_test = lblEncoder.fit_transform(targets_test)

#create count vectorized in order to count words
wordCounter = CountVectorizer()

#VALIDATION - choose between multinomialNB or LogisticRegression
x_train, x_test, y_train, y_test = train_test_split(tweets_train, targets_train, test_size=0.4, random_state=0)

#transform textual data to numeric
text_wordCnt_train=wordCounter.fit_transform(x_train)
text_wordCnt_test=wordCounter.transform(y_train)

logreg = LogisticRegression()

"""
logreg.fit(text_wordCnt_train, targets_train)
pred = logreg.predict(text_wordCnt_test)"""
print(logreg.score(text_wordCnt_train, targets_train))
print(classification_report(targets_test,pred))


#TRAINING AND TESTING
#transform textual data to numeric
text_wordCnt_train=wordCounter.fit_transform(tweets_train)
text_wordCnt_test=wordCounter.transform(tweets_test)

model = MultinomialNB()
model = model.fit(text_wordCnt_train, targets_train)

print(model.score(text_wordCnt_train, targets_train))

pred = model.predict(text_wordCnt_test)

print(classification_report(targets_test,pred))

finalLabels = lblEncoder.inverse_transform(pred)
"""
#write results into file
f = open("test-pred.txt", "w+")
for label in finalLabels:
    f.write(label+"\n")

f.close()"""