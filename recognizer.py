import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn import metrics

#in the tweets data, the last column is available for a handful of tweets and does not serve as useful information -> remove it with arange(0,4)
tweets_train = np.genfromtxt("tweets-train-data.csv", delimiter=',', dtype=None, usecols=np.arange(0,4), encoding=None)
targets_train = np.genfromtxt("tweets-train-targets.csv", delimiter='\n', dtype=None, encoding=None)

tweets_test = np.genfromtxt("tweets-test-data.csv", delimiter=',', dtype=None, usecols=np.arange(0,4), encoding=None)
targets_test = np.genfromtxt("tweets-test-targets.csv", delimiter='\n', dtype=None, encoding=None)

