import numpy as np
import matplotlib.pyplot as plt
import csv

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import KFold, learning_curve


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

#transform textual data to numeric
text_wordCnt_train=wordCounter.fit_transform(tweets_train)
text_wordCnt_test=wordCounter.transform(tweets_test)

#VALIDATION PHASE with kfold (adjusting alpha parameter)
kf = KFold(n_splits=3, shuffle=True, random_state=42)

alpha=[1.0, 0.9, 0.8, 0.7, 0.5, 0.4, 0.3, 0.1]
accuracy_scores = []

for alpha_value in alpha:
    accuracy = 0
    for train_index, test_index in kf.split(tweets_train):
        x_train, x_test = wordCounter.fit_transform(tweets_train[train_index]), wordCounter.transform(tweets_train[test_index])
        y_train, y_test = targets_train[train_index], targets_train[test_index]

        mnb=MultinomialNB(alpha=alpha_value)
        mnb = mnb.fit(x_train, y_train)
        pred = mnb.predict(x_test)
        accuracy = accuracy + accuracy_score(y_test, pred)
    
    accuracy = accuracy/3
    accuracy_scores.append(accuracy)

best_index = np.array(accuracy_scores).argmax()
best_alpha = alpha[best_index]

#TRAINING AND TESTING
print("Training model with alpha: "+ str(best_alpha))
model = MultinomialNB(best_alpha)
model = model.fit(text_wordCnt_train, targets_train)

print(model.score(text_wordCnt_train, targets_train))

pred = model.predict(text_wordCnt_test)

print(classification_report(targets_test,pred))

finalLabels = lblEncoder.inverse_transform(pred)

#plotting learning curve
model = MultinomialNB(best_alpha)
train_sizes, train_scores, val_scores = learning_curve(model, text_wordCnt_train, targets_train, scoring='accuracy', cv=3)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
val_scores_mean = np.mean(val_scores, axis=1)
val_scores_std = np.std(val_scores, axis=1)


# Plot the mean  for the training scores
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")

# Plot the  std for the training scores
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color="r")

# Plot the mean  for the validation scores
plt.plot(train_sizes, val_scores_mean, 'o-', color="g", label="Cross-validation score")

# Plot the std for the validation scores
plt.fill_between(train_sizes, val_scores_mean - val_scores_std, val_scores_mean + val_scores_std, alpha=0.1, color="g")
plt.ylim(0.05,1.3)             # set bottom and top limits for y axis
plt.legend()
plt.show()

#write results into file
f = open("test-pred.txt", "w+")
for label in finalLabels:
    f.write(label+"\n")

f.close()