# ML_tweets - Machine Learning course UniTN | Assignment 2
Recognize Hilary Clinton and Donald Trump from their tweets.

Training set instances: 4833
Test set instances: 1611
Baseline Accuracy: 0.50403


Data
-------
text : text of the tweet
datetime : the time of the tweet
retweet_count : the retweet count
favorite_count : the favourite count
place_full_name : the place the tweet was posted from


Targets
-------
who : str
'HC': Hilary Clinton
'DT': Donald Trump

# Preliminary operations
Before explaining how the predictor was created, I have to speak briefly of the dataset. Since
MultinomialNB was chosen as a learning algorithm, the only actual data that was used was
the text of the tweet. This is because the other fields of the dataset (namely date, retweets,
likes, city) are less useful and contain less relevant information, especially the date and city
ones.

# Learning algorithm and validation
Multinomial Naive Bayes is a classifier well suited for text classification. While other
classifiers, like Logistic Regression, were considered, MultinomialNB was chosen for its
precision (slightly higher than LogisticRegression) and high speed. Before feeding the
dataset to the classifier, though, some steps were required: first of all, all the targets for the
training and testing got transformed from the “HC/DT” format to a “0/1” format via
LabelEncoder. After that, CountVectorizer was used in order to transform the text of the
tweets in matrices containing the occurrences of each and every word, which are then given
to MultinomialNB.

MultinomialNB has a parameter “alpha” with a default value of 1.0, which is a smoothing
parameter useful in situations where the system has to predict who wrote a tweet, and this
contains one or more words which were to present in the training set. The optimization of
this parameter was done through k-fold cross-validation, splitting the training set in 3 folds.
The results are the following:
Alpha | Precision | Recall  |  F1-score
------------ | ------------ | ------------ | ------------
1.0 | 0.8712 | 0.8704 | 0.8704
0.9 | 0.8716 | 0.8708 | 0.8708
0.8 | 0.8718 | 0.8710 | 0.8710
0.7 | 0.8715 | 0.8708 | 0.8708
0.6 | 0.8717 | 0.8710 | 0.8710
0.5 | 0.8700 | 0.8694 | 0.8693
0.4 | 0.8683 | 0.8677 | 0.8677
0.3 | 0.8676 | 0.8671 | 0.8671
0.2 | 0.8677 | 0.8671 | 0.8671
0.1 | 0.8665 | 0.8659 | 0.8658

Thus, the model was then trained over the full training set using an alpha parameter equal to
0.8.

