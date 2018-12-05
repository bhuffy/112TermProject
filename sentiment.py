## Project: Email Text Analyzer Tool: Display

"""
Author: Bennett Huffman
Last Modified: 11/19/18
Course: 15-112
Section: H
"""

## IMPORTS

import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.tokenize import word_tokenize
from nltk.classify import ClassifierI
import string

import pickle
import codecs
import statistics

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

## Text Classifier (i.e. spam or not spam

shortPositive = codecs.open("practice/reviews/positive.txt","r", encoding='latin2').read()
shortNegative = codecs.open("practice/reviews/negative.txt","r", encoding='latin2').read()

documents = []
allWords = []
# J is adjective
allowedWordTypes = ["J"]

for p in shortPositive.split("\n"):
    documents.append((p, "pos"))
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos: #POS tags to tuples (word, POS)
        if w[1][0] in allowedWordTypes: #in POS, if first letter is a type of adjective, append to list
            allWords.append(w[0].lower())

for p in shortNegative.split("\n"):
    documents.append((p, "neg"))
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowedWordTypes:
            allWords.append(w[0].lower())
    
shortPosWords = word_tokenize(shortPositive)
shortNegWords = word_tokenize(shortNegative)

for word in shortPosWords:
    allWords.append(word.lower())

for word in shortNegWords:
    allWords.append(word.lower())

# creates object mapping most to least common words in key/value pair
allWords = nltk.FreqDist(allWords)
word_features = list(allWords.keys())[:5000] # gets the top 5000 words

# we want to get each word in a given document and determine, whether they're in the top 3000 words
def findFeatures(document):
    words = word_tokenize(document)
    features = dict()
    for w in word_features:
        features[w] = (w in words) # creates true/false if word in document mapping to list of features
    return features
    
featureSets = [(findFeatures(review), category) for (review, category) in documents]

# Now we want to train and test the features to the data;
trainingSet = featureSets[:10000]
testingSet = featureSets[10000:]

# Training is where we say here are the words, here is the data in the top 3000 list of all words; how many they appear in negative reviews and how many times they appear in positive reviews. If they appear significantly more in negative reviews the word is probably important to negative reviews and likewise for positive reviews.

# In the testing set, we feed the data the documents and don't tell it what category they are; we ask the machine to tell us and compare it to the known category we have to identify how accurate it is.

# We're going to use the Naive Bayes algorithm (works on strong independent assumptions for each feature, which is why it's "naive". But because the algorithm is so basic, we can scale it to massive proportions since it doesn't take much processing)

# posterior (the likelihood) = prior occurrences * likelihood / evidence

classifier = nltk.NaiveBayesClassifier.train(trainingSet)
print("Naive Bayes Algorithm Accuracy:", nltk.classify.accuracy(classifier, testingSet) * 100, "%")
classifier.show_most_informative_features(15)

## Testing Classifiers
# We want to run a bunch of different classifiers and use a voting system to determine the mean accuracy of the models. Each of these has their own default parameters and can be customized, but what we're trying to do is gather a list of accuracies and take a vote on each one, which will get us a slightly higher accuracy (throwing out unuseful ones) and this will create reliability. We'll also have a good score and certainty of that score in subsequent analyses. 

print("Original Naive Bayes Algorithm Accuracy:", nltk.classify.accuracy(classifier, testingSet) * 100, "%")
classifier.show_most_informative_features(15)

# MNB Classifier
MNBClassifier = SklearnClassifier(MultinomialNB())
MNBClassifier.train(trainingSet)
print("MNBClassifier Algorithm Accuracy:", nltk.classify.accuracy(MNBClassifier, testingSet) * 100, "%")

# GaussianNB Classifier
# GaussianNBClassifier = SklearnClassifier(GaussianNB())
# GaussianNBClassifier.train(trainingSet)
# print("GaussianNB Algorithm Accuracy:", nltk.classify.accuracy(GaussianNBClassifier, testingSet) * 100, "%")

# BernoulliNB Classifier
BernoulliNBClassifier = SklearnClassifier(BernoulliNB())
BernoulliNBClassifier.train(trainingSet)
print("BernoulliNB Algorithm Accuracy:", nltk.classify.accuracy(BernoulliNBClassifier, testingSet) * 100, "%")

# LogisticRegression Classifier
LogisticRegressionClassifier = SklearnClassifier(LogisticRegression())
LogisticRegressionClassifier.train(trainingSet)
print("LogisticRegression Algorithm Accuracy:", nltk.classify.accuracy(LogisticRegressionClassifier, testingSet) * 100, "%")

# SGDClassifier Classifier
SGDClassifierClassifier = SklearnClassifier(SGDClassifier())
SGDClassifierClassifier.train(trainingSet)
print("SGDClassifier Algorithm Accuracy:", nltk.classify.accuracy(SGDClassifierClassifier, testingSet) * 100, "%")

# SVC Classifier
# SVCClassifier = SklearnClassifier(SVC())
# SVCClassifier.train(trainingSet)
# print("SVC Algorithm Accuracy:", nltk.classify.accuracy(SVCClassifier, testingSet) * 100, "%")

# LinearSVC Classifier
LinearSVCClassifier = SklearnClassifier(LinearSVC())
LinearSVCClassifier.train(trainingSet)
print("LinearSVC Algorithm Accuracy:", nltk.classify.accuracy(LinearSVCClassifier, testingSet) * 100, "%")

# NuSVC Classifier
NuSVCClassifier = SklearnClassifier(NuSVC())
NuSVCClassifier.train(trainingSet)
print("NuSVC Algorithm Accuracy:", nltk.classify.accuracy(NuSVCClassifier, testingSet) * 100, "%")

## Voting System

#We'll pass a list of classifiers into our class
class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self.classifiers = classifiers
    
    def classify(self, features):
        votes = []
        for c in self.classifiers:
            v = c.classify(features)
            votes.append(v)
        return self.findMode(votes)
        
    #https://stackoverflow.com/questions/50959614/no-unique-mode-found-2-equally-common-values
    @staticmethod
    def findMode(lst):
        countTable = statistics._counts(lst)
        tableLen = len(countTable)
    
        if tableLen == 1:
            maxMode = statistics.mode(lst)
        else:
            newLst = []
            for i in range(tableLen):
                newLst.append(countTable[i][0])
            maxMode = max(newLst) # use the max value here
        return maxMode
        
    def confidence(self, features):
        votes = []
        for c in self.classifiers:
            v = c.classify(features)
            votes.append(v)
        choiceVotes = votes.count(self.findMode(votes)) # how many of most popular votes were in list
        conf = choiceVotes / len(votes)
        return conf
    
## Vote Algorithm Classifier

votedClassifier = VoteClassifier(classifier,
                                MNBClassifier,
                                BernoulliNBClassifier,
                                LogisticRegressionClassifier,
                                SGDClassifierClassifier,
                                SVCClassifier,
                                LinearSVCClassifier,
                                NuSVCClassifier)

print("votedClassifier Algorithm Accuracy:", nltk.classify.accuracy(votedClassifier, testingSet) * 100, "%")
print("Classification:", votedClassifier.classify(testingSet[0][0]), "Confidence:", votedClassifier.confidence(testingSet[0][0]) * 100, "%")
print("Classification:", votedClassifier.classify(testingSet[1][0]), "Confidence:", votedClassifier.confidence(testingSet[1][0]) * 100, "%")
print("Classification:", votedClassifier.classify(testingSet[2][0]), "Confidence:", votedClassifier.confidence(testingSet[2][0]) * 100, "%")
print("Classification:", votedClassifier.classify(testingSet[3][0]), "Confidence:", votedClassifier.confidence(testingSet[3][0]) * 100, "%")


"""
Naive Bayesian Classifier algorithm:
P(c|x) = (P(x|c) * P(c)) / P(x)

where:

P(c|x) is the posterior probability of class (c, target) given predictor (x, attributes).
P(c) is the prior probability of class.
P(x|c) is the likelihood which is the probability of predictor given class.
P(x) is the prior probability of predictor.
"""