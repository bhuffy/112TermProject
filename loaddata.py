## Project: Email Text Analyzer Tool: Load Data File

"""
Author: Bennett Huffman
Last Modified: 11/19/18
Course: 15-112
Section: H
"""

## IMPORTS

# general imports
import csv
import pandas
import enchant

#Move these to analyze file when trying to separate
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import wordnet, stopwords
import string

# clustering imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


## Classes

class Email(object):
    def __init__(self, data):
        self.dateTime = str(data['Date Time'])
        self.fromName = str(data['From Name'])
        self.fromEmail = str(data['From Email'])
        self.toEmail = str(data['To Email'])
        self.subject = str(data['Subject'])
        self.messageId = str(data['Message Id'])
        self.body = str(data['Body'])
      
## Create Email List

# CITATION: https://realpython.com/python-csv/
def createEmailList(path):
    emails = []
    # creates dictonary from values in each line of CSV
    with open(path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        start = True
        for row in csv_reader:
            emails.append(Email(row))
    return emails

#get all words in bodies and subject
def getAllWords(emails):
    allWords = []
    for email in emails:
        allWords.extend(word_tokenize(email.body))
        allWords.extend(word_tokenize(email.subject))
    return allWords

# returns list of all email bodies
def getEmailBodyList(emails):
    documents = []
    for email in emails:
        documents.append(email.body)
    return documents

# returns list of all email subject lines
def getEmailSubjectList(emails):
    documents = []
    for email in emails:
        documents.append(email.subject)
    return documents

#filter words such that they make sense
def filterWords(words):
    stopWords = set(stopwords.words("english")) # check if stop word
    d = enchant.Dict("en_US") # checks if in english dictionary
    filteredWords = []
    for word in words:
        if word.isalpha() and word not in stopWords and d.check(word):
            filteredWords.append(word.lower())
    return filteredWords

## Synonyms

#creates a dictionary of the synonyms for each word
def createSynDictionary(words):
    synDict = dict()
    for word in words:
            synDict[word] = findSynonyms(word, words)
    return synDict

# finds the synonyms of a given word via wordnet synsets
def findSynonyms(word, words):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            if l.name() in words:
                synonyms.add(l.name())
    return synonyms

# Adds the frequency of each word and its synonyms to the word
def addWordFrequency(text, synDict):
    for word in synDict:
        count = 0
        for synonym in synDict[word]:
            count += text.count(synonym)
        synDict[word] = (synDict[word], count)
    
#Give the number of times each word and it's synonyms appear in a text
def getSimDictionary(documents, words):
    synDict = createSynDictionary(words)
    addWordFrequency(text, synDict)
    return synDict
    
## Clustering for labels

def findKMeans(documents):
    # vectorize text
    print(documents[0])
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(documents)
    
    # start with 2 clusters
    true_k = 2
    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
    model.fit(X)
    
    print("Top terms per cluster:")
    order_centroids = model.cluster_centers_.argsort()[:, ::-1] # don't know what this does...?
    terms = vectorizer.get_feature_names()
    for i in range(true_k):
        print("Cluster %d:" % i),
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind]),
        print()
    
    print("\n")
    print("Prediction")
    
    Y = vectorizer.transform(["chrome browser to open."])
    prediction = model.predict(Y)
    print(prediction)
    
    Y = vectorizer.transform(["My cat is hungry."])
    prediction = model.predict(Y)
    print(prediction)

"""
1. take all bodies of emails
2. find nouns for clustering
3. find most frequent
3. find adjectives for classification of sentiment
4. for 

"""

## Sentiment Analysis

def plotFrequencyDist(words):
    words.plot(30,cumulative=False)

# Get each word and for each of them, create a dictionary whether the top 1000 words are in the document
def findFeatures(email):
    words = set(email)
    features = dict()
    for w in word_features:
        features[w] = (w in words) # creates true/false if word in document mapping to list of features
    return features

def analyze():
    # get emails from csv and filter them
    emails = createEmailList('data/emails.csv')
    allWords = getAllWords(emails)
    words = filterWords(allWords)
    
    # frequency distribution
    wordDist = nltk.FreqDist(words) # makes object mapping most to least common words in key/value pair
    print(wordDist.most_common(15)) # prints 15 most common words
    plotFrequencyDist(wordDist)
    
    # get top 1000 words
    wordFeatures = list(wordDist.keys())[:1000] # gets the top 1000 words
    print(wordFeatures)
    
    # similar words in dictionary
    documents = getEmailBodyList(emails)
    similarityDict = getSimDictionary(documents, words)
    print(similarityDict)
    
    # clustering
    findKMeans(documents)


## Exporting back to CSV

# Using CSV with pandas, might want later for csv export
# df = pandas.read_csv('data/Consumer_Complaints.csv', header=0)
# for row in df.iterrows():
#     print(row[1])

analyze()