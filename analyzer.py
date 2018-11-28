## Project: Email Text Analyzer Tool: Analyzer

"""
Author: Bennett Huffman
Last Modified: 11/19/18
Course: 15-112
Section: H
"""

## IMPORTS

# general imports for handling csv files
import csv
import pandas
import enchant

#Move these to analyze file when trying to separate
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import wordnet, stopwords
import string

# k-means imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# network diagram
import networkx as nx


## Email Pre-Processing & Word Filtering

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
            synonyms.add(l.name())
    return synonyms

# Adds the frequency of each word and its synonyms to the word
def addWordFrequency(document, synDict):
    for word in synDict:
        count = 0
        for synonym in synDict[word]:
            count += text.count(synonym)
        synDict[word] = (synDict[word], count)
    
#Give the number of times each word and it's synonyms appear in a text
def getSimDictionary(documents, words):
    synDict = createSynDictionary(words)
    addWordFrequency(documents[5], synDict)
    return synDict
    
## Clustering

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
    
    Y = vectorizer.transform(["College is great, please apply and come to Lawrence!."])
    prediction = model.predict(Y)
    print(prediction)
    
    Y = vectorizer.transform(["Bennett, congratulations on something, please visit us. We can do something amazing."])
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

def plotFrequencyDist(wordDist):
    wordDist.plot(30,cumulative=False)
    
def plotConnectionDist(words):
    df = pandas.DataFrame({ 'from':['A', 'B', 'C','A'], 'to':['D', 'A', 'E','C']})
    # G=nx.from_pandas_dataframe(df, 'from', 'to')
    # nx.draw(G, with_labels=True)
    # plt.show()

# Get each word and for each of them, create a dictionary whether the top 1000 words are in the document
def findFeatures(email):
    words = set(email)
    features = dict()
    for w in word_features:
        features[w] = (w in words) # creates true/false if word in document mapping to list of features
    return features

def analyze(filename, features):
    results = []
    # get emails from csv and filter them
    emails = createEmailList(filename)
    allWords = getAllWords(emails)
    words = filterWords(allWords)
    
    #frequency distribution
    wordDist = nltk.FreqDist(words) # makes object mapping most to least common words in key/value pair
    
    if features['labels'][0]:
        generateLabels()
    if features['freqdist'][0]:
    # frequency distribution
        wordDist.most_common(15)
        plotFrequencyDist(wordDist)
        plotConnectionDist(wordDist)
    if features['netdiagram'][0]:
        drawNetworkDiagram(canvas, data)
    if features['summary'][0]:
        drawSummarization(canvas, data)
    if features['sentiment'][0]:
        drawSentimentAnalysis(canvas, data)
    if features['exportCSV'][0]:
        drawExportCSVButton(canvas, data)
        
    # get top 1000 words
    wordFeatures = list(wordDist.keys())[:1000] # gets the top 1000 words
    print(wordFeatures)
    
    # similar words in dictionary
    documents = getEmailBodyList(emails)
    # similarityDict = getSimDictionary(documents, words)
    # print(similarityDict)
    
    # clustering
    findKMeans(documents)


## Exporting back to CSV

# Using CSV with pandas, might want later for csv export
# df = pandas.read_csv('data/Consumer_Complaints.csv', header=0)
# for row in df.iterrows():
#     print(row[1])


## Run Analyzer
features = {
    'labels': [False, "#f5f5f5"],
    'freqdist': [False, "#f5f5f5"],
    'netdiagram': [False, "#f5f5f5"],
    'summary': [False, "#f5f5f5"],
    'sentiment': [False, "#f5f5f5"],
    'exportCSV': [False, "#f5f5f5"]
}

analyze('data/emails.csv', features)