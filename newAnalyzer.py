## Project: Email Text Analyzer Tool: Analyzer

"""
Author: Bennett Huffman
Last Modified: 11/19/18
Course: 15-112
Section: H
"""

## IMPORTS

# csv handling
import csv
import pandas as pd
import enchant

# word processing
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import wordnet, stopwords
import string

# sentiment analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# summarization
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

# k-means
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# network diagram
import networkx as nx
    
## Email Storage & Access

# store all emails from csv file
class Email(object):
    def __init__(self, data):
        self.dateTime = str(data['Date Time'])
        self.fromName = str(data['From Name'])
        self.fromEmail = str(data['From Email'])
        self.toEmail = str(data['To Email'])
        self.subject = str(data['Subject'])
        self.messageId = str(data['Message Id'])
        self.body = str(data['Body'])
        self.sentiment = None
        self.summary = ""
        self.label = ""
        
    def setSummary(self, summary):
        self.summary = summary
        
    def setSentiment(self, sentiment):
        self.sentiment = sentiment

# CITATION: https://realpython.com/python-csv/
def createEmailList(path):
    emails = []
    # creates dictonary from values in each line of CSV
    with open(path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            emails.append(Email(row))
    return emails

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
    
## Email Pre-Processing
    
#get all words in bodies and subject
def getAllWords(emails):
    allWords = []
    for email in emails:
        allWords.extend(word_tokenize(email.body))
        allWords.extend(word_tokenize(email.subject))
    return allWords
    
#filter out stop words and non-sensical strings
def filterWords(words):
    stopWords = set(stopwords.words("english")) # stop words
    d = enchant.Dict("en_US") # english dictionary
    filteredWords = []
    for word in words:
        if word.isalpha() and word not in stopWords and d.check(word):
            filteredWords.append(word.lower())
    return filteredWords
    
# need to create a list of common email addresses, copyright, subscribe/unsubscribe, pronouns, etc. to filter out
def filterSentences(sentences):
    filteredSentences = []
    for sentence in sentences:
        allWords = filterWords(nltk.word_tokenize(sentence))
        if len(allWords) > 2 and "http" not in sentence:
            filteredSentences.append(sentence)
    return filteredSentences
    
## Synonym Mapping (Network Diagram)

# creates a dictionary of the synonyms for each word
def createSynDictionary(allWords, words):
    synDict = dict()
    for word in words:
        synonyms = findSynonyms(word)
        synDict[word] = synonymsInDocument(allWords, synonyms)
    return synDict

# finds the synonyms of a given word via wordnet synsets; NOTE: pronouns + some words have no synsets (i.e, because, it, everything, etc.)
def findSynonyms(word):
    synonyms = set()
    for synset in wordnet.synsets(word):
        for lem in synset.lemmas():
            synonyms.add(lem.name())
        # for hypernym in synset.hypernyms():
        #     for lem in hypernym.lemmas():
        #         synonyms.add(hypernym.name())
    return synonyms
    
# returns all synonyms that exist in as document
def synonymsInDocument(allWords, synonyms):
    documentSyns = set()
    for syn in synonyms:
        if syn != word and syn in allWords:
            documentSyns.add(lem.name())
    return documentSyns

# Adds the frequency of each word and its synonyms to the word
def addWordFrequency(document, synDict):
    for word in synDict:
        count = 0
        for synonym in synDict[word]:
            count += document.count(synonym)
        synDict[word] = (synDict[word], count)

## Summarization

# CITATION: https://dev.to/davidisrawi/build-a-quick-summarizer-with-python-and-nltk
# CITATION: https://gist.github.com/Sebastian-Nielsen/3bc45cbba6cb25837f5a6f11dbeeb044
# Creates summary based on sentences that have the most, most frequent words above a theshold (the average)

def createDocFreqDist(words):
    freqTable = dict()
    for word in words:
        # make sure words are the same when gauging frequency
        word = word.lower()
        word = stemmer.stem(word)
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1
    return freqTable
    
# score sentences
def scoreSentences(sentences, freqTable):
    sentenceValues = dict()
    for sentence in sentences:
        for word in freqTable:
            if word in sentence.lower():
                if sentence in sentenceValues:
                    sentenceValues[sentence] += freqTable[word]
                else:
                    sentenceValues[sentence] = freqTable[word]
    return sentenceValues
    
# now determine avg value of a sentence from document to compare sentences against and pick out the best ones based on multiple!
def findAvgSentenceValue(sentenceValues):
    sum = 0
    average = None
    for sentence in sentenceValues:
        sum += sentenceValues[sentence]
    
    # Average value of a sentence from original text
    if len(sentenceValues) > 0:
        average = int(sum / len(sentenceValues))
    return average
    
# gets most significant sentences from a document
def summarizeDocuments(emails):
    for email in emails:
        if email.body == "":
            continue
        
        # create frequency table of all words
        words = filterWords(word_tokenize(email.body))
        freqTable = createDocFreqDist(words)
        
        # create sentences
        sentences = filterSentences(sent_tokenize(email.body))
        sentenceValues = scoreSentences(sentences, freqTable)
        averageValue = findAvgSentenceValue(sentenceValues)
        
        # create summary based on frequency
        summary = ""
        for sentence in sentences:
            if sentence in sentenceValues and sentenceValues[sentence] > (1.7 * averageValue):
                summary +=  " " + sentence
        print(summary)
        email.setSummary(summary)
    
## Frequency Distribution

def plotFrequencyDist(wordDist):
    wordDist.plot(30,cumulative=False)

## Sentiment Analysis

# CITATION: https://www.pingshiuanchua.com/blog/post/simple-sentiment-analysis-python
# CITATION: https://monkeylearn.com/sentiment-analysis/ (READ THIS!)
# CITATION: http://www.laurentluce.com/posts/twitter-sentiment-analysis-using-python-and-nltk/

# Create a matplotlib chart with the results graphing the general sentiments!

# Implement Naive Bayesian Algorithm and classifier later!
def getSentiment(sentence):
    sentAnalyzer = SentimentIntensityAnalyzer()
    score = sentAnalyzer.polarity_scores(sentence)
    return score
    
def analyzeSentiments(emails):
    for email in emails:
        if email.body == "":
            continue
        score = getSentiment(email.body)
        email.setSentiment(score)

## CSV Export

def createDataFrame(emails):
    data = dict({'Date Time': [], 'From Name': [], 'From Email': [],
                    'To Name': [], 'To Email': [], 'Subject': [],
                    'Message Id': [], 'Body': [], 'Label': [],
                    'Summary': [], 'Sentiment': []
    })
    # create dictionary from email objects
    for email in emails:
        print("hi")
        data['Date Time'].append(email.dateTime)
        data['From Name'].append(email.fromName)
        data['From Email'].append(email.fromEmail)
        data['To Name'].append("")
        data['To Email'].append(email.toEmail)
        data['Subject'].append(email.subject)
        data['Message Id'].append(email.messageId)
        data['Body'].append(email.body)
        data['Label'].append(email.label)
        data['Summary'].append(email.summary)
        data['Compound Sentiment'].append(email.sentiment['compound'])
        data['Positive Sentiment'].append(email.sentiment['pos'])
        data['Neutral Sentiment'].append(email.sentiment['neu'])
        data['Negative Sentiment'].append(email.sentiment['neg'])
    
    return pd.DataFrame.from_dict(data)

def exportToCSV(emails):
    df = createDataFrame(emails)
    df.to_csv("data/out.csv", encoding='utf-8', index=False)

## Labeling

def generateLabels(wordDist):
    wordDict = dict(wordDist)
    labelDict = dict()
    
    for word in wordDict:
        
        if word in labelDict:
            labelDict[word] = (set(word), wordDict[word])

## Analysis Controller

def analyze(filename, featuresDict):
    # get emails from csv and filter them
    emails = createEmailList(filename)
    
    # frequency distribution
    words = filterWords(getAllWords(emails))
    uniqueWords = set(words)
    
    # maps most to least common words in key/value pair object
    wordDist = nltk.FreqDist(words)
    wordDist.most_common(15)
    plotFrequencyDist(wordDist)
    
    # labeling
    labels = generateLabels(wordDist)
    print(labels)
    
    # summarization
    summarizeDocuments(emails)
    
    # sentiment analysis
    analyzeSentiments(emails)
    
    # network diagram synonym mapping: 12.5 minutes!!
    # synConnections = createSynDictionary(words, uniqueWords)
    
    # csv export
    exportToCSV(emails)
    

## RUN ANALYSIS

features = {
    'labels': [False, "#f5f5f5"],
    'freqdist': [False, "#f5f5f5"],
    'netdiagram': [False, "#f5f5f5"],
    'summary': [False, "#f5f5f5"],
    'sentiment': [False, "#f5f5f5"],
    'exportCSV': [False, "#f5f5f5"]
}

analyze('data/emails.csv', features)