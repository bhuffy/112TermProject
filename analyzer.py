## Project: Email Text Analyzer Tool: Analyzer

"""
Author: Bennett Huffman
Last Modified: 11/28/18
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
# regular expressions
import re

# sentiment analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# summarization
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

# k-means
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# matplotlib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, rrulewrapper, RRuleLocator
import datetime

# network diagram
import networkx as nx

# testing
import time
    
## Colors

BRIGHTBLUE = "#3E92CC"
DARKBLUE = "#274060"
HOVER_DARKBLUE = "#152C49"
MAINBLUE = "#3E92CC"
MEDBLUE = "#004385"
OFFWHITE = "#D6E8F4"
WHITE = "#FFFFFF"
    
## Email Storage & Access

# store all emails from csv file
class Email(object):
    def __init__(self, data):
        self.date = str(data['Date Time'])
        self.fromName = str(data['From Name'])
        self.fromEmail = str(data['From Email'])
        self.toEmail = str(data['To Email'])
        self.subject = str(data['Subject'])
        self.messageId = str(data['Message Id'])
        self.body = str(data['Body'])
        self.sentiment = None
        self.summary = ""
        self.labels = None
        self.phones = None
        self.emailAddresses = None
        self.streetAddresses = None
        
    def setSummary(self, summary):
        self.summary = summary
        
    def setSentiment(self, sentiment):
        self.sentiment = sentiment
        
    def setLabels(self, labels):
        self.labels = labels
    
    def addPhones(self, phones):
        self.phones = phones
    
    def addEmailAddresses(self, emailAddresses):
        self.emailAddresses = emailAddresses
        
    def addStreetAddresses(self, streetAddresses):
        self.streetAddresses = streetAddresses

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
def getAllSentences(emails):
    allSentences = []
    for email in emails:
        allSentences.extend(sent_tokenize(processEmailText(email.body)))
        allSentences.extend(sent_tokenize(processEmailText(email.subject)))
    return allSentences
    
#get all words in bodies and subject
def getAllWords(sentences):
    allWords = []
    for sentence in sentences:
        allWords.extend(word_tokenize(sentence))
        allWords.extend(word_tokenize(sentence))
    return allWords
    
#filter out stop words and non-sensical strings
def filterWords(words):
    stopWords = set(stopwords.words("english")) # stop words
    d = enchant.Dict("en_US") # english dictionary
    filteredWords = []
    for word in words:
        if len(word) > 2 and word.isalpha() and word not in stopWords and d.check(word):
            filteredWords.append(word.lower())
    return filteredWords
    
# need to create a list of common email addresses, copyright, subscribe/unsubscribe, pronouns, etc. to filter out
def filterSentences(sentences):
    filteredSentences = []
    for sentence in sentences:
        allWords = filterWords(nltk.word_tokenize(sentence))
        if len(allWords) > 2:
            filteredSentences.append(sentence)
    return filteredSentences
    
# CITATION: http://www.noah.org/wiki/RegEx_Python#email_regex_pattern
def processEmailText(text):
    # remove phone, email, links, and then anything non alpha-numeric;
    text = re.sub(r"((1?[\s-]?\(?\d{3}\)?[\s\-\.]?\d{3}[\s\-\.]?\w{4}))|([a-zA-Z0-9+_\-\.]+@[0-9a-zA-Z][.-0-9a-zA-Z]*.[a-zA-Z]{3})|(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)|([0-9]* .*, .* [a-zA-Z]{2} [0-9]{5}-?[0-9]{4}?)|([^a-zA-Z0-9\s_\-\.\,\:\;\'])", "", text).replace("\n", " ")
    text = ' '.join(text.split())
    text = re.split(r"(This email|This e-mail|You’re receiving this|If you no longer wish|To opt-out|Best regards,|Best wishes,|Fond regards,|Kind regards,|Regards,|Sincerely,|Sincerely yours,|With appreciation,|Yours sincerely,|Yours truly,|Yours truly,|Thank you,|Unsubscribe|unsubscribe|To unsubscribe|Contact Preferences|Follow this link|Our system has detected|All rights reserved|No plain text content|---|To view this message in your browser|tel:|Fax:|The fine print|Copyright)", text)[0]
    return text
    
## Network Diagram

# creates a dictionary of the synonyms for each word
def createSynDictionary(allWords, words):
    synDict = dict()
    for word in words:
        synonyms = findSynonyms(word)
        synDict[word] = synonymsInDocument(allWords, synonyms, word)
    return synDict

# finds the synonyms of a given word via wordnet synsets; NOTE: pronouns + some words have no synsets (i.e, because, it, everything, etc.)
def findSynonyms(word):
    synonyms = set()
    for synset in wordnet.synsets(word):
        for lem in synset.lemmas():
            synonyms.add(lem.name())
        for hypernym in synset.hypernyms():
            for lem in hypernym.lemmas():
                synonyms.add(lem.name())
    return synonyms
    
# returns all synonyms that exist in as document
def synonymsInDocument(allWords, synonyms, word):
    documentSyns = set()
    for syn in synonyms:
        if syn != word and syn in allWords:
            documentSyns.add(lem.name())
    return documentSyns

# creates the 2 column dataframe mapping the connections
def createNetworkDF(synDict):
    data = dict({'from': [], 'to': []})
    # create dictionary from email objects
    for word1 in synDict:
        for word2 in synDict[word1]:
            data['from'].append(word1)
            data['to'].append(word2)
    return pd.DataFrame.from_dict(data)
    
def createNetDiagram(df):
    pass

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

def findPhoneNumbers(text):
    phoneNumbers = re.findall("(1?[\s-]?\(?\d{3}\)?[\s\-\.]?\d{3}[\s\-\.]?\w{4})", text)
    newNumbers = []
    for pnumber in phoneNumbers:
        newNumbers.append(re.sub(r"[\\n\s]", "", pnumber))
    return newNumbers
    
def findEmailAddresses(text):
    emailAddresses = re.findall('[a-zA-Z0-9+_\-\.]+@[0-9a-zA-Z][.-0-9a-zA-Z]*.[a-zA-Z]{3}', text)
    return emailAddresses
    
def findStreetAddresses(text):
    streetAddresses = re.findall('([0-9]* .*, .* [a-zA-Z]{2} [0-9]{5}-?[0-9]{4}?)', text)
    return streetAddresses
    
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
            
        # add found phone numbers and emails into email field
        email.addPhones(findPhoneNumbers(email.body))
        email.addEmailAddresses(findEmailAddresses(email.body))
        email.addStreetAddresses(findStreetAddresses(email.body))
        text = processEmailText(email.body)
        
        # create frequency table of all words
        words = filterWords(word_tokenize(text))
        freqTable = createDocFreqDist(words)
        
        # create sentences
        sentences = filterSentences(sent_tokenize(text))
        sentenceValues = scoreSentences(sentences, freqTable)
        averageValue = findAvgSentenceValue(sentenceValues)
        
        # create summary based on frequency
        maxValue = 0
        summary = ""
        for sentence in sentences:
            if sentence in sentenceValues:
                if sentenceValues[sentence] > maxValue:
                    summary = sentence
                elif sentenceValues[sentence] == maxValue:
                    summary += sentence
        print(summary)
        email.setSummary(summary)
        
    
## Frequency Distribution

def plotFrequencyDist(wordDist):
    wordDist.plot(50, cumulative=False, title="Frequency Distribution", color=BRIGHTBLUE)

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
        
def prepSentimentGraph(emails):
    print("Preparing sentiment graph features...")
    dates = []
    sentiments = []
    for email in emails:
        if email.sentiment != None:
            month, day, year = email.date.split(",")[0].split("/")
            dates.append(datetime.date(int(year), int(month), int(day)))
            sentiments.append(email.sentiment['compound'])
    return (dates, sentiments)
        
def graphSentiment(data):
    formatter = DateFormatter('%m/%d/%y')
    fig, ax = plt.subplots()
    # 0: dates, 1: sentiments; email label with point markers 
    plt.plot_date(data[0], data[1], label='emails', color=BRIGHTBLUE, marker=".")
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_tick_params(rotation=30, labelsize=8)
    
    plt.xlabel('Date')
    plt.ylabel('Sentiment')
    plt.title('Sentiment Graph')
    plt.legend()
    # plt.subplots_adjust(left=0.09, bottom=0.20, right=0.94, top=0.90, wspace=0.2, hspace=0)
    
    plt.show()

## CSV Export

def createDataFrame(emails):
    data = dict({'Date Time': [], 'From Name': [], 'From Email': [],
                    'To Name': [], 'To Email': [], 'Subject': [],
                    'Message Id': [], 'Body': [], 'Labels': [],
                    'Summary': [], 'Compound Sentiment': [], 'Positive Sentiment': [],
                    'Neutral Sentiment': [], 'Negative Sentiment': [], 'Phone Numbers': [],
                    'Email Addresses': [], 'Street Addresses': []
    })
    # create dictionary from email objects         self.phones = None self.emailAddresses = None
    for email in emails:
        data['Date Time'].append(email.date)
        data['From Name'].append(email.fromName)
        data['From Email'].append(email.fromEmail)
        data['To Name'].append("")
        data['To Email'].append(email.toEmail)
        data['Subject'].append(email.subject)
        data['Message Id'].append(email.messageId)
        data['Body'].append(email.body)
        data['Summary'].append(email.summary)
        if email.labels != None:
            data['Labels'].append(email.labels)
        else:
            data['Labels'].append(email.labels)
        if email.sentiment != None:
            data['Compound Sentiment'].append(email.sentiment['compound'])
            data['Positive Sentiment'].append(email.sentiment['pos'])
            data['Neutral Sentiment'].append(email.sentiment['neu'])
            data['Negative Sentiment'].append(email.sentiment['neg'])
        else:
            data['Compound Sentiment'].append("")
            data['Positive Sentiment'].append("")
            data['Neutral Sentiment'].append("")
            data['Negative Sentiment'].append("")
        if email.phones != None:
            data['Phone Numbers'].append(email.phones)
        else:
            data['Phone Numbers'].append("")
        if email.emailAddresses != None:
            data['Email Addresses'].append(email.emailAddresses)
        else:
            data['Email Addresses'].append("")
        if email.streetAddresses != None:
            data['Street Addresses'].append(email.streetAddresses)
        else:
            data['Street Addresses'].append("")
    
    return pd.DataFrame.from_dict(data)

def exportToCSV(emails):
    df = createDataFrame(emails)
    df.to_csv("data/out.csv", encoding='utf-8', index=False)

## Labeling

# tags all words to their respective parts of speech
def tagWords(words):
    return nltk.pos_tag(words)

# creates a frequency distribution based on the given part of speech
def createTagDist(tagPrefix, taggedWords):
    wordsForDictionary = []
    for (word, tag) in taggedWords:
        if tag.startswith(tagPrefix):
            wordsForDictionary.append((tag, word))
    fd = nltk.FreqDist(wordsForDictionary)
    return fd

def generateLabels(wordDict):
    labelDict = dict()
    
    for posWordPair in wordDict:
        # get word from pos and word tuple
        word = posWordPair[1]
        synonyms = findSynonyms(word)
        #check if word is in any of the existing labels
        wordExists = False
        for label in labelDict:
            #CITATION: disjoint => comparison efficiency! https://stackoverflow.com/questions/3170055/test-if-lists-share-any-items-in-python
            if word in labelDict[label][0] or not synonyms.isdisjoint(labelDict[label][0]):
                labelDict[label][0].add(word)
                labelDict[label][1] += wordDict[posWordPair]
                wordExists = True
                break
        
        #if word isn't in any labels, create new one!
        if not wordExists:
            labelDict[word] = [set([word]), wordDict[posWordPair]]
    return labelDict
    
def labelEmails(emails, labels):
    for email in emails:
        indivLabels = set()
        if email.body != "" or email.subject != "":
            for label in labels:
                if any(word in email.body for word in labels[label][0]) or \
                    any(word in email.subject for word in labels[label][0]):
                        indivLabels.add(label)
        print(indivLabels)
        email.setLabels(indivLabels)

def prepLabelGraph(emails, topLabels):
    print("Preparing label graph features...")
    labels = dict((label, 0) for label in topLabels)
    for email in emails:
        if email.labels not in [None, set()]:
            for label in email.labels:
                labels[label] += 1
    return (list(labels.keys()), list(labels.values()))
    
def graphLabels(data):
    fig, ax = plt.subplots()
    # 0: labels (words), 1: label frequency
    plt.bar(data[0], data[1], label="Frequency", color=BRIGHTBLUE)
    ax.xaxis.set_tick_params(rotation=70, labelsize=8)
    
    plt.xlabel('Labels')
    plt.ylabel('Frequency')
    plt.title('Label Frequency')
    plt.legend()
    # plt.subplots_adjust(left=0.09, bottom=0.20, right=0.94, top=0.90, wspace=0.2, hspace=0)
    
    plt.show()

## Analysis Controller

def analyze(filename, features):
    t1 = time.time()
    results = {'labels': [], 'freqdist': [], 'netdiagram': None,
                'summary': None, 'sentiment': None, 'exportCSV': None
    }
    
    # get emails from csv and filter them if file exists
    if filename not in [None, ""]:
        print("Processing emails...")
        emails = createEmailList(filename)
        sentences = getAllSentences(emails)
        words = filterWords(getAllWords(sentences))
        uniqueWords = set(words)
        wordDist = nltk.FreqDist(words)
        
        # labeling
        if features['labels'][0]:
            print("Generate labels!")
            t1 = time.time()
            
            # Takes approx. 1 minute to tag all words; Noun (NN), Adverb(RB), Verb (VB), JJ (Adjective)
            taggedWords = tagWords(words)
            labelDist = createTagDist("NN", taggedWords)
            
            # labelDist should have frequency of nouns, adjectives, adverbs, verbs
            labels = generateLabels(dict(labelDist))
            # CITATION: https://stackoverflow.com/questions/1747817/create-a-dictionary-with-list-comprehension-in-python
            topLabelsDict = {key:labels[key] for key in list(labels.keys())[:20]}
            print(topLabelsDict)
            
            #label emails and get keys
            labelEmails(emails, topLabelsDict)
            topLabels = list(labels.keys())[:15] # top 15 labels
            
            # prepare data for graph
            data = prepLabelGraph(emails, list(topLabelsDict.keys()))
            graphLabels(data)
            results['labels'] = topLabels
            t2 = time.time()
            print("Labels", (t2-t1)/60)
        
        # frequency distribution
        if features['freqdist'][0]:
            print("Generate frequency distribution!")
            t1 = time.time()
            results['freqdist'] = wordDist.most_common(15)
            t2 = time.time()
            print("Freq Dist", (t2-t1)/60)
        
        # network diagram
        if features['netdiagram'][0]:
            # synonym mapping: 12.5 minutes!!
            synConnections = createSynDictionary(words, uniqueWords)
            print(synConnections)
            # TODO: create diagram from pandas df
            # netdf = createNetworkDF(synConnections)
            # createNetDiagram(netdf)
            results['netdiagram'] = wordDist
            
        # summarization
        if features['summary'][0]:
            print("Summarize documents!")
            t1 = time.time()
            summarizeDocuments(emails)
            t2 = time.time()
            print("Summarization", (t2-t1)/60)
        
        # sentiment analysis
        if features['sentiment'][0]:
            print("Analyze sentiment!")
            t1 = time.time()
            analyzeSentiments(emails)
            data = prepSentimentGraph(emails)
            results['sentiment'] = data
            t2 = time.time()
            print("Sentiment Analysis", (t2-t1)/60)
            
        # csv export
        if features['exportCSV'][0]:
            print("Exporting to CSV...")
            t1 = time.time()
            exportToCSV(emails)
            results['exportCSV'] = 'data/out.csv'
            t2 = time.time()
            print("CSV Export", (t2-t1)/60)
    
    t2 = time.time()
    print((t2-t1)/60, results)
    return results
    
## RUN ANALYSIS

# features = {
#     'labels': [False, "#f5f5f5"],
#     'freqdist': [False, "#f5f5f5"],
#     'netdiagram': [True, "#f5f5f5"],
#     'summary': [False, "#f5f5f5"],
#     'sentiment': [False, "#f5f5f5"],
#     'exportCSV': [False, "#f5f5f5"]
# }
# 
# analyze('data/emails.csv', features)

"""
Match image files:
(\w+)\.(jpg|png|gif|pdf|docx)
"""