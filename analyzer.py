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
        self.label = ""
        self.phones = None
        self.emailAddresses = None
        
    def setSummary(self, summary):
        self.summary = summary
        
    def setSentiment(self, sentiment):
        self.sentiment = sentiment
        
    def setLabel(self, label):
        self.label = label
    
    def addPhones(self, phones):
        self.phones = phones
    
    def addEmailAddresses(self, emailAddresses):
        self.emailAddresses = emailAddresses

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
        if word.isalpha() and len(word) > 1 and word not in stopWords and d.check(word):
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
    
def processEmailBody(text):
    # remove phone, email, links, and then anything non alpha-numeric
    text = re.sub(r"((1?[\s-]?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\w{4}))|([a-zA-Z0-9+_\-\.]+@[0-9a-zA-Z][.-0-9a-zA-Z]*.[a-zA-Z]{3})|(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)|([^a-zA-Z0-9\s_\-\.\,\:\;])", "", text)
    print(text)
    # sentences = nltk.sent_tokenize(text)
    # sentences = [nltk.word_tokenize(sent) for sent in sentences]
    # sentences = [nltk.pos_tag(sent) for sent in sentences]
    return text
    
## Network Diagram

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
        for hypernym in synset.hypernyms():
            for lem in hypernym.lemmas():
                synonyms.add(lem.name())
    return synonyms
    
# returns all synonyms that exist in as document
def synonymsInDocument(allWords, synonyms):
    documentSyns = set()
    for syn in synonyms:
        if syn != word and syn in allWords:
            documentSyns.add(lem.name())
    return documentSyns

# creates the 2 column dataframe mapping the connections
def createNetworkDF(synDict):
    data = dict({'to': [], 'from': []})
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
    phoneNumbers = re.findall("(1?[\s-]?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\w{4})", text)
    newNumbers = []
    for pnumber in phoneNumbers:
        newNumbers.append(re.sub(r"[\\n\s]", "", pnumber))
    print(newNumbers)
    return newNumbers
    
def findEmailAddresses(text):
    emailAddresses = re.findall('[a-zA-Z0-9+_\-\.]+@[0-9a-zA-Z][.-0-9a-zA-Z]*.[a-zA-Z]{3}', text)
    print(emailAddresses)
    return emailAddresses
    
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
        text = processEmailBody(email.body)
        
        # create frequency table of all words
        words = filterWords(word_tokenize(email.body))
        freqTable = createDocFreqDist(words)
        
        # create sentences
        sentences = filterSentences(sent_tokenize(text))
        sentenceValues = scoreSentences(sentences, freqTable)
        averageValue = findAvgSentenceValue(sentenceValues)
        
        # create summary based on frequency
        summary = ""
        for sentence in sentences:
            if sentence in sentenceValues and sentenceValues[sentence] > (1.8 * averageValue):
                summary +=  " " + sentence
        email.setSummary(summary)
    
## Frequency Distribution

def plotFrequencyDist(wordDist):
    wordDist.plot(100, cumulative=False)

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
    # 0: dates, 1: sentiments; email label with point markers and 
    plt.plot_date(data[0], data[1], label='emails', color=BRIGHTBLUE, marker=".")
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_tick_params(rotation=30, labelsize=8)
    
    plt.xlabel('Date')
    plt.ylabel('Sentiment')
    plt.title('Sentiment Graph')
    plt.legend()
    
    plt.show()

## CSV Export

def createDataFrame(emails):
    data = dict({'Date Time': [], 'From Name': [], 'From Email': [],
                    'To Name': [], 'To Email': [], 'Subject': [],
                    'Message Id': [], 'Body': [], 'Label': [],
                    'Summary': [], 'Compound Sentiment': [], 'Positive Sentiment': [],
                    'Neutral Sentiment': [], 'Negative Sentiment': []
    })
    # create dictionary from email objects
    for email in emails:
        data['Date Time'].append(email.date)
        data['From Name'].append(email.fromName)
        data['From Email'].append(email.fromEmail)
        data['To Name'].append("")
        data['To Email'].append(email.toEmail)
        data['Subject'].append(email.subject)
        data['Message Id'].append(email.messageId)
        data['Body'].append(email.body)
        data['Label'].append(email.label)
        data['Summary'].append(email.summary)
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
    
    return pd.DataFrame.from_dict(data)

def exportToCSV(emails):
    df = createDataFrame(emails)
    df.to_csv("data/out.csv", encoding='utf-8', index=False)

## Labeling

# CITATION: using disjoint for comparison due to efficiency! https://stackoverflow.com/questions/3170055/test-if-lists-share-any-items-in-python

def tagWords(text):
    return nltk.pos_tag(text)

def createTagDict(targetTag, taggedText):
    cfd = nltk.ConditionalFreqDist((tag, word) for (word, tag) in taggedText if tag.startswith(targetTag))
    return dict((tag, cfd[tag].most_common(5)) for tag in cfd.conditions())

def generateLabels(wordDist):
    wordDict = dict(wordDist)
    labelDict = dict()
    
    for word in wordDict:
        synonyms = findSynonyms(word)
        #check if word is in any of the existing labels
        wordExists = False
        for label in labelDict:
            if word in labelDict[label][0] or not synonyms.isdisjoint(labelDict[label][0]):
                labelDict[label][0].add(word)
                labelDict[label][1] += wordDict[word]
                wordExists = True
                break
        
        #if word isn't in any labels, create new one!
        if not wordExists:
            labelDict[word] = [set([word]), wordDict[word]]
    
    return labelDict
    

## Clustering (TESTING IN PROGRESS)

# def findKMeans(documents):
#     # vectorize text
#     vectorizer = TfidfVectorizer(stop_words='english')
#     X = vectorizer.fit_transform(documents)
#     
#     # start with 2 clusters
#     true_k = 2
#     model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
#     model.fit(X)
#     
#     print("Top terms per cluster:")
#     order_centroids = model.cluster_centers_.argsort()[:, ::-1] # don't know what this does...?
#     terms = vectorizer.get_feature_names()
#     for i in range(true_k):
#         print("Cluster %d:" % i),
#         for ind in order_centroids[i, :10]:
#             print(' %s' % terms[ind]),
#         print()
#     
#     print("\n")
#     print("Prediction")
#     
#     Y = vectorizer.transform(["College is great, please apply and come to Lawrence!."])
#     prediction = model.predict(Y)
#     print(prediction)
#     
#     Y = vectorizer.transform(["Bennett, congratulations on something, please visit us. We can do something amazing."])
#     prediction = model.predict(Y)
#     print(prediction)

"""
1. take all bodies of emails
2. find nouns for clustering
3. find most frequent
4. find adjectives for classification of sentiment
"""

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
        words = filterWords(getAllWords(emails))
        uniqueWords = set(words)
        wordDist = nltk.FreqDist(words)
        
        # labeling
        if features['labels'][0]:
            print("Generate labels!")
            labelDist = tagWords(words)
            labels = generateLabels(wordDist)
            topLabels = list(labels.keys())[:10] # top 10 labels
            
            results['labels'] = topLabels
        
        # frequency distribution
        if features['freqdist'][0]:
            print("Generate frequency distribution!")
            results['freqdist'] = wordDist.most_common(10)
        
        # network diagram
        if features['netdiagram'][0]:
            # synonym mapping: 12.5 minutes!!
            # synConnections = createSynDictionary(words, uniqueWords)
            # TODO: create diagram from pandas df
            # netdf = createNetworkDF(synConnections)
            # createNetDiagram(netdf)
            results['netdiagram'] = wordDist
            
        # summarization
        if features['summary'][0]:
            print("Summarize documents!")
            summarizeDocuments(emails)
        
        # sentiment analysis
        if features['sentiment'][0]:
            print("Analyze sentiment!")
            analyzeSentiments(emails)
            data = prepSentimentGraph(emails)
            
            results['sentiment'] = data
            
        # csv export
        if features['exportCSV'][0]:
            exportToCSV(emails)
            results['exportCSV'] = 'data/out.csv'
    
    t2 = time.time()
    print((t2-t1)/60, results)
    return results
    

## RUN ANALYSIS

features = {
    'labels': [False, "#f5f5f5"],
    'freqdist': [False, "#f5f5f5"],
    'netdiagram': [False, "#f5f5f5"],
    'summary': [True, "#f5f5f5"],
    'sentiment': [False, "#f5f5f5"],
    'exportCSV': [False, "#f5f5f5"]
}

analyze('data/emails.csv', features)

"""
REGEX: CITATION: http://www.noah.org/wiki/RegEx_Python#email_regex_pattern

Match phone numbers:
phoneNumbers = re.findall("1?[\s-]?\(?(\d{3})\)?[\s-]?\d{3}[\s-]?\w{4}", email)
phoneRegex = re.compile(r"1?[\s-]?\(?(\d{3})\)?[\s-]?\d{3}[\s-]?\w{4}", re.IGNORECASE)
re.sub(STUFF)

Match emails:
emailAddresses = re.findall('[a-zA-Z0-9+_\-\.]+@[0-9a-zA-Z][.-0-9a-zA-Z]*.[a-zA-Z]+', email)
emailRegex = re.compile(r'[a-zA-Z0-9+_\-\.]+@[0-9a-zA-Z][.-0-9a-zA-Z]*.[a-zA-Z]+', re.IGNORECASE)
re.sub(STUFF)

urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', page)
re.sub(STUFF)

Match image files:
(\w+)\.(jpg|png|gif|pdf|docx)
"""