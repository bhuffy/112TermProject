## Project: Email Text Analyzer Tool: Load Data File

"""
Author: Bennett Huffman
Last Modified: 11/19/18
Course: 15-112
Section: H
"""

## IMPORTS

import csv
import pandas

#Move these to analyze file when done
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet, stopwords
import string

## Classes

# Email Class
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

# Complaint data: data/Consumer_Complaints.csv

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

#filter words such that they make sense
def filterWords(words):
    stopWords = set(stopwords.words("english"))
    filteredWords = []
    for word in words:
        if word.isalpha() and word not in stopWords:
            filteredWords.append(word.lower())
    return filteredWords

# Get each word and for each of them, create a dictionary whether the top 1000 words are in the document
def findFeatures(email):
    words = set(email)
    features = dict()
    for w in word_features:
        features[w] = (w in words) # creates true/false if word in document mapping to list of features
    return features

def analyze():
    emails = createEmailList('data/emails.csv')
    allWords = getAllWords(emails)
    filteredWords = filterWords(allWords)
    
    filteredWords = nltk.FreqDist(filteredWords) # makes object mapping most to least common words in key/value pair
    wordFeatures = list(filteredWords.keys())[:1000] # gets the top 1000 words
    
    # how do I create a training dataset...?
    # featureSets = [(findFeatures(email), category) for (email, category) in emails]
    print(filteredWords.most_common(15)) # prints 15 most common words
    print(filteredWords)
    print(wordFeatures)
analyze()



# Using CSV with pandas, might want later for csv export?
# df = pandas.read_csv('data/Consumer_Complaints.csv', header=0)
# for row in df.iterrows():
#     print(row[1])