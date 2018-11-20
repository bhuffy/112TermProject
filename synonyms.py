## Synonym Frequency

## IMPORTS

import nltk
from nltk.corpus import wordnet, stopwords
import numpy as np
import matplotlib
import string

## SYNONYMS

# removes stop words and punctuation
def filterWords(words):
    stopWords = set(stopwords.words("english"))
    filteredWords = []
    for word in words:
        if word not in string.punctuation and word not in stopWords:
            filteredWords.append(word)
    return filteredWords

#creates a dictionary of the synonyms for each word
def createSynDictionary(words):
    synDict = dict()
    for word in words:
            synDict[word] = findSynonyms(word)
    return synDict

# finds the synonyms of a given word via wordnet synsets
def findSynonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
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
def analyzeText():
    text = """At eight o'clock on Thursday morning Arthur didn't feel very good. Everything looks great, like trains and cupcakes. I can't    believe it's not butter!"""
    words = nltk.word_tokenize(text)
    filteredWords = filterWords(words)
    synDict = createSynDictionary(filteredWords)
    addWordFrequency(text, synDict)
    print(synDict)

analyzeText()