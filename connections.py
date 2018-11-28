## Project: Email Text Analyzer Tool: Network Diagram

"""
Author: Bennett Huffman
Last Modified: 11/19/18
Course: 15-112
Section: H
"""

## IMPORTS

import nltk
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.tokenize import word_tokenize
import string

import numpy as np
 
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from string import punctuation
from collections import Counter

# https://pythondata.com/text-analytics-visualization/

