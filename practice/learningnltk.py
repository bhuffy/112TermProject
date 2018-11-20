import nltk
import numpy as np
import matplotlib
from nltk.book import *

text1.concordance("monstrous") # finds all sentences in text with word monstrous
text1.similar("monstrous")
text1.common_contexts([“monstrous”,"very"])
text1.generate()

