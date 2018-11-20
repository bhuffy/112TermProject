import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

text = """At eight o'clock on Thursday morning Arthur didn't feel very good. Everything is wonderful, like trains and cupcakes. I can't believe it's not butter!"""

#NOTE: Can make own tokenizers... maybe something to look into?
# print(nltk.sent_tokenize(text))
# print(nltk.word_tokenize(text))
# 
# train_text = state_union.raw("2005-GWBush.txt") # .raw takes the raw text from a file
# sample_text = state_union.raw("2006-GWBush.txt")
# custom_sent_tokenizer = PunktSentenceTokenizer(train_text) # trains tokenier
# 
# tokenized = custom_sent_tokenizer.tokenize(sample_text)
# 
# def processContent():
#     try:
#         for i in tokenized:
#             words = nltk.word_tokenize(i)
#             tagged = nltk.pos_tag(words)
#             
#             namedEnt = nltk.ne_chunk(tagged, binary=True)
#             namedEnt.draw()
#     except Exception as e:
#         print(str(e))
#     
# processContent()
    
"""
CC coordinating conjunction
CD cardinal digit
DT determiner
EX existential there (like: “there is” … think of it like “there exists”)
FW foreign word
IN preposition/subordinating conjunction
JJ adjective ‘big’
JJR adjective, comparative ‘bigger’
JJS adjective, superlative ‘biggest’
LS list marker 1)
MD modal could, will
NN noun, singular ‘desk’
NNS noun plural ‘desks’
NNP proper noun, singular ‘Harrison’
NNPS proper noun, plural ‘Americans’
PDT predeterminer ‘all the kids’
POS possessive ending parent’s
PRP personal pronoun I, he, she
PRP$ possessive pronoun my, his, hers
RB adverb very, silently,
RBR adverb, comparative better
RBS adverb, superlative best
RP particle give up
TO, to go ‘to’ the store.
UH interjection, errrrrrrrm
VB verb, base form take
VBD verb, past tense took
VBG verb, gerund/present participle taking
VBN verb, past participle taken
VBP verb, sing. present, non-3d take
VBZ verb, 3rd person sing. present takes
WDT wh-determiner which
WP wh-pronoun who, what
WP$ possessive wh-pronoun whose
WRB wh-abverb where, when
"""
## Lemmatizing

# from nltk.stem import WordNetLemmatizer
# 
# lemmatizer = WordNetLemmatizer()
# print(lemmatizer.lemmatize("cats"))

## Finding files

# print(nltk.__file__)




