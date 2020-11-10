import pandas as pd
import numpy as np
import random

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

import re
from nltk import pos_tag
from nltk.tokenize import regexp_tokenize, word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer 



def get_wordnet_pos(treebank_tag):
    '''
    Function takes in a string and assigns it a part of speech tag.
    Used for lemmatizing.
    '''
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
def product_target(string):
    '''
    Takes in the data from the product column through the string parameter.
    Returns a string that the clean function will replace with 'product_target' if it sees any instances of it.
    '''
    s = string.lower()
    if s == 'no target':
        return ''
    elif s == 'ipad':
        return 'ipad'
    elif s == 'apple':
        return 'apple'
    elif s == 'ipad or iphone app':
        return 'app'
    elif s == 'iphone':
        return 'iphone'
    elif s == 'other apple product or service':
        return ''
    elif s == 'google':
        return 'google'
    elif s == 'other google product or service':
        return ''
    elif s == 'android':
        return 'android'
    elif s == 'android app':
        return 'android'
    else:
        return 'Unknown target'

def txt_clean(txt, lem):
    '''
    Takes in a string and returns a cleaned up version of it.
    Will be ran by itself in the df_clean function.
    '''
    
    sw = stopwords.words('english')
    sw.extend(['link', 'rt', 'get'])
    punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~“!#'
    no_accents_re = re.compile('^[a-z]+$')
    accents = ['á', 'â', 'ã', 'à', 'å', 'ª', 'ç', 'è', '¼', '¾', 'î', 'ï', 'ì', 'ó', 'ö', 'ð', 'ü', 'ù', 'û', 'ý']
    twitter_re = re.compile('[@][a-zA-Z]*')
    num_re = re.compile('^\d+$')
    
    # splitting the text up into words
    t = txt[0].split(' ')
    # turning the words lowercase
    t = [w.lower() for w in t]
    # removing punctuation
    t = [w.translate(w.maketrans('','', punctuation)) for w in t]
    # removing @'s which are twitter jargon
    t = [w for w in t if not twitter_re.match(w)]
    # removing leftover numbers
    t = [w for w in t if not num_re.match(w)]
    # removing words with accents
    t = [w for w in t if no_accents_re.match(w)]
    # removing stop words and more twitter jargon
    t = [w for w in t if w not in sw]
    # change targets in string to 'product_target'
    t = ['product_target' if w == product_target(txt[1]) else w for w in t]
    # removing empty strings
    t = [w for w in t if w]
    # word lemmatizing
    if lem: 
        lemm = WordNetLemmatizer()
        t = pos_tag(t)
        t = [(w[0], get_wordnet_pos(w[1])) for w in t]
        t = [lemm.lemmatize(w[0], w[1]) for w in t]
    # joining all the strings together into one
    return ' '.join(t)

def emotion_label(string):
    '''
    Turns the emotion values into numerical labels.
    '''
    s = string
    if s == 'Positive emotion':
        return 2
    elif s == 'No emotion toward brand or product':
        return 1
    elif s == 'Negative emotion':
        return 0
    else:
        print('Unknown emotion')

def df_clean(df, lem = True):
    '''
    A function that returns a cleaned up dataframe.
    It also drops all the no-product rows.
    Will take a dataframe as an argument, and can also pass the argument 'False' to turn off lemmatizer if needed.
    
    '''
    df.columns = ['text', 'product', 'emotion']
    df = df[df['emotion'] != 'I can\'t tell']
    df.dropna(inplace = True)
    df['text_product'] = df.apply(lambda x: list([x['text'], x['product']]), axis = 1)
    df['emotion'] = df['emotion'].map(emotion_label)
    df['txt_cleaned'] = df['text_product'].apply(txt_clean, args = (lem,))
    df.drop(columns = ['text', 'product', 'text_product'], inplace = True)
    return df









class Vectorizer:
    '''
    Vectorizer class.
    When initializing it, requires a type parameter and a tuple to input ngram parameters.
    A type parameter of 'cv' will create a Count Vectorizer, while a parameter of 'tfidf' will create a Tfidf Vectorizer.
    The first number in the ngram is the minimum ngram, the second number is the maximum ngram.
    
    Methods include .fit(), .transform(), and .fit_transform().  They should work exactly the same as usual models with the same inputs and outputs.
    '''
    def __init__(self, vec_type, ngram = (1,1)):
        '''
        Function requires a vec_type argument of 'cv' or 'tfidf' and a tuple for ngrams.
        Vectorizer is initialized with the type denoted in the vec_type parameter.
        '''
        if type(ngram) is not tuple:
            print('Unknown tuple, format should be (minimum n-gram, maximum n-gram)')
            return False
        
        if vec_type == 'cv':
            self.vec = CountVectorizer(ngram_range = ngram)
        elif vec_type == 'tfidf':
            self.vec = TfidfVectorizer(ngram_range = ngram)
        else:
            print('Unknown vectorizer type')
            return False
        
    def fit(self, X, y = None):
        '''
        Requires an input data X.
        Fits the vectorizer to it.
        '''
        self.vec.fit(X)

    def transform(self, X, y):
        '''
        Requires an input data X and y.
        Transforms the input data X and returns it.
        '''
        X_vec = self.vec.transform(X)
        X_vec = pd.DataFrame.sparse.from_spmatrix(X_vec)
        X_vec.columns = sorted(self.vec.vocabulary_)
        X_vec.set_index(y.index, inplace = True)
        return X_vec
    
    def fit_transform(self, X, y):
        '''
        Requires an input data X and y.
        Fits the vectorizer to the input data X, transforms it, and then returns it.
        '''
        self.vec.fit(X)
        X_vec = self.vec.transform(X)
        X_vec = pd.DataFrame.sparse.from_spmatrix(X_vec)
        X_vec.columns = sorted(self.vec.vocabulary_)
        X_vec.set_index(y.index, inplace = True)
        return X_vec
    
    
    

