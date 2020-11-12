import pandas as pd
import numpy as np
import random
import pathlib


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
    if isinstance(txt, list):
        t = txt[0].split(' ')
    else:
        t = txt.split(' ')
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
    # change targets in string to 'product_target' if a target exists
    if isinstance(txt, list):
        t = ['product_target' if w in product_target(txt[1]) else w for w in t]
        if txt[1].lower() in ['android app', 'ipad or iphone app']:
            t = [w for w in t if w is not 'app']
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

def get_wordnet_pos(treebank_tag):
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

def df_clean(df = None, lem = True):
    '''
    A function that returns a cleaned up dataframe.
    It also drops all the no-product rows.
    Will take a dataframe as an argument, and can also pass the argument 'False' to turn off lemmatizer if needed.
    
    '''
    if df is None:
        this_path = pathlib.Path().absolute()
        d_path = this_path.parent / "data" 
        df = pd.read_csv(d_path/'judge-1377884607_tweet_product_company.csv', encoding = 'latin1')
    df.columns = ['text', 'product', 'emotion']
    df = df[df['emotion'] != 'I can\'t tell']
    df.dropna(inplace = True)
    df['text_product'] = df.apply(lambda x: list([x['text'], x['product']]), axis = 1)
    df['emotion'] = df['emotion'].map(emotion_label)
    df['txt_cleaned'] = df['text_product'].apply(txt_clean, args = (lem,))
    df.drop(columns = ['text', 'product', 'text_product'], inplace = True)
    return df


def external_data(lem = True):
    '''
    A function that returns a big array with all three external datasets cleaned up so that it is similar to our base data.
    Has an optional parameter of lem, by default it is true but setting it to false will leave out the lemmatizing.
    '''
    this_path = pathlib.Path().absolute()
    d_path = this_path.parent / "data" 

    df_1 = pd.read_csv(d_path/'Apple-Twitter-Sentiment-DFE.csv', encoding = 'latin1')
    df_1 = df_1[['sentiment', 'text']]
    df_1.columns = ['emotion', 'text']
    dic_1 = {'5': 2, '3' : 1, '1': 0}
    df_1.replace({'emotion': dic_1}, inplace = True)
    df_1['txt_cleaned'] = df_1['text'].apply(txt_clean, args = (lem,))
    df_1.drop('text', axis = 1, inplace = True)

    df_2 = pd.read_csv(d_path/'Deflategate-DFE.csv', encoding = 'latin1')
    df_2 = df_2[['deflate_sentiment', 'text']]
    df_2.columns = ['emotion', 'text']
    dic_2 = {'positive': 2, 'slightly positive': 2, ('neutral') : 1, 'negative': 0, 'slightly negative': 0}
    df_2.replace({'emotion': dic_2}, inplace = True)
    df_2['txt_cleaned'] = df_2['text'].apply(txt_clean, args = (lem,))
    df_2.drop('text', axis = 1, inplace = True)

    df_3 = pd.read_csv(d_path/'Coachella-2015-2-DFE.csv', encoding = 'latin1')
    df_3 = df_3[['coachella_sentiment', 'text']]
    df_3.columns = ['emotion', 'text']
    df_3 = df_3[df_3['emotion'] != 'cant tell']
    dic_3 = {'positive': 2, 'neutral' : 1, 'negative': 0}
    df_3.replace({'emotion': dic_3}, inplace = True)
    df_3['txt_cleaned'] = df_3['text'].apply(txt_clean, args = (lem,))
    df_3.drop('text', axis = 1, inplace = True)

    return pd.concat([df_1, df_2, df_3])







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