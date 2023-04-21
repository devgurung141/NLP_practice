import numpy as np
import pandas as pd
import regex as re


import unicodedata
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def clean_text(text):
    lemmatizer = WordNetLemmatizer()
    stopwords = nltk.corpus.stopwords.words('english')
    unaccented_text = unicodedata.normalize('NFKD',text).encode('ascii', 'ignore').decode('utf-8', 'ignore').lower()
    words= re.sub(r'[^\w\s]', ' ', unaccented_text).split()
    clean_text = [lemmatizer.lemmatize(word) for word in words if word not in stopwords]
    return ' '.join(clean_text) 


def sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_dict = sia.polarity_scores(text)
    if sentiment_dict['compound'] >= 0.05:
        return 'Positve'
    elif sentiment_dict['compound']<= -0.05:
        return 'Negative'
    else:
        return 'Neutral'
    
    
def prepare_df(df):
    df.rename(columns = {'text_':'text'},inplace=True)
    df['is_fake'] = df.label.map({'CG': 1, 'OR':0})
    df.drop('label',axis=1, inplace=True)
    df['clean_text'] = df['text'].apply(clean_text)
    df['word_count'] = df.clean_text.str.len()
    df['sentiment'] = df['clean_text'].apply(sentiment)
    
    return df

    
    
    
    