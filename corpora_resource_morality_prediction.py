__author__ = 'mirko'
import re
import numpy
import glob
import spacy
import pickle
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import Counter
import emoji
import pickle


### PRENDE IN INPUT 1 TESTO E LO RIPULISCE DA STOP WORDS, URL, MENTIONS, HASHTAGS ###
from sklearn.svm import SVC





class MoralAttitude(object):


    languages = { 'en' : "en_core_web_lg",'es' : "es_core_news_lg" }
    language=None
    clf=None
    vectorizer=None
    nlp=None
    filename_vectorizer=None
    filename_clf=None
    def __init__(self,language):
        print("Loading morality profily for "+language+"...")
        self.nlp=spacy.load(self.languages[language])

        self.filename_vectorizer="cache/"+language+"_vectorizer_morality_cache.pickle"
        self.filename_clf="cache/"+language+"_clf_morality_cache.pickle"
        if os.path.isfile(self.filename_vectorizer) and os.path.isfile(self.filename_clf):

            self.vectorizer = pickle.load(open(self.filename_vectorizer,'rb'))
            self.clf = pickle.load(open(self.filename_clf,'rb'))
        else:

            df = pd.read_csv('external resources/en/Morality-in-Knowledge-Graphs-master/moral_corpus.csv')
            #id,text,target,language,set,HS,TR,AG
            to_be_vectorized=[]
            labels=[]
            i=0
            j=0
            for tweet in df.iloc[:].values:
                i+=1

                j+=1
                to_be_vectorized.append(self.cleaning_text(tweet[1]))
                labels.append(tweet[3])
                #print(i,j)



            self.vectorizer = TfidfVectorizer(ngram_range=(1,3),
                                     analyzer="word",
                                     # stop_words="english",
                                     lowercase=True,
                                     binary=True,
                                     max_features=500000)
            X = self.vectorizer.fit_transform(to_be_vectorized)
            self.clf = SVC(kernel="linear") #kernel='rbf'
            self.clf.fit(X, labels)
            pickle.dump(self.vectorizer,open(self.filename_vectorizer,'wb'))
            pickle.dump(self.clf,open(self.filename_clf,'wb'))


        return

    def clean_corpus(self,a_list):
        to_be_vectorized = [self.cleaning_text(t) for t in a_list]
        return to_be_vectorized

    def cleaning_text(self,a_text):
        #print(a_text)
        a_text = re.sub("#\w#", " ", a_text)
        a_text = emoji.demojize(a_text)
        a_text = re.sub("#",' ',a_text)
        a_text = re.sub("https://\S+",'',a_text)
        parsed = self.nlp(a_text)
        tokenized = [x.text for x in parsed if x.pos_ != 'PUNCT' and x.is_stop is False]
        tokenized = " ".join(tokenized)
        #print(tokenized)

        return tokenized

    def make_labels(self,a_list):
        labels = [t.label for t in a_list]

        return labels


    def predict_user_morality(self,texts):
        print("predicting user...")
        counter = Counter
        cleaned_texts = [self.cleaning_text(t) for t in texts]
        X_test = self.vectorizer.transform(cleaned_texts)
        test_predict = self.clf.predict(X_test)
        #print(test_predict)
        counted_prediction = counter(test_predict)

        return ["hs_yes"], [counted_prediction[1]]


if __name__ == '__main__':

    language="en"
    print(language)
    hateeval = MoralAttitude(language)
    texts=["@Ushimama1 @MielCaldero @USMC It is OBVIOUS from your tweet that you are too LOW IQ to understand or appreciate ""immigration done the LEGAL way"" so go whine to somone else. #Snowflakes#StopIllegalImmigration#BuildThatWall#MAGA",
           "It's easy to feel like you can't make a difference when the injustice of the world seems so overwhelming. But if we all do our part, we will see things change. Do your part for refugees this week  and help them rebuild their lives. âž https://t.co/QzFjSqWwJ8"]

    print(hateeval.predict_user_morality(texts))
