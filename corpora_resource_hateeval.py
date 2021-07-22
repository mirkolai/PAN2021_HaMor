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





class HatEval(object):


    languages = { 'en' : "en_core_web_lg",'es' : "es_core_news_lg" }
    language=None
    clf=None
    vectorizer=None
    nlp=None
    def __init__(self,language):
        self.nlp=spacy.load(self.languages[language])
        print("Loading HateEval for "+language+"...")


        if os.path.isfile("cache/"+language+"_vectorizer_cache.pickle") and os.path.isfile("cache/"+language+"_clf_cache.pickle"):

            self.vectorizer = pickle.load(open("cache/"+language+"_vectorizer_cache.pickle",'rb'))
            self.clf = pickle.load(open("cache/"+language+"_clf_cache.pickle",'rb'))
        else:

            df = pd.read_csv('external resources/'+language+'/hateeval/hateval2019_target.csv')
            #id,text,target,language,set,HS,TR,AG
            to_be_vectorized=[]
            labels=[]
            i=0
            j=0
            for tweet in df.iloc[:].values:
                i+=1
                if tweet[3]==language:
                    j+=1
                    to_be_vectorized.append(self.cleaning_text(tweet[1]))
                    labels.append(tweet[5])

            self.vectorizer = TfidfVectorizer(ngram_range=(1,3),
                                     analyzer="word",
                                     # stop_words="english",
                                     lowercase=True,
                                     binary=True,
                                     max_features=500000)
            X = self.vectorizer.fit_transform(to_be_vectorized)
            self.clf = SVC(kernel="linear") #kernel='rbf'
            self.clf.fit(X, labels)
            pickle.dump(self.vectorizer,open("cache/"+language+"_vectorizer_cache.pickle",'wb'))
            pickle.dump(self.clf,open("cache/"+language+"_clf_cache.pickle",'wb'))


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


    def predict_user(self,texts):
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
    hateeval = HatEval(language)
    texts=["RT #USER#: jurisprudencia Felices 119 años, #USER#. Historia que tú hiciste, historia por hacer. Hala Madrid!!❤️ #URL#",
           "#USER# illegalization Vamos sin prisa pero vamos con todo ! #URL#"]
    texts=["@Ushimama1 @MielCaldero @USMC It is OBVIOUS from your tweet that you are too LOW IQ to understand or appreciate ""immigration done the LEGAL way"" so go whine to somone else. #Snowflakes#StopIllegalImmigration#BuildThatWall#MAGA",
           "It's easy to feel like you can't make a difference when the injustice of the world seems so overwhelming. But if we all do our part, we will see things change. Do your part for refugees this week  and help them rebuild their lives. âž https://t.co/QzFjSqWwJ8"]

    print(hateeval.predict_user(texts))
