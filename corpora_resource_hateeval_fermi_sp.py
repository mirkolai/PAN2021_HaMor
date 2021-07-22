__author__ = 'mirko'
import re
import fasttext
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
from sklearn.svm import SVC
from scipy.sparse import csr_matrix, hstack
import tensorflow_hub as hub

class Fermi(object):
    embed=None
    languages = { 'en' : "en_core_web_lg",'es' : "es_core_news_lg" }
    language=None
    clf=None
    nlp=None
    universal_encoder=None
    sentence_embeddings=None
    def __init__(self,language):

        print("Loading Fermi for "+language+"...")

        print("loading Spacy model..")
        self.nlp=spacy.load(self.languages[language])
        print("loading Universal encoder..")
        self.embed= hub.load("external resources/en/universal-sentence-encoder_4")
        print("reading files...")

        file_name="cache/"+language+"_hateeval_tweets_and_labels_cache.pickle"
        if os.path.isfile(file_name):
            load=pickle.load(open(file_name,'rb'))
            to_be_vectorized=load['to_be_vectorized']
            labels=load['labels']
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
                print(i,j)
            dump={ 'labels':labels,'to_be_vectorized':to_be_vectorized }
            pickle.dump(dump,open(file_name,'wb'))

        print("training the model...")
        #crea modello
        file_name="cache/"+language+"_Fermi_clf_cache.pickle"
        if os.path.isfile(file_name):
            self.clf = pickle.load(open(file_name,'rb'))
        else:
            X=self.embed(to_be_vectorized)
            self.clf = SVC(kernel="rbf")
            self.clf.fit(X, labels)
            pickle.dump(self.clf,open(file_name,'wb'))

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
        X_test=self.embed(cleaned_texts)

        test_predict = self.clf.predict(X_test)
        #print(test_predict)
        counted_prediction = counter(test_predict)

        return ["hs_yes"], [counted_prediction[1]]


if __name__ == '__main__':

    language="en"
    print(language)
    hateeval = Fermi(language)
    texts=["@Ushimama1 @MielCaldero @USMC It is OBVIOUS from your tweet that you are too LOW IQ to understand or appreciate ""immigration done the LEGAL way"" so go whine to somone else. #Snowflakes#StopIllegalImmigration#BuildThatWall#MAGA",
          "It's easy to feel like you can't make a difference when the injustice of the world seems so overwhelming. But if we all do our part, we will see things change. Do your part for refugees this week  and help them rebuild their lives. âž https://t.co/QzFjSqWwJ8"]

    print(hateeval.predict_user(texts))
