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


### PRENDE IN INPUT 1 TESTO E LO RIPULISCE DA STOP WORDS, URL, MENTIONS, HASHTAGS ###
from sklearn.svm import SVC
from scipy.sparse import csr_matrix, hstack


import io

def load_vectors(fname):
    print("loading fasttext... ")
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data


class Atalay(object):


    languages = { 'en' : "en_core_web_lg",'es' : "es_core_news_lg" }
    language=None
    clf=None
    nlp=None
    vectorizer_bow_binary=None
    vectorizer_bow_tfidf=None
    vectorizer_char_binary=None
    vectorizer_char_tfidf=None
    #sentence_embeddings=None
    fasttext_model=None
    def __init__(self,language):
        print("Loading Atalay for "+language+"...")
        print("loading fasttext model..")
        self.fasttext_model = fasttext.load_model("external resources/es/fasttext/cc.es.300.bin")
        print("loading Spacy model..")
        self.nlp=spacy.load(self.languages[language])
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
                #print(i,j)
            dump={ 'labels':labels,'to_be_vectorized':to_be_vectorized }
            pickle.dump(dump,open(file_name,'wb'))
        print("binary bow 1-2grams...")
        #binary bow 1-2grams
        file_name="cache/"+language+"_Atalay_vectorizer_bow_binary_cache.pickle"
        if os.path.isfile(file_name):
            self.vectorizer_bow_binary = pickle.load(open(file_name,'rb'))
        else:
            self.vectorizer_bow_binary = CountVectorizer(ngram_range=(1, 2),
                                     analyzer="word",
                                     # stop_words="english",
                                     lowercase=True,
                                     binary=True,
                                     max_features=500000)
            self.vectorizer_bow_binary.fit(to_be_vectorized)
            pickle.dump(self.vectorizer_bow_binary,open(file_name,'wb'))
        X = self.vectorizer_bow_binary.transform(to_be_vectorized)
        print(X.shape,"X.shape")
        print("td-idf bow 1-2grams...")
        #td-idf bow 1-2grams
        file_name="cache/"+language+"_Atalay_vectorizer_bow_tfidf_cache.pickle"
        if os.path.isfile(file_name):
            self.vectorizer_bow_tfidf = pickle.load(open(file_name,'rb'))
        else:
            self.vectorizer_bow_tfidf = TfidfVectorizer(ngram_range=(1, 2),
                                     analyzer="word",
                                     # stop_words="english",
                                     lowercase=True,
                                     binary=True,
                                     max_features=500000)
            self.vectorizer_bow_tfidf.fit(to_be_vectorized)
            pickle.dump(self.vectorizer_bow_tfidf,open(file_name,'wb'))

        X = csr_matrix(hstack((X,self.vectorizer_bow_tfidf.transform(to_be_vectorized))))
        print(X.shape,"X.shape")
        print("binary char 3-5grams...")
        #binary char 3-5grams
        file_name="cache/"+language+"_Atalay_vectorizer_char_binary_cache.pickle"
        if os.path.isfile(file_name):
            self.vectorizer_char_binary = pickle.load(open(file_name,'rb'))
        else:
            self.vectorizer_char_binary = CountVectorizer(ngram_range=(3, 5),
                                     analyzer="char",
                                     # stop_words="english",
                                     lowercase=True,
                                     binary=True,
                                     max_features=500000)
            self.vectorizer_char_binary.fit(to_be_vectorized)
            pickle.dump(self.vectorizer_char_binary,open(file_name,'wb'))
        X = csr_matrix(hstack((X,self.vectorizer_char_binary.transform(to_be_vectorized))))
        print(X.shape,"X.shape")
        print("td-idf char 3-5grams...")
        #td-idf char 3-5grams
        file_name="cache/"+language+"_Atalay_vectorizer_char_tfidf_cache.pickle"
        if os.path.isfile(file_name):
            self.vectorizer_char_tfidf = pickle.load(open(file_name,'rb'))
        else:
            self.vectorizer_char_tfidf = TfidfVectorizer(ngram_range=(3, 5),
                                 analyzer="char",
                                 # stop_words="english",
                                 lowercase=True,
                                 binary=True,
                                 max_features=500000)
            self.vectorizer_char_tfidf.fit(to_be_vectorized)
            pickle.dump(self.vectorizer_char_tfidf,open(file_name,'wb'))
        X = csr_matrix(hstack((X,self.vectorizer_char_tfidf.transform(to_be_vectorized))))
        print(X.shape,"X.shape")

        #tweet embeddings
        print("tweet embeddings...")
        """file_name="cache/"+language+"_Atalay_sentence_embedding_cache.pickle"
        if os.path.isfile(file_name):
            self.sentence_embeddings=pickle.load(open(file_name,'rb'))
        else:
            i=len(to_be_vectorized)
            self.sentence_embeddings=[]
            for text in to_be_vectorized:
                i-=1
                print(i)
                self.sentence_embeddings.append(self.nlp(text).vector)
            pickle.dump(self.sentence_embeddings,open(file_name,'wb'))
        X = csr_matrix(hstack((X,self.sentence_embeddings)))"""
        file_name="cache/"+language+"_Atalay_fasttext_sentence_embedding_cache.pickle"
        if os.path.isfile(file_name):
            self.sentence_embeddings=pickle.load(open(file_name,'rb'))
        else:
            i=len(to_be_vectorized)
            self.sentence_embeddings=[]
            for text in to_be_vectorized:
                i-=1
                #print(i)
                self.sentence_embeddings.append(self.fasttext_model.get_sentence_vector(text))
            pickle.dump(self.sentence_embeddings,open(file_name,'wb'))
        X = csr_matrix(hstack((X,self.sentence_embeddings)))
        print(X.shape,"X.shape")


        print("training the model...")
        #crea modello
        file_name="cache/"+language+"_Atalay_clf_cache.pickle"
        if os.path.isfile(file_name):
            self.clf = pickle.load(open(file_name,'rb'))
        else:
            self.clf = SVC(kernel="linear") #kernel='rbf'
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
        X_test = self.vectorizer_bow_binary.transform(cleaned_texts)
        print(X_test.shape,"X_test.shape")
        X_test = csr_matrix(hstack((X_test,self.vectorizer_bow_tfidf.transform(cleaned_texts))))
        print(X_test.shape,"X_test.shape")
        X_test = csr_matrix(hstack((X_test,self.vectorizer_char_binary.transform(cleaned_texts))))
        print(X_test.shape,"X_test.shape")
        X_test = csr_matrix(hstack((X_test,self.vectorizer_char_tfidf.transform(cleaned_texts))))
        print(X_test.shape,"X_test.shape")
        """X_test = csr_matrix(hstack((X_test,[self.nlp(text).vector for text in texts])))
        print(X_test.shape,"X_test.shape") """
        X_test = csr_matrix(hstack((X_test,[self.fasttext_model.get_sentence_vector(text) for text in texts])))
        print(X_test.shape,"X_test.shape")


        test_predict = self.clf.predict(X_test)
        #print(test_predict)
        counted_prediction = counter(test_predict)

        return ["hs_yes"], [counted_prediction[1]]


if __name__ == '__main__':

    language="es"
    print(language)
    hateeval = Atalay(language)
    texts=["RT #USER#: jurisprudencia Felices 119 años, #USER#. Historia que tú hiciste, historia por hacer. Hala Madrid!!❤️ #URL#",
           "Un moro invasor asesina a una española en Dúrcal; la izquierda vuelve a encubrir y proteger al asesino por ser inmigrante."]
    #texts=["@Ushimama1 @MielCaldero @USMC It is OBVIOUS from your tweet that you are too LOW IQ to understand or appreciate ""immigration done the LEGAL way"" so go whine to somone else. #Snowflakes#StopIllegalImmigration#BuildThatWall#MAGA",
    #       "It's easy to feel like you can't make a difference when the injustice of the world seems so overwhelming. But if we all do our part, we will see things change. Do your part for refugees this week  and help them rebuild their lives. âž https://t.co/QzFjSqWwJ8"]

    print(hateeval.predict_user(texts))
