__author__ = 'mirko'
import re
import time
import emoji
import numpy
import glob

import pandas as pd
import spacy
import os
import xml.etree.ElementTree as ET
import wikipedia
import urllib.request
from bs4 import BeautifulSoup
def clean(text):
    text=re.sub("#\w*#"," ",text)
    text=give_emoji_free_text(text)
    return text
def give_emoji_free_text(text):
    allchars = [str for str in text]
    emoji_list = [c for c in allchars if c in emoji.UNICODE_EMOJI]
    clean_text = ' '.join([str for str in text.split() if not any(i in str for i in emoji_list)])

    return clean_text


class NerWikipedia(object):

    #nlp = spacy.load("en_core_web_lg")
    #nlp = spacy.load("es_core_news_lg")
    languages = { 'en' : "en_core_web_lg",'es' : "es_core_news_lg" }
    language=None
    nlp=None
    vocabulary_entities=None
    people=None
    def __init__(self,language):
        print("Loading NER profile for ",language)
        self.language=language
        self.nlp=spacy.load(self.languages[language])
        self.vocabulary_entities = pd.read_csv('external resources/'+self.language+'/wikipedia/target_people_'+self.language+'.csv')
        self.people = self.vocabulary_entities['index'].tolist()
    def get_user_entities(self,user_id,test=False):
        entities=[]
        cache="external resources/"+self.language+"/wikipedia/ner/"+user_id+".txt"
        if os.path.isfile(cache):
            print("ner form cache")
            cache_file=open(cache)
            for row in cache_file:
                row=row.replace("\n","")
                tweet_entities=[]
                for col in row.split("******"):
                    if len(col.strip())>0:
                        tweet_entities.append(col)
                entities.append(tweet_entities)
            cache_file.close()
        else:
            cache_file=open(cache,"w")
            if test:
                print("test......")
                user_file = open("pan21-author-profiling-test-without-gold/"+self.language+"/"+user_id+".xml")
            else:
                print("train......")
                user_file = open("pan21-author-profiling-training-2021-03-14/"+self.language+"/"+user_id+".xml")

            tree = ET.parse(user_file)
            root = tree.getroot()
            i=0
            for document in root[0]:
                i+=1
                text=clean(document.text)
                tokenized = self.nlp(text)
                twitter_entities=[]
                for ent in tokenized.ents:
                    if ent.label_ == 'PERSON' or ent.label_ == 'PER':
                        if len(ent.text.replace(" ",""))>0:
                            twitter_entities.append(ent.text)
                            cache_file.write(ent.text+"******")
                entities.append(twitter_entities)
                if i <200:
                    cache_file.write("\n")
            cache_file.close()
            user_file.close()
        return entities

    def get_user_wikipedia_categories(self,entities):

        target_people = []
        categories=[]
        ent = 0
        for twt_entity in entities:
            #print(twt_entity)
            i = 0
            if len(twt_entity)==0:
                target_people.append(i)
            else:
                ent+=len(twt_entity)
                for entity in twt_entity:
                    #print(entity)
                    cache = "external resources/"+self.language+"/wikipedia/entities/"+entity
                    pages=[]
                    if os.path.exists(cache):
                        print("recovering ",entity,"from cache")
                        page_urls=glob.glob(cache+"/")
                        for page_url in page_urls:
                            pages.append(page_url.split("/")[-1])
                    else:
                        print("recovering ",entity," from wikipedia ")

                        try:
                            pages = wikipedia.search(entity)
                            print(pages)
                            time.sleep(1)
                            direc = os.mkdir(cache)
                            #save in cache:
                            for page_url in pages:
                                file_out=open(cache+"/"+page_url,"w")
                                file_out.close()
                            ###########

                        except Exception as e:
                            page_url=[]
                            print("error ",e)
                            continue
                        targeted = any([page_url for page_url in pages if page_url in self.people])
                        print(targeted)
                        if targeted: i += 1
                    #print("pages",pages)
                    #aggiungere stringa con tutte le categorie

                target_people.append(i)
        if len(target_people)==0:
            target_people.append(0)
        if ent>0:
            print(numpy.mean(target_people))
            print(numpy.sum(target_people))
            print(ent)
            print(numpy.std(target_people))
            print(numpy.sum(target_people)/ent)
            return ["ner_mean"]+\
                ["targ_tot"]+\
                ["ner_std"]+["ent_tot"]+["ner_prop"],[
                numpy.mean(target_people),
                numpy.sum(target_people),
                numpy.std(target_people),
                ent,
                numpy.sum(target_people)/ent]
        else:
            print(numpy.mean(target_people))
            print(numpy.sum(target_people))
            print(ent)
            print(numpy.std(target_people))
            print(numpy.sum(target_people)/ent)
            return ["ner_mean"] + \
                   ["targ_tot"] + \
                   ["ner_std"]+["ent_tot"]+ ["ner_prop"], [
                   numpy.mean(target_people),
                   numpy.sum(target_people),
                   numpy.std(target_people),
                   ent,
                   0
            ]






if __name__ == '__main__':

    texts=["@tedcruz And, Clinton #HandOverTheServer she wiped clean + 30k deleted emails, explains dereliction of duty/lies re #Benghazi,etc #tcot #SemST",
           "All these fears about Obama but a total refusal to address the problem is NONWHITE IMMIGRANTS"]

    nerWiki_en = NerWikipedia("en")
    entities = nerWiki_en.get_user_entities(texts)
    print(entities)
    categories = nerWiki_en.get_user_wikipedia_categories(entities)
    print(categories)
    '''directory = glob.glob("pan21-author-profiling-training-2021-03-14/"+nerWiki_en.language+"/*.xml")
    for f in directory:
        user_id = f.split('/')[-1][:-4]
        entities = nerWiki_en.get_user_entities(texts)
        print(entities)
        # entities=["Obama","Trump","Bobby Jindal"]
        categories = nerWiki_en.get_user_wikipedia_categories(entities)
        #print(categories)'''

'''        nerWiki_es = NerWikipedia("es")
        user_id = '0a15953f63e4f4ddd8ba43680b74bcc1'
        entities = nerWiki_en.get_user_entities(user_id)
        print(entities)
        # entities=["Obama","Trump","Bobby Jindal"]
        categories = nerWiki_en.get_user_wikipedia_categories(entities)
        print(categories)
'''

