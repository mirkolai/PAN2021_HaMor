__author__ = 'mirko'
import re
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
    return text

class CommunicationBehavior(object):

    #nlp = spacy.load("en_core_web_lg")
    #nlp = spacy.load("es_core_news_lg")
    languages = { 'en' : "en_core_web_lg",'es' : "es_core_news_lg" }
    language=None
    nlp=None
    def __init__(self,language):
        print("Loading Communication Behavior for",language)
        self.language=language
        self.nlp=spacy.load(self.languages[language])

    def get_user_communicative_styles(self,texts):

        total = len(texts)
        retweet = len([x for x in texts if x.startswith('RT #USER#')])
        reply = len([x for x in texts if x.startswith('#USER#')])
        own = total-retweet-reply

        return ["retweet"] + \
                ["reply"] + \
                ["own"], [retweet/total,
                                                reply/total,
                                                own/total]






if __name__ == '__main__':

    print("main")


