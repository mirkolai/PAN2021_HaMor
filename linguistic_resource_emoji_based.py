__author__ = 'mirko'
import re
import numpy
import glob
import spacy
import os
import xml.etree.ElementTree as ET
import wikipedia
import urllib.request
import emoji
from bs4 import BeautifulSoup
def clean(text):
    text=re.sub("#\w*#"," ",text)
    return text

class EmojiFeature(object):

    #nlp = spacy.load("en_core_web_lg")
    #nlp = spacy.load("es_core_news_lg")
    languages = { 'en' : "en_core_web_lg",'es' : "es_core_news_lg" }
    language=None
    nlp=None
    def __init__(self,language):
        print("Loading emoji profile...")
        self.language=language
        self.nlp=spacy.load(self.languages[language])

    def get_user_emoji_features(self, texts):
        emojis='BLANK '
        for text in texts:
            emojis+=' '.join(c for c in text if c in emoji.UNICODE_EMOJI)
        return emojis

    def get_bio_features(self, texts):
        emojis='BLANK '
        for text in texts:
            emojis+=' '.join(c for c in text if c in emoji.UNICODE_EMOJI)

        return ["LightSkinTone","MediumLight","MediumSkin","MediumDark","DarkSki",
                "woman","man",
                "advertising",
                "fire",
                "anger",
                "religious",
                "nationalism",
                "brexit",
                #"anti-feminism",
                #"anti-semitism",
                #"love"
                ],\
               [emojis.count("🏻"),emojis.count("🏼"),emojis.count("🏽"),emojis.count("🏾"),emojis.count("🏿"),
                emojis.count("♀"),emojis.count("♂"),#45
                emojis.count("📣")+emojis.count("🔴")+emojis.count("🚨"),#45
                emojis.count("🔥")+emojis.count("💥"), #515
                emojis.count("🤬")+emojis.count("😡")+emojis.count("👿"),#52
                emojis.count("🙏")+emojis.count("✝"),#52.5
                emojis.count("🇺 🇸"),#54
                emojis.count("🇬 🇧"),#55
                #emojis.count("🍆")+emojis.count("💦"),
                #emojis.count("✡"),
                #emojis.count("😘")+emojis.count("😍")+emojis.count("🥰")+emojis.count("💕")+emojis.count("💖"),
                ]



if __name__ == '__main__':
    language="en"
    gender=EmojiFeature(language)
    file_names=glob.glob("pan21-author-profiling-training-2021-03-14/"+language+"/*.xml")
    emojis_dic={}
    for file_name in file_names:
        xml_file = open(file_name, 'r')
        data = xml_file.read()
        soup = BeautifulSoup(data, "xml")


        author = soup.find('author')
        lang=author['lang']
        label=author['class']
        author_id=file_name.split("/")[-1].replace(".xml","")
        documents = author.find_all('document')
        i=0
        emojis=''
        for document in documents:
            for c in document.text:
                if c in emoji.UNICODE_EMOJI:
                    if c not in emojis_dic:
                        emojis_dic[c]=0
                    emojis_dic[c]+=1
        import operator
    print(sorted(emojis_dic.items(), key=operator.itemgetter(1),reverse=True))
    """print(emojis)
    print("man ",emojis.count("♂"))
    print("woman ",emojis.count("♀"))
    print("Light Skin Tone ",emojis.count("🏻"))
    print("Medium-Light ",emojis.count("🏼"))
    print("Medium Skin ",emojis.count("🏽"))
    print("Medium-Dark ",emojis.count("🏾"))
    print("Dark Ski ",emojis.count("🏿"))
    print("Religius ",emojis.count("🙏"))
    print("Advertising ",emojis.count("📣")+emojis.count("🔴")+emojis.count("🚨"))
    print("fire ",emojis.count("🔥"))
    print("Angry ",)"""


    #print("USA ",emojis.count("🇺 🇸"))
