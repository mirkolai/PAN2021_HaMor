__author__ = 'mirko'
import re
import numpy
import spacy
import pickle
import os
import csv


class HurtLex(object):

    pattern_split = re.compile(r"\W+")
    word_category={}
    category=[]
    languages = { 'en' : "en_core_web_lg",'es' : "es_core_news_lg" }
    language=None
    nlp=None

    def __init__(self,language):
        print("Loading hurtLex for "+language+"...")
        self.nlp=spacy.load(self.languages[language])
        self.stop_words={}
        for word in self.nlp.Defaults.stop_words:
            self.stop_words[word]=0
        self.word_category = {}
        self.category=[]
        self.language=language
        if os.path.isfile("cache/"+language+"_similar_words_cache.pickle"):
            infile = open("cache/"+language+"_similar_words_cache.pickle",'rb')
            self.similar_words_cache = pickle.load(infile)
            infile.close()
        else:
            self.similar_words_cache = {}
        file_name="external resources/"+language+"/hurtlex/hurtlex.tsv"
        spamreader = csv.reader(open(file_name), delimiter='\t', quotechar='"')
        for row in spamreader:
            #print(row)
            level=row[0]
            if level !='inclusive':
                category=row[2]
                if category in ["ps","dmc","pr","om","svp","re"]:
                    if category not in self.category:
                        self.category.append(category)
                    if category not in self.word_category:
                        self.word_category[category]=[]
                    word=row[4]
                    if word in self.nlp.vocab.strings:
                            self.word_category[category].append(word)
        #print(self.word_category)
        return

    def spacy_most_similar(self, word, topn=10):
        #print("spacy_most_similar",word)
        try:
            ms = self.nlp.vocab.vectors.most_similar(numpy.asarray([self.nlp.vocab.vectors[self.nlp.vocab.strings[word]]]), n=topn)
        except:
            return ()
        return set([self.nlp.vocab.strings[w].lower() for w in ms[0][0]])


    def get_text_hurtlex_profile(self,text):

        values=[0]*len(self.word_category.keys())

        tokens = self.pattern_split.split(text.lower())
        #print("tokens",tokens)
        for token in tokens:
            if token == "" or token in self.stop_words:
                continue
            #print("token",token)
            for category,words in self.word_category.items():
                if token in words:
                    values[self.category.index(category)]+=1
                    break
                else:
                    for word in words:
                        if word not in self.similar_words_cache:
                            self.similar_words_cache[word]=self.spacy_most_similar(word, topn=10)
                            output=open("cache/"+self.language+"_similar_words_cache.pickle", "wb")
                            pickle.dump(self.similar_words_cache,output)
                            output.close()
                        #print(token, word, self.similar_words_cache[word])
                        if token in self.similar_words_cache[word]:
                            values[self.category.index(category)]+=1
                            break
        return self.category, values

    def get_user_hurtlex_profile(self,texts):
        values=[]
        for text in texts:

            value=[0]*len(self.word_category.keys())
            tokens = self.pattern_split.split(text.lower())
            for token in tokens:
                if token == "" or token in self.stop_words:
                    continue
                #print("token:",token)
                for concept,words in self.word_category.items():
                    if token in words:
                        value[self.category.index(concept)]+=1
                        break
                    else:
                        for word in words:
                            if word not in self.similar_words_cache:
                                self.similar_words_cache[word]=self.spacy_most_similar(word, topn=10)
                            #print(token, word, self.similar_words_cache[word])
                            if token in self.similar_words_cache[word]:
                                value[self.category.index(concept)]+=1
                                break
            values.append(value)
        return [concept+"_mean" for concept in self.category]+\
               [concept+"_tot" for concept in self.category]+\
               [concept+"_std" for concept in self.category], numpy.concatenate(
               (numpy.mean(numpy.array(values),axis=0),
                numpy.sum(numpy.array(values),axis=0),
                numpy.std(numpy.array(values),axis=0))
                )

if __name__ == '__main__':


    hl = HurtLex("en")
    print("en")
    print(hl.word_category.keys())
    concepts, values=hl.get_text_hurtlex_profile("@tedcruz And, #HandOverTheServer she wiped clean + 30k deleted emails, explains dereliction of duty/lies re #Benghazi,etc #tcot #SemST")
    print("concepts", "values")
    print(concepts, values)
    texts=["@tedcruz And, #HandOverTheServer she wiped clean + 30k deleted emails, explains dereliction of duty/lies re #Benghazi,etc #tcot #SemST",
           "All these fears about safety but a total refusal to address the problem is NONWHITE IMMIGRANTS"]
    concepts, values=hl.get_user_hurtlex_profile(texts)
    print("concepts", "values")
    print(concepts, values)

    hl_es = HurtLex("es")
    print("es")
    concepts, values=hl_es.get_text_hurtlex_profile("RT #USER#: Felices 119 años, #USER#. Historia que tú hiciste, historia por hacer. Hala Madrid!!❤️ #URL#")
    print("concepts", "values")
    print(concepts, values)
    texts=["RT #USER#: jurisprudencia Felices 119 años, #USER#. Historia que tú hiciste, historia por hacer. Hala Madrid!!❤️ #URL#",
           "#USER# illegalization proxenetismo Vamos sin prisa pero vamos con todo ! #URL#"]
    concepts, values=hl_es.get_user_hurtlex_profile(texts)
    print("concepts", "values")
    print(concepts, values)
