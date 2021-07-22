__author__ = 'mirko'
import re
import numpy
import glob
import spacy
import pickle
import os


class MoralityInKnowledgeGraph(object):

    pattern_split = re.compile(r"\W+")
    morality_conceptnet={}
    concepts=[]
    languages = { 'en' : "en_core_web_lg",'es' : "es_core_news_lg" }
    language=None
    nlp=None

    def __init__(self,language):
        print("Loading morality profile for "+language+"...")
        self.nlp=spacy.load(self.languages[language])
        self.stop_words={}
        for word in self.nlp.Defaults.stop_words:
            self.stop_words[word]=0
        self.morality_conceptnet = {}
        self.concepts=[]
        self.language=language
        if os.path.isfile("cache/"+language+"_similar_words_cache.pickle"):
            infile = open("cache/"+language+"_similar_words_cache.pickle",'rb')
            self.similar_words_cache = pickle.load(infile)
            infile.close()
        else:
            self.similar_words_cache = {}
        file_names=glob.glob("external resources/"+language+"/Morality-in-Knowledge-Graphs-master/MFD-Linking/conceptnet/*")
        #print(language,len(file_names))
        for file_name in file_names:
            concept=file_name.split("/")[-1].replace(".txt","")
            if concept not in ["authority","authorityvirtue","authorityvice",
                               "loyalty","loyaltyvirtue","loyaltyvice",
                               ]:
                continue
            self.concepts.append(concept)
            self.morality_conceptnet[concept]=[]
            file=open(file_name)
            for line in file:
                word=line.replace("\n","").replace("_"," ")
                if not word.startswith("NULL"):
                    #if word not in self.morality_conceptnet:
                    if word in self.nlp.vocab.strings:
                        self.morality_conceptnet[concept].append(word)
        return

    def spacy_most_similar(self, word, topn=10):
        #print("spacy_most_similar",word)
        try:
            ms = self.nlp.vocab.vectors.most_similar(numpy.asarray([self.nlp.vocab.vectors[self.nlp.vocab.strings[word]]]), n=topn)
        except:
            return ()
        return set([self.nlp.vocab.strings[w].lower() for w in ms[0][0]])


    def get_text_morality_profile(self,text):

        values=[0]*len(self.morality_conceptnet.keys())

        tokens = self.pattern_split.split(text.lower())
        #print("tokens",tokens)
        for token in tokens:
            if token == "" or token in self.stop_words:
                continue
            #print("token",token)
            for concept,words in self.morality_conceptnet.items():
                if token in words:
                    values[self.concepts.index(concept)]+=1
                else:
                    for word in words:
                        if word not in self.similar_words_cache:
                            self.similar_words_cache[word]=self.spacy_most_similar(word, topn=10)
                            output=open("cache/"+self.language+"_similar_words_cache.pickle", "wb")
                            pickle.dump(self.similar_words_cache,output)
                            output.close()
                        #print(token, word, self.similar_words_cache[word])
                        if token in self.similar_words_cache[word]:
                            values[self.concepts.index(concept)]+=1
                            break
        return self.concepts, values

    def get_user_morality_profile(self,texts):
        values=[]
        for text in texts:

            value=[0]*len(self.morality_conceptnet.keys())
            tokens = self.pattern_split.split(text.lower())
            for token in tokens:
                if token == "" or token in self.stop_words:
                    continue
                #print("token:",token)
                for concept,words in self.morality_conceptnet.items():
                    if token in words:
                        value[self.concepts.index(concept)]+=1
                    else:
                        for word in words:
                            if word not in self.similar_words_cache:
                                self.similar_words_cache[word]=self.spacy_most_similar(word, topn=10)
                            #print(token, word, self.similar_words_cache[word])
                            if token in self.similar_words_cache[word]:
                                value[self.concepts.index(concept)]+=1
                                break
            values.append(value)
        return [concept+"_mean" for concept in self.concepts]+\
               [concept+"_tot" for concept in self.concepts]+\
               [concept+"_std" for concept in self.concepts], numpy.concatenate(
               (numpy.mean(numpy.array(values),axis=0),
                numpy.sum(numpy.array(values),axis=0),
                numpy.std(numpy.array(values),axis=0))
                )


if __name__ == '__main__':


    mkg = MoralityInKnowledgeGraph("en")
    print("en")
    concepts, values=mkg.get_text_morality_profile("@tedcruz And, #HandOverTheServer she wiped clean + 30k deleted emails, explains dereliction of duty/lies re #Benghazi,etc #tcot #SemST")
    print("concepts", "values")
    print(concepts)
    print(values)
    texts=["@tedcruz And, arsehole #HandOverTheServer she wiped clean + 30k deleted emails, explains dereliction of duty/lies re #Benghazi,etc #tcot #SemST",
           "All these fears about safety but a total refusal to address the problem is NONWHITE IMMIGRANTS"]
    concepts, values=mkg.get_user_morality_profile(texts)
    print("concepts", "values")
    print(concepts)
    print(values)
    mkg_es = MoralityInKnowledgeGraph("es")
    print("es")
    print(mkg_es.morality_conceptnet.keys())
    concepts, values=mkg_es.get_text_morality_profile("RT #USER#: Felices 119 años, #USER#. Historia que tú hiciste, historia por hacer. Hala Madrid!!❤️ #URL#")
    print("concepts", "values")
    print(concepts)
    print(values)
    texts=["RT #USER#: jurisprudencia Felices 119 años, #USER#. Historia que tú hiciste, historia por hacer. Hala Madrid!!❤️ #URL#",
           "#USER# illegalization Vamos sin prisa pero vamos con todo ! #URL#"]
    concepts, values=mkg_es.get_user_morality_profile(texts)
    print("concepts", "values")
    print(concepts)
    print(values)
