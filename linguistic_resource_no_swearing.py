__author__ = 'mirko'
import re
import numpy
import glob
import spacy
import pickle
import os


class NoSwearing(object):

    pattern_split = re.compile(r"\W+")
    no_swearing={}
    concepts=[]
    languages = { 'en' : "en_core_web_lg",'es' : "es_core_news_lg" }
    language=None
    nlp=None

    def __init__(self,language):
        print("Loading no swearing for "+language+"...")
        self.nlp=spacy.load(self.languages[language])
        self.stop_words={}
        for word in self.nlp.Defaults.stop_words:
            self.stop_words[word]=0

        self.no_swearing = []
        self.language=language
        if os.path.isfile("cache/"+language+"_similar_words_cache.pickle"):
            infile = open("cache/"+language+"_similar_words_cache.pickle",'rb')
            self.similar_words_cache = pickle.load(infile)
            infile.close()
        else:
            self.similar_words_cache = {}
        file_names=glob.glob("external resources/"+language+"/noswearing/noswearing.txt")
        #print(language,len(file_names))
        for file_name in file_names:
            concept=file_name.split("/")[-1].replace(".txt","")
            self.no_swearing=[]
            file=open(file_name)
            for line in file:
                word=line.replace("\n","").replace("_"," ")
                if word in self.nlp.vocab.strings:
                    self.no_swearing.append(word)
        return

    def spacy_most_similar(self, word, topn=10):
        #print("spacy_most_similar",word)
        try:
            ms = self.nlp.vocab.vectors.most_similar(numpy.asarray([self.nlp.vocab.vectors[self.nlp.vocab.strings[word]]]), n=topn)
        except:
            return ()
        return set([self.nlp.vocab.strings[w].lower() for w in ms[0][0]])


    def get_text_no_swearing_profile(self,text):

        values=[0]

        tokens = self.pattern_split.split(text.lower())
        #print("tokens",tokens)
        for token in tokens:

            if token == "" or token in self.stop_words:
                continue
            #print("token",token)
            for word in self.no_swearing:
                if token in word:
                    values[0]+=1
                else:
                    if word not in self.similar_words_cache:
                        self.similar_words_cache[word]=self.spacy_most_similar(word, topn=10)
                        output=open("cache/"+self.language+"_similar_words_cache.pickle", "wb")
                        pickle.dump(self.similar_words_cache,output)
                        output.close()
                    #print(token, word, self.similar_words_cache[word])
                    if token in self.similar_words_cache[word]:
                        values[0]+=1
                        break
        return "no_swearing", values

    def get_user_no_swearing_profile(self,texts):
        values=[]
        for text in texts:

            value=[0]
            tokens = self.pattern_split.split(text.lower())
            #print(self.nlp.Defaults.stop_words)
            for token in tokens:
                if token == "" or token in self.stop_words:
                    continue
                #print("token:",token)
                for word in self.no_swearing:
                    if token in word:
                        value[0]+=1
                        #print(token,word)
                        break
                    else:
                        if word not in self.similar_words_cache:
                            self.similar_words_cache[word]=self.spacy_most_similar(word, topn=10)
                        #print(token, word, self.similar_words_cache[word])
                        if token in self.similar_words_cache[word]:
                            value[0]+=1
                            #print(token,word)
                            break
            values.append(value)
        return ["no_swearing_mean"]+\
               ["no_swearing_tot"]+\
               ["no_swearing_std"], numpy.concatenate(
               (numpy.mean(numpy.array(values),axis=0),
                numpy.sum(numpy.array(values),axis=0),
                numpy.std(numpy.array(values),axis=0))
                )


if __name__ == '__main__':


    noS = NoSwearing("en")
    print("en")
    concepts, values=noS.get_text_no_swearing_profile("@tedcruz And, #HandOverTheServer she wiped clean + 30k deleted emails, explains dereliction of duty/lies re #Benghazi,etc #tcot #SemST")
    print("concepts", "values")
    print(concepts)
    print(values)
    texts=["@tedcruz And, arsehole #HandOverTheServer she wiped clean + 30k deleted emails, explains dereliction of duty/lies re #Benghazi,etc #tcot #SemST",
           "All these fears about safety but a total refusal to address the problem is NONWHITE IMMIGRANTS"]
    concepts, values=noS.get_user_no_swearing_profile(texts)
    print("concepts", "values")
    print(concepts)
    print(values)
    noS_es = NoSwearing("es")
    print("es")
    concepts, values=noS_es.get_text_no_swearing_profile("RT #USER#: Felices 119 años, #USER#. Historia que tú hiciste, historia por hacer. Hala Madrid!!❤️ #URL#")
    print("concepts", "values")
    print(concepts)
    print(values)
    texts=["RT #USER#: jurisprudencia Felices 119 años, #USER#. Historia que tú hiciste, historia por hacer. Hala Madrid!!❤️ #URL#",
           "#USER# illegalization Vamos sin prisa pero vamos con todo ! #URL#"]
    concepts, values=noS_es.get_user_no_swearing_profile(texts)
    print("concepts", "values")
    print(concepts)
    print(values)
