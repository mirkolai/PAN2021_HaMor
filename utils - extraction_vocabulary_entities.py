import glob
import re
from collections import Counter
import pandas as pd
import spacy
path = "/home/nbdotti62/PycharmProjects/PAN/moral_values-AT-pan21-author-profiling-training/external resources/en/wikipedia/entities/*"

nlp = spacy.load('en_core_web_lg')
counter = Counter

folders = glob.glob(path)
l = list()
for folder in folders:
    print(folder)
    files = glob.glob(folder+'/*')
    for file in files:
        f = open(file)
        l.extend([x[9:-1] for x in f if x.startswith('Category:')])
        #print(l)
s = str()
counted = Counter(l)
en_pattern = re.compile('emigrants\sto\sthe\sunited\sstates|(?=.*african-american|women|lgbt|muslim)(?=.*people|writers|politicians|singers|players|scientists|musicians|legistrators|members|journalists|rappers).*|feminist')
es_pattern = re.compile('musulmanes|emigrantes|escritoras|gobernadoras|lesbianas|políticas|feministas|(?=.*mujeres|lgbt)(?=.*activistas|futbolistas|deportistas|personas|periodistas|pianistas|transgénero|políticos|escritores).*')

### CHANGE en_pattern TO es_pattern IN ORDER TO FILTER THE LIST OF CATEGORIES FOR SPANISH LANGUAGE ###
x = {k:v for k,v in counted.items() if re.search(en_pattern,k.lower())}
for k in x:
    s=s+k+' '
print(s)
tokenized = [x.text for x in nlp(s)]
c = counter(tokenized)
print(c)

#print(counted)
df = pd.DataFrame.from_dict(x,orient='index').reset_index()
df.to_csv('vocabulary_en.csv',index=False)
