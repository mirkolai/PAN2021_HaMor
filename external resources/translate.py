import requests
import json
import glob
import os.path
import time
import re
def translateLemma(lemma):

    service_url = 'https://babelnet.io/v6/getSenses'

    key  = '<YOUR KEY>'

    params = {
        'lemma' : lemma,
        'searchLang' : 'en',
        'targetLang' : "es",
        'source' : "MCR_ES",
        'key'  : key
    }

    response = requests.post(service_url, data=params)
    response.encoding = 'utf-8'

    try:
        if response.status_code == 200:
            data = json.loads(response.text)
            print(json.dumps(data))
            result=data[0]['properties']['lemma']['lemma']
            return result
    except:
        return None


lemmas={}
#Translate Morality in Knowledge Graph
files_names=glob.glob("../external resources/en/Morality-in-Knowledge-Graphs-master/MFD-Linking/conceptnet/aut*")
for file_name in files_names:
    output_file_name=file_name.replace("/en/","/es/")
    if not os.path.isfile(output_file_name):
        input=open(file_name)
        output=open(output_file_name,"w")
        for row in input:
            row=row.replace("\n","")
            if row not in lemmas:
                result=translateLemma(row)
                time.sleep(3)
                lemmas[row]=result
            else:
                result=lemmas[row]
            print(row,result)
            if result is not None:
                output.write(result+"\n")
            else:
                output.write("NULL\t"+row+"\n")

        input.close()
        output.close()


##tradurre anche the ratial slur
lemmas={}
#Translate Morality in Knowledge Graph
files_names=glob.glob("../external resources/en/The Racial Slur Database/rsdb.csv")
for file_name in files_names:
    print(file_name)
    output_file_name=file_name.replace("/en/","/es/")
    if not os.path.isfile(output_file_name+"r"):
        input=open(file_name)
        output=open(output_file_name,"a")
        for row in input:
            row=row.replace("\n","")
            row=row.split(",")
            if row[1] not in  ['blacks',
                               'asians',
                               'mixed races',
                               'jews',
                               'arabs', 
                               'hispanics', 
                               'mexicans', 
                               'chinese', 
                               'muslims', 
                               ]:
                continue
            print(row)
            if row[0] not in lemmas:
                if len(re.findall("[0-9]{1,}",row[0]))>0:
                    result=None
                    continue
                result=translateLemma(row[0])
                time.sleep(3)
                lemmas[row[0]]=result
            else:
                result=lemmas[row[0]]
            print(row,result)
            if result is not None:
                output.write(result.lower()+","+row[1]+"\n")
            else:
                output.write(row[0]+","+row[1]+"\n")

        input.close()
        output.close()
