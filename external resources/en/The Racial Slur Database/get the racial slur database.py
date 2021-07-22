import urllib.request
from bs4 import BeautifulSoup
import urllib.parse
import re
from operator import itemgetter
import csv
url='http://www.rsdb.org/full'
site = urllib.request.Request(url)
page = urllib.request.urlopen(site)


soup = BeautifulSoup(page,'html.parser',from_encoding="iso-8859-5")
trs = soup.find_all('tr')
file=open("rsdb.csv","w")
filecsv=csv.writer(file,delimiter=",",quotechar="\"")
categories={}
for tr in trs:
    if len(tr.find_all('th'))>0:
        continue
    tds = tr.find_all('td')
    slur=tds[0]
    #print(urllib.parse.unquote(slur.find('a')['href'].split("/")[-1]))
    slur=slur.find('a').getText().lower()
    slur=re.sub(r'\(*\)', '', slur)
    for s in slur.split("/"):
        s=s.strip().lower()
        represents = tds[1]
        represents = represents.find('a')['href'].split("/")[-1]
        represents = urllib.parse.unquote(represents)
        represents = re.sub(r'[^\w\s]', '', represents)
        represents = represents.replace("_"," ").strip()
         #print(represents)
        reasonOrigins=tds[2]
        print(s,represents)
        if represents not in categories:
            categories[represents]=0
        categories[represents]+=1
        filecsv.writerow([s,represents])
print(dict( sorted(categories.items(), key=itemgetter(1),reverse=True)))
