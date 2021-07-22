import urllib.request
from bs4 import BeautifulSoup
import re
import pandas as pd


def make_wiki_link(a_string):
    a_string = re.sub(' ','_',a_string)
    url = 'https://en.wikipedia.org/wiki/Category:'+a_string

    return url

def get_wiki_html(an_url):
    page = urllib.request.urlopen(an_url)
    soup = BeautifulSoup(page, 'html.parser')

    return soup

def wiki_category_scraping(a_html):
    div = a_html.find('div',id='mw-pages')
    entities = div.find_all('li')
    names = [x.text for x in entities]
    return names


def multi_url(an_url,a_list):
    html = get_wiki_html(an_url)
    next = html.find_all('a', title='Category:' + s)
    link = [x['href'] for x in next if x.text == 'next page']
    #print(l)
    if len(link) > 0:
        a_list.append(an_url)
        an_url = 'https://en.wikipedia.org/' + link[0]
        multi_url(an_url,a_list)
    else:
        a_list.append(an_url)
        return a_list

    return a_list

df = pd.read_csv('/home/nbdotti62/PycharmProjects/PAN/vocabulary_en.csv')

target_df = pd.DataFrame()
for item in df.iloc[:].values:
    s = item[0]
    lista = list()
    url = make_wiki_link(s)
    multi = multi_url(url, lista)
    for x in multi:

            html = get_wiki_html(x)
            scraped = wiki_category_scraping(html)
            d ={k:s for k in scraped}
            temp_df = pd.DataFrame.from_dict(d,orient='index').reset_index()

            target_df= target_df.append(temp_df)
            #print(target_df)
            #print(d)

target_df = target_df.drop_duplicates(subset=['index'])
target_df.to_csv('target_peoples.csv',index=False)





#html = get_wiki_html(url)






#categories = wiki_category_scraping(html,s)


#boh = html.find('a',title='Category:African-American female singers')
