import requests
import pandas as pd

def get_response(url, params):
    resp = requests.get(url, params)
    return resp.json()

url = 'https://es.wikipedia.org/w/api.php?'
params = {
    "action": "query",
    "cmprop":"ids|title",
    "cmtitle":'Category:Nigerian_writers',
    "cmlimit": "150",
    "list": "categorymembers",
    "format": "json",
    #"cmcontinue":"page|NNNN|TITLE",
}


df = pd.read_csv('vocabulary_es.csv')
target_df =pd.DataFrame(columns=['person','category'])

for category in df.iloc[:].values:
    #print(category)
    params = {
        "action": "query",
        "cmprop": "ids|title",
        "cmtitle": 'Categoría:'+category[0],
        "cmlimit": "150",
        "list": "categorymembers",
        "format": "json",
        # "cmcontinue":"page|NNNN|TITLE",
    }

    while True:

        json_resp = get_response(url, params)
        #print(json_resp)
        for x in json_resp['query']['categorymembers']:
            target_df = target_df.append({'person':x['title'],'category':category[0]},ignore_index=True)
            #print(x['title'],x['pageid'])

        if 'continue' in json_resp:
            params['cmcontinue'] = json_resp["continue"]['cmcontinue']
            json_resp = get_response(url, params)
            #print(json_resp)
        else:
            break
target_df = target_df.drop_duplicates()
target_df.to_csv('target_people_es.csv',index=False)