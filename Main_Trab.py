import requests
from bs4 import BeautifulSoup
from requests_oauthlib import OAuth1
import pandas as pd
import Pyhton_Learning.Modulos as Modulos
import TVEmbeddingWork.model_PVDBOW
from sklearn.model_selection import train_test_split
import TVEmbeddingWork.model_FasText

RawData_path = "Tweet_Data/RawData.csv"
ProcessedData_path = "Tweet_Data/InputData.csv"
Train_path = "Tweet_Data/TrainData.csv"
test_path = "Tweet_Data/TestData.csv"
Model_path = 'Tweet_Data/'

##################### SET API KEYS ############################
consumer_key = '6oc1TPJ8hHKxNa0itYbfIqkxv'
consumer_secret = 'DDfBJp9tPegR0e1b1i6v3KU2OGoNKSIrT4AKWWZJPre1UWIYpg'
token_key = '1117789254212554752-s2fWlZZix4NtBKYg7hEkgx63otFOib'
token_secret = 'Pz1Sfff4xkJmPSY7Gny8dYOFZDGGfrprfrtWbSGEHwYzx'

auth_params = {
    'app_key':'6oc1TPJ8hHKxNa0itYbfIqkxv',
    'app_secret':'DDfBJp9tPegR0e1b1i6v3KU2OGoNKSIrT4AKWWZJPre1UWIYpg',
    'oauth_token':'1117789254212554752-s2fWlZZix4NtBKYg7hEkgx63otFOib',
    'oauth_token_secret':'Pz1Sfff4xkJmPSY7Gny8dYOFZDGGfrprfrtWbSGEHwYzx'
}

# Creating an OAuth Client connection
auth = OAuth1 (
    auth_params['app_key'],
    auth_params['app_secret'],
    auth_params['oauth_token'],
    auth_params['oauth_token_secret']
)


##################### QUERY TWEETS ###########################

url_rest = "https://api.twitter.com/1.1/search/tweets.json"


preFilter = 'reforma previdencia -filter:retweets -filter:replies' #


params = {'q': preFilter, 'count': 100000, 'lang': 'pt',  'result_type': 'recent'}
results = requests.get(url_rest, params=params, auth=auth)

tweets = results.json()

messages = [BeautifulSoup(tweet['text'], 'html5lib').get_text() for tweet in tweets['statuses']]


pd.DataFrame(messages).to_csv(RawData_path,sep='\t')
#-----------------------------------------------------------------------------------------------------------------------------------##

################################## PRE PROCESSING #############################################

aux = []
hasht = []
for line in messages:
    aux.append(str(line))
    hasht.append(Modulos.get_hashtags(line))
hasht = pd.DataFrame(hasht)

df = []
df = pd.DataFrame(df)

tweet_processed_text=[]
for line in aux:
    line = Modulos.remover_acentos(line)
    line=Modulos.deEmojify(line)
    tweet_processed_text.append(str(Modulos.preprocess(line.rstrip())))

tweet_processed_text=pd.DataFrame(tweet_processed_text)
pd.DataFrame(tweet_processed_text).to_csv(ProcessedData_path,sep='\t')

#################################### Rotulagem Manual############################################
## Split base rotulada manualmente ##
df = pd.read_csv('/Users/sergiojunior/PycharmProjects/TVEmbeddingWork/Tweet_Data/InputDataSA.csv',sep=";")
df = pd.DataFrame(df)
train, test=train_test_split(df,test_size=0.1,random_state=123)
#-------------------------------------------------------------------------------------------------------------------------------------'''
##################### Embedding pelo PV-DBOW ####################
# w1:janela de palavras
# e1:n epochs
# cv:cross validation
# vs:vector size

TVEmbeddingWork.model_PVDBOW.build_PVDBOW(train['0'], train['1'], test['0'], test['1'],w1=1,e1=100,cv=3,vs=5)


'''-----------------------------------------------------------------------------------------------------------------------------------'''

##################### Embedding pelo Fast Text ####################

TVEmbeddingWork.model_FasText.build_FastText('/Users/sergiojunior/PycharmProjects/TVEmbeddingWork/Tweet_Data/InputDataSA.csv',train['0'], train['1'], test['0'], test['1'],e=100,k=1,cv=3,n=1)