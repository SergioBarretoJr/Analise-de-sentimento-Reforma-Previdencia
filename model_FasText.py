
import fastText
import sys
import os
import nltk
nltk.download('punkt')
import csv
import datetime
from bs4 import BeautifulSoup
import re
import itertools
import pandas as pd
import numpy as np
import emoji
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


def transform_instance(row):
    cur_row = []
    #Prefix the index-ed label with __label__
    label = "__label__" + row[1]
    cur_row.append(label)
    cur_row.extend(row[0].lower()).split()
    return cur_row

'''training_data_path ='/Users/sergiojunior/PycharmProjects/FastText-sentiment-analysis-for-tweets-master/tweets_treino.csv'
validation_data_path ='/Users/sergiojunior/PycharmProjects/FastText-sentiment-analysis-for-tweets-master/tweets_validation.csv'
model_path ='/Users/sergiojunior/PycharmProjects/FastText-sentiment-analysis-for-tweets-master/'
model_name="model-pt"'''

def build_FastText(training_data_path,xt1,yt1,xv1,yv1,e,k,cv,n):
    print('Training start')

    hyper_params = {"lr": 0.01,
                    "epoch":e,
                    "wordNgrams": n,
                    "dim": 5,
                    "minCount": 1,
                    "loss":'softmax',
                    "ws": k,
    }

    print(str(datetime.datetime.now()) + ' START=>' + str(hyper_params) )

    # Train the model.
    model = fastText.train_supervised(input=training_data_path, **hyper_params)
    print("Model trained with the hyperparameter \n {}\n".format(hyper_params))

    # CHECK PERFORMANCE
    print(str(datetime.datetime.now()) + 'Training complete.' + str(hyper_params) )
    #validationcv=pd.read_csv('/Users/sergiojunior/PycharmProjects/TVEmbeddingWork/Tweet_Data/InputDataSA2.csv',sep=';')
    #xt1=pd.DataFrame(xt1)
    #yt1=pd.DataFrame(yt1)
    #xv1 =pd.DataFrame(xv1)
    #yv1 =pd.DataFrame(yv1)
    xv=[]
    yv=[]
    for line in xv1:

        xv.append(model.get_sentence_vector(line))
    for line in yv1:
        yv.append(line)


    xt = []
    yt = []
    for line in xt1:

        xt.append(model.get_sentence_vector(line))
    for line in yt1:
        yt.append(line)
    #print(xv)
    modelKNN = KNeighborsClassifier()
    modelKNN.fit(xt, yt)
    scores = cross_val_score(modelKNN, xv, yv,scoring='accuracy',cv=cv)
    #print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))