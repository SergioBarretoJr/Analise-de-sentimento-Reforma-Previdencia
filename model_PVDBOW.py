import logging
import random
import numpy as np
import pandas as pd
from gensim.models import doc2vec
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from gensim.models.doc2vec import Doc2Vec
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import cross_val_score

import codecs

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def label_sentences(corpus, label_type):

    labeled = []
    for i, v in enumerate(corpus):
        label = label_type + '_' + str(i)
        labeled.append(doc2vec.LabeledSentence(v.split(), [label]))
    return labeled

def get_vectors(doc2vec_model, corpus_size, vectors_size, vectors_type):
    vectors = np.zeros((corpus_size, vectors_size))
    for i in range(0, corpus_size):
        prefix = vectors_type + '_' + str(i)
        vectors[i] = doc2vec_model.docvecs[prefix]
    return vectors

def train_doc2vec(corpus,w,e,Vs):
    logging.info("Building Doc2Vec vocabulary")
    d2v = doc2vec.Doc2Vec(min_count=1,  #Ignores all words with total frequency lower than this
                          window=w,  # The maximum distance between the current and predicted word within a sentence
                          vector_size=Vs,  # Dimensionality of the generated feature vectors
                          workers=5,  # Number of worker threads to train the model
                          alpha=0.01,  # The initial learning rate
                          min_alpha=0.00025,  # Learning rate will linearly drop to min_alpha as training progresses
                          dbow_words=0,
                          hs=0,
                          seed=123,
                          dm=0,)  # dm=1 means 'distributed memory' (PV-DM:predict a center word from the randomly
                                                                        # sampled set of words by taking as input — 
                                                                        # the context words and a paragraph id.)
                                 # and dm =0 means 'distributed bag of words' (PV-DBOW: ignores the context words in
                                                                                      # the input, but force the model
                                                                                      # to predict words randomly sampled
                                                                                     # from the paragraph in the output.
    #print(corpus)
    d2v.build_vocab(corpus)

    logging.info("Training Doc2Vec model")
    for epoch in range(e):
        #logging.info('Training iteration #{0}'.format(epoch))
        d2v.train(corpus, total_examples=d2v.corpus_count, epochs=d2v.epochs)
        # shuffle the corpus
        random.shuffle(corpus)
        # decrease the learning rate
        d2v.alpha -= 0.0002


    logging.info("Saving trained Doc2Vec model")
    d2v.save("d2v.model")
    return d2v

def train_classifier(d2v, training_vectors, training_labels,Vs):
    logging.info("Classifier training")
    train_vectors = get_vectors(d2v, len(training_vectors), Vs, 'Train')
    model = KNeighborsClassifier()
    model.fit(train_vectors, np.array(training_labels))
    return model

def test_classifier(d2v, classifier, testing_vectors, testing_labels,Vs,cv):
    logging.info("Classifier testing")
    test_vectors = get_vectors(d2v, len(testing_vectors), Vs, 'Test')
    scores=cross_val_score(classifier, test_vectors, testing_labels, cv=cv)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

def build_PVDBOW(xt1,yt1,xv1,yv1, w1,e1,cv,vs):

    #Separando treinamento, teste, classe
    #x_train, x_test, y_train, y_test, all_data = read_dataset(input_trains,input_test)
    print("# de tweets: ",len(xt1)+len(xv1))
    print("# de tweets treino: ",len(xt1))
    print("# de tweets test: ", len(xv1))
    print("% de tweets test: ", len(xv1)*100/(len(xt1)+len(xv1)))

    xt1 = label_sentences(xt1, 'Train')
    xv1 = label_sentences(xv1, 'Test')

    df2=xt1+xv1

    doc2vec = train_doc2vec(df2,w1,e1,vs)

    #usando a representação vetorial para treinar classificador
    classifier = train_classifier(doc2vec, xt1, yt1,vs)

    #testando classificador gerado
    test_classifier(doc2vec, classifier, xv1, yv1,vs,cv)

