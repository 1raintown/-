import nltk
from gensim.models import word2vec
import numpy as np
import string
from nltk import word_tokenize
import itertools
from gensim.models import KeyedVectors
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import pandas as pd
import random
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer



data=pd.read_csv('./data.csv')
multi3_1=data[data['Target Name']=='comp.graphics'].sample(200)
multi3_2=data[data['Target Name']=='rec.autos'].sample(200)
multi3_3=data[data['Target Name']=='talk.politics.guns'].sample(200)

multi3=multi3_1.append(multi3_2).append(multi3_3)


multi5_1=data[data['Target Name']=='comp.graphics'].sample(200)
multi5_2=data[data['Target Name']=='comp.windows.x'].sample(200)
multi5_3=data[data['Target Name']=='rec.sport.hockey'].sample(200)
multi5_4=data[data['Target Name']=='rec.autos'].sample(200)
multi5_5=data[data['Target Name']=='talk.politics.guns'].sample(200)

multi5=multi5_1.append(multi5_2).append(multi5_3).append(multi5_4).append(multi5_5)


feature_size=100
window_context=5
min_word_count=1
sample=1e-3

#去标点
tokenized_corpus_3=[]
clean_article=multi3['Clean Article'].tolist()
for i in clean_article:
    remove = str.maketrans('','',string.punctuation)
    i = i.translate(remove)
    a=word_tokenize(i)
    text=' '.join(a)
    tokenized_corpus_3.append(text)
multi3['Clean Article remove punc']= tokenized_corpus_3

tokenized_corpus_5=[]
clean_article=multi5['Clean Article'].tolist()
for i in clean_article:
    remove = str.maketrans('','',string.punctuation)
    i = i.translate(remove)
    a=word_tokenize(i)
    text=' '.join(a)
    tokenized_corpus_5.append(text)
multi5['Clean Article remove punc']= tokenized_corpus_5

#计算权重


def tfidf_wtd_avg_word_vectors(words,tfidf_vector,tfidf_vocabulary,model,num_features):
    word_tfidfs=[tfidf_vector[0,tfidf_vocabulary.get(word)] if tfidf_vocabulary.get(word) else 0 for word in words]
    word_tfidf_map={word:tfidf_val for word,tfidf_val in zip(words,word_tfidfs)}
    feature_vector=np.zeros((num_features,),dtype='float64')
    vocabulary=set(model.wv.index_to_key)
    wts=0
    for word in words:
        if word in vocabulary:
            word_vector=model.wv[word]
            weighted_word_vector=word_tfidf_map[word]*word_vector
            wts=wts+word_tfidf_map[word]
            feature_vector=np.add(feature_vector,weighted_word_vector)
    if wts:
        feature_vector=np.divide(feature_vector,wts)
    return feature_vector

def tfidf_weighted_averaged_word_vectorizer(corpus,tfidf_vectors,tfidf_vocabulary,model,num_features):
    docs_tfidfs=[(doc,doc_tfidf) for doc,doc_tfidf in zip(corpus,tfidf_vectors)]
    features=[tfidf_wtd_avg_word_vectors(tokenized_sentence,tfidf,tfidf_vocabulary,model,num_features) for tokenized_sentence,tfidf in docs_tfidfs]
    return np.array(features)

#multi3
train_corpus,test_corpus,train_label_names,test_label_names = train_test_split(np.array(multi3['Clean Article remove punc'])
                                                                        ,np.array(multi3['Target Name'])
                                                                        ,test_size=0.33,random_state=42)

tv = TfidfVectorizer(min_df=0.0, max_df=1.0, norm='l2', use_idf=True, smooth_idf=True)
tv_train_features=tv.fit_transform(train_corpus)
tv_test_features=tv.fit_transform(test_corpus)


train_words=[word_tokenize(i) for i in train_corpus ]
test_words=[word_tokenize(i) for i in test_corpus ]

w2v_model_3=word2vec.Word2Vec(train_words+test_words,
                              vector_size=feature_size,
                              window=window_context,
                           min_count=min_word_count,
                              sample=sample,sg=1)

tv_voca=tv.vocabulary_
train_features=tfidf_weighted_averaged_word_vectorizer(train_corpus,tv_train_features.toarray(),tv_voca,w2v_model_3,100)
test_features=tfidf_weighted_averaged_word_vectorizer(test_corpus,tv_test_features.toarray(),tv_voca,w2v_model_3,100)