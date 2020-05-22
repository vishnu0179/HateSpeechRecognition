#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import os




print(os.listdir('./input'))


train = pd.read_csv('./input/train_E6oV3lV.csv');
test = pd.read_csv('./input/test_tweets_anuFYb8.csv')



train.head()
test.head()


train['label'] = train['label'].astype('category')




train.info()



from nltk.stem import WordNetLemmatizer
from nltk import tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
nltk.download('wordnet')




train['text_lem'] = [''.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]',' ',text)) for text in lis]) for lis in train['tweet']]
test['text_lem'] = [''.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]',' ',text)) for text in lis]) for lis in test['tweet']]


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(train['text_lem'],train['label'])



vect = TfidfVectorizer(ngram_range = (1,4)).fit(X_train)




vect_transformed_X_train = vect.transform(X_train)
vect_transformed_X_test = vect.transform(X_test)




from sklearn.linear_model import LogisticRegression

modelLR = LogisticRegression(C=100).fit(vect_transformed_X_train,y_train)




from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

predictionsLR = modelLR.predict(vect_transformed_X_test)
sum(predictionsLR==1),len(y_test),f1_score(y_test,predictionsLR)
accuracy_score(y_test, predictionsLR)






