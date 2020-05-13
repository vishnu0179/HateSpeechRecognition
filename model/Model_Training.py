## HOW WE TRAINED THE MODEL

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import pickle
from nltk.stem import WordNetLemmatizer
from nltk import tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn.model_selection import train_test_split
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
#Importing different models but got best accuracy on LOGISTIC REGRESSION



#LOADING DATA
train = pd.read_csv(r"../input/train_E6oV3lV.csv")
test = pd.read_csv(r"../input/test_tweets_anuFYb8.csv")



#PLOTTING HISTOGRAM
train['label'].hist()
plt.show()


#TURNING LABELS INTO CATEGORIES
train['label'] = train['label'].astype('category')

#printing data's information (total values, null values, etc)
train.info()



#Removing @, special characters and lemmatizing every word
train['text_lem'] = [''.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]',' ',text)) for text in lis]) for lis in train['tweet']]
test['text_lem'] = [''.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]',' ',text)) for text in lis]) for lis in test['tweet']]



#print(train)#['text_lem'])
# train['text_lem']
# print(train.shape)








#TRAIN TEST SPLIT
X_train,X_test,y_train,y_test = train_test_split(train['text_lem'],train['label'])




#AS the model cant understand words so converting sentences into vector
#USING TF-IDF vectorizer (Term Freequency - Inverse Document Frequency)
#another benifit of using TF-IDF vectorizer that is automatically removes wordds which are very common like stopwords,
#so you dont have to remove stopwords it will work fine !!!

#Fiting vector on data
vect = TfidfVectorizer(ngram_range = (1,4)).fit(X_train)


#Transforming train and test data based on that data
vect_transformed_X_train = vect.transform(X_train)
vect_transformed_X_test = vect.transform(X_test)



#Printing shapes
print(vect_transformed_X_train.shape)
print(vect_transformed_X_test.shape)
print(y_train.shape)
print(y_test.shape)



#Using Logistic Regression
modelLR = LogisticRegression(C=100).fit(vect_transformed_X_train,y_train)
predictionsLR = modelLR.predict(vect_transformed_X_test)

#Printing Accuracy
print('Accuracy :',accuracy_score(y_test,predictionsLR))



#printing Predictions
print(predictionsLR)





#TO save the model and vector for future use on other systems



# with open('model2','wb') as f:
#     pickle.dump(modelLR,f)
    

# with open('vector2','wb') as f:
#     pickle.dump(vect,f)
    

