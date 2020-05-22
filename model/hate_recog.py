import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
import os
import joblib
import nltk.corpus
from nltk.stem import WordNetLemmatizer
from nltk import tokenize
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
import re



class our_model():

    
    def __init__(self):
        
        
        model = os.path.join(os.getcwd(),'model/model')
        vector = os.path.join(os.getcwd(),'model/vector')
        self.estimator = joblib.load(model)
        self.preprocessor = joblib.load(vector)
        self.sw = set(stopwords.words('english'))    # all english stopwords as
        self.hate =int(0)
        self.not_hate=int(0)
        
        
    def get_input(self,text):
        """
        TAKES INPUT IN STRING AND PREDICTS WHETHER IT IS HATE SPEECH OR NOT
        
        """
        
        vector =self.preprocessor.transform([text])
        proba = self.estimator.predict(vector)
        
    
        if proba==0:
            print('not_hate')          
        if proba==1:
            print('hate')
            
        

    
        
    def get_csv(self,file):
        """
        TAKES INPUT AS CSV FILE AND PREDICTS AND PLOT 
        
        """
        # for cleaning the file
        
        file = self.cleaning(file)  
<<<<<<< HEAD
        
        
=======
>>>>>>> a231eef3f2c249b2aba3125f1a85c36eaebe6ddf
        vector =self.preprocessor.transform(file['tweet'])
        proba = self.estimator.predict(vector)
        self.no_of_items=int(file.shape[0])
       
        for val in proba:
            if val==0:
                self.not_hate+=1
                #print('not_hate')
                
            if val==1:
                self.hate+=1
                #print('hate')
                
        #plot function 
        self.plot(self.hate,self.not_hate)
            
        
    
    
    # remove stop words
    def filter_words(self,word_list):
        useful_words = [ w for w in word_list if w not in self.sw ]
        return(useful_words)
     
    
    
    
    # to clean the tweets 
    def cleaning(self,file):
        
        """
        removes url hashtags @ symbols whitespaces
        
        """
        
        
        file['tweet'] = [''.join(     [WordNetLemmatizer().lemmatize(   re.sub('[^A-Za-z]',' ',text  )     ) for text in lis     ]      )      for lis in file['tweet']       ]
        a=[]
        for text in file['tweet']:
            word_list = word_tokenize(text)
            
            text=self.filter_words(word_list)
        
            a.append(text)
            
            
        train_text = []
        for i in a:
            sent=''
            for  j in i:
                sent += str(j) + ' '
            train_text.append(sent)
        file['tweet'] = train_text
        
        return file 
    
    
    
    
    #percentage calculation function
    def percent(self,value,total):
        return float(value)/float(total)
    
    
   
    
    
    # pie chart plotting function
    def plot(self,x,y):
         
        h_perc=self.percent(x,self.no_of_items)
        nh_perc=self.percent(y,self.no_of_items)
        print('\n\nPERCENTAGES\n\n')
        fig=plt.figure()
        sizes = [h_perc,nh_perc]
        colors = ['red','green']
        plt.pie(sizes , labels=['hate','not hate'] ,colors = colors ,startangle = 90 ,explode=(0.1,0) ,  shadow=True,autopct='%1.1f%%')
        plt.legend()
        plt.axis('equal')
        
        plt.tight_layout()
        fig.savefig('./public/plot.jpeg')
       
        

        
