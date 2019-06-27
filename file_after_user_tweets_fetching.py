from model.hate_recog import our_model
import os
import pandas as pd
import numpy as np



path=str(os.getcwd())+'\\userTweets.csv'
file=pd.read_csv(path)
file=pd.DataFrame(file)
print(type(file))
print('loading model')
hey=our_model()
print('Analyzing')
hey.get_csv(file)

#pd.read_csv(os.getcwd())