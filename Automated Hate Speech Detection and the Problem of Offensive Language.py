
# coding: utf-8

# # Replication for results in Davidson et al. 2017. "Automated Hate Speech Detection and the Problem of Offensive Language"

# In[40]:


import pandas as pd
import numpy as np
import pickle
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem.porter import *
import string
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS
from textstat.textstat import *
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report,accuracy_score
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import seaborn
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Loading the data

# In[41]:


df = pd.read_csv(r"C:\Users\amanr\Desktop\labeled_data.csv")


# In[42]:


df


# In[43]:


df.describe()


# In[44]:


df.columns


# ### Columns key:
# count = number of CrowdFlower users who coded each tweet (min is 3, sometimes more users coded a tweet when judgments were determined to be unreliable by CF).
# 
# 
# hate_speech = number of CF users who judged the tweet to be hate speech.
# 
# 
# offensive_language = number of CF users who judged the tweet to be offensive.
# 
# 
# neither = number of CF users who judged the tweet to be neither offensive nor non-offensive.
# 
# 
# class = class label for majority of CF users.
# 
#     0 - hate speech
#     1 - offensive  language
#     2 - neither
# 
# tweet = raw tweet text
# 

# In[67]:


#df['class'].hist()


# This histogram shows the imbalanced nature of the task - most tweets containing "hate" words as defined by Hatebase were 
# only considered to be offensive by the CF coders. More tweets were considered to be neither hate speech nor offensive language than were considered hate speech.

# In[47]:


tweets=df.tweet


# ## Feature generation

# In[48]:


stopwords=stopwords = nltk.corpus.stopwords.words("english")

other_exclusions = ["#ff", "ff", "rt"]
stopwords.extend(other_exclusions)

stemmer = PorterStemmer()


def preprocess(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, '', parsed_text)
    parsed_text = re.sub(mention_regex, '', parsed_text)
    return parsed_text

def tokenize(tweet):
    """Removes punctuation & excess whitespace, sets to lowercase,
    and stems tweets. Returns a list of stemmed tokens."""
    tweet = " ".join(re.split("[^a-zA-Z]*", tweet.lower())).strip()
    tokens = [stemmer.stem(t) for t in tweet.split()]
    return tokens

def basic_tokenize(tweet):
    """Same as tokenize but without the stemming"""
    tweet = " ".join(re.split("[^a-zA-Z.,!?]*", tweet.lower())).strip()
    return tweet.split()

vectorizer = TfidfVectorizer(
    tokenizer=tokenize,
    preprocessor=preprocess,
    ngram_range=(1, 3),
    stop_words=stopwords,
    use_idf=True,
    smooth_idf=False,
    norm=None,
    decode_error='replace',
    max_features=10000,
    min_df=5,
    max_df=0.75
    )


# In[109]:


#Construct tfidf matrix and get relevant scores
tf = vectorizer.fit(tweets)
tfidf = vectorizer.transform(tweets).toarray()

vocab = {v:i for i, v in enumerate(vectorizer.get_feature_names())}
#print(vocab)
idf_vals = vectorizer.idf_

idf_dict = {i:idf_vals[i] for i in vocab.values()} #keys are indices; values are IDF scores


#print(idf_vals)
#print(idf_dict)


# In[110]:


#Get POS tags for tweets and save as a string
tweet_tags = []
for t in tweets:
    tokens = basic_tokenize(preprocess(t))
    tags = nltk.pos_tag(tokens)
    tag_list = [x[1] for x in tags]
    tag_str = " ".join(tag_list)
    tweet_tags.append(tag_str)
    #print(tweet_tags)


# In[111]:


#We can use the TFIDF vectorizer to get a token matrix for the POS tags
pos_vectorizer = TfidfVectorizer(
    tokenizer=None,
    lowercase=False,
    preprocessor=None,
    ngram_range=(1, 3),
    stop_words=None,
    use_idf=False,
    smooth_idf=False,
    norm=None,
    decode_error='replace',
    max_features=5000,
    min_df=5,
    max_df=0.75,
    )

#print(pos_vectorizer)


# In[112]:


#Construct POS TF matrix and get vocab dict
po = pos_vectorizer.fit(tweets_tag)_transform(pd.Series(tweet_tags)).toarray()
pos_vocab = {v:i for i, v in enumerate(pos_vectorizer.get_feature_names())}


# In[113]:


#print(pos_vocab)


# In[114]:


#Now get other features
sentiment_analyzer = VS()

def count_twitter_objs(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE
    4) hashtags with HASHTAGHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned.
    
    Returns counts of urls, mentions, and hashtags.
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    hashtag_regex = '#[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, 'URLHERE', parsed_text)
    parsed_text = re.sub(mention_regex, 'MENTIONHERE', parsed_text)
    parsed_text = re.sub(hashtag_regex, 'HASHTAGHERE', parsed_text)
    return(parsed_text.count('URLHERE'),parsed_text.count('MENTIONHERE'),parsed_text.count('HASHTAGHERE'))

def other_features(tweet):
    """This function takes a string and returns a list of features.
    These include Sentiment scores, Text and Readability scores,
    as well as Twitter specific features"""
    sentiment = sentiment_analyzer.polarity_scores(tweet)
    
    words = preprocess(tweet) #Get text only
    
    syllables = textstat.syllable_count(words)
    num_chars = sum(len(w) for w in words)
    num_chars_total = len(tweet)
    num_terms = len(tweet.split())
    num_words = len(words.split())
    avg_syl = round(float((syllables+0.001))/float(num_words+0.001),4)
    num_unique_terms = len(set(words.split()))
    
    ###Modified FK grade, where avg words per sentence is just num words/1
    FKRA = round(float(0.39 * float(num_words)/1.0) + float(11.8 * avg_syl) - 15.59,1)
    ##Modified FRE score, where sentence fixed to 1
    FRE = round(206.835 - 1.015*(float(num_words)/1.0) - (84.6*float(avg_syl)),2)
    
    twitter_objs = count_twitter_objs(tweet)
    retweet = 0
    if "rt" in words:
        retweet = 1
    features = [FKRA, FRE,syllables, avg_syl, num_chars, num_chars_total, num_terms, num_words,
                num_unique_terms, sentiment['neg'], sentiment['pos'], sentiment['neu'], sentiment['compound'],
                twitter_objs[2], twitter_objs[1],
                twitter_objs[0], retweet]
    #features = pandas.DataFrame(features)
    return features

def get_feature_array(tweets):
    feats=[]
    for t in tweets:
        feats.append(other_features(t))
    return np.array(feats)


# In[115]:


other_features_names = ["FKRA", "FRE","num_syllables", "avg_syl_per_word", "num_chars", "num_chars_total",                         "num_terms", "num_words", "num_unique_words", "vader neg","vader pos","vader neu",                         "vader compound", "num_hashtags", "num_mentions", "num_urls", "is_retweet"]


# In[116]:


feats = get_feature_array(tweets)


# In[117]:


#Now join them all up
M = np.concatenate([tfidf,pos,feats],axis=1)


# In[118]:


print(M.shape)
print(M)


# In[119]:


#Finally get a list of variable names
variables = ['']*len(vocab)
for k,v in vocab.items():
    variables[v] = k

pos_variables = ['']*len(pos_vocab)
for k,v in pos_vocab.items():
    pos_variables[v] = k

feature_names = variables+pos_variables+other_features_names


# # Running the model
# 
# The best model was selected using a GridSearch with 5-fold CV.

# In[120]:


X = pd.DataFrame(M)
y = df['class'].astype(int)


# In[121]:


select = SelectFromModel(LogisticRegression(class_weight='balanced',penalty="l1",C=0.01))
X_ = select.fit_transform(X,y)


# In[122]:


#model = LinearSVC(class_weight='balanced',C=0.01, penalty='l2', loss='squared_hinge',multi_class='ovr').fit(X_, y)


# In[123]:


model = LogisticRegression(class_weight='balanced',penalty='l2',C=0.01).fit(X_,y)


# In[124]:


y_preds = model.predict(X_)


# ## Evaluating the results

# In[128]:


print((X_.shape))


# In[129]:


report = classification_report( y, y_preds )
accu =accuracy_score(y,y_preds)


# In[130]:


print(report,accu)


# In[74]:


# new_text = pd.read_csv(r'C:/users/amanr/desktop/PROJECT/Aman/test_tweets_anuFYb8.csv')
# new_text = new_text['tweet']
# t=[]
# for i in new_text:
#     t.append(preprocess(i))
# t


# In[94]:


# plt.rc('pdf', fonttype=42)
# plt.rcParams['ps.useafm'] = True
# plt.rcParams['pdf.use14corefonts'] = True
# plt.rcParams['text.usetex'] = True
# plt.rcParams['font.serif'] = 'Times'
# plt.rcParams['font.family'] = 'serif'
# from sklearn.metrics import confusion_matrix

# confusion_matrix = confusion_matrix(y,y_preds)
# matrix_proportions = np.zeros((3,3))
# for i in range(0,3):
#     matrix_proportions[i,:] = confusion_matrix[i,:]/float(confusion_matrix[i,:].sum())
# names=['Hate','Offensive','Neither']
# confusion_df = pd.DataFrame(matrix_proportions, index=names,columns=names)
# plt.figure(figsize=(5,5))
# seaborn.heatmap(confusion_df,annot=True,annot_kws={"size": 12},cmap='gist_gray_r',cbar=False, square=True,fmt='.2f')
# plt.ylabel(r'\textbf{True categories}',fontsize=14)
# plt.xlabel(r'\textbf{Predicted categories}',fontsize=14)
# plt.tick_params(labelsize=12)

# #Uncomment line below if you want to save the output
# #plt.savefig('confusion.pdf')


# In[95]:


#True distribution
#y.hist()


# In[96]:


#pd.Series(y_preds).hist()

