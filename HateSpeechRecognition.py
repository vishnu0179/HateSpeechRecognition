import tweepy
import os
import jsonpickle
import sys
from fetch_tweets import tweetAnalyzer

api_key = 'IgvYuzdJzJHWHSGiubz98UGoe'
api_secret = 'XtFcOGOOVO4rtFHgByIPUYqLt0W9VlkwqfDTAjZovkyvVFqBfl'

auth = tweepy.AppAuthHandler(api_key, api_secret)

api = tweepy.API(auth,wait_on_rate_limit = True,wait_on_rate_limit_notify = True)

if(not api):
    print('Cant Authenticate')
    sys.exit(-1)

choice = 1
if(choice == 1):
    searchQuery = 'trump'  # this is what we're searching for
    maxTweets =100 
    tweetsPerQry = 50
    fName = 'tweets1.txt'
    tweet_count = 0
    
    sinceId = None
    maxId = -1000000    
    print('Fetching {0} tweets'.format(maxTweets))
    
    with open(fName,'w') as f:
        while tweet_count < maxTweets:
            if (maxId<=0):
                if (not sinceId):
                    new_tweets = api.search(q=searchQuery,count = tweetsPerQry, tweet_mode = "extended")
                else:
                    new_tweets = api.search(q = searchQuery, count = tweetsPerQry, tweet_mode = "extended", since_id = sinceId)
            else:
                if(not sinceId):
                    new_tweets = api.search(q=searchQuery,count = tweetsPerQry, tweet_mode = "extended", max_id=str(maxId - 1))
                else:
                    new_tweets = api.search(q=searchQuery,count = tweetsPerQry, tweet_mode = "extended", max_id = str(maxId -1), since_id = sinceId)
            if not new_tweets:
                print("No more Tweets found")
                break
            tweet_count += len(new_tweets)
            maxId = new_tweets[-1].id    

df=tweetAnalyzer().tweets_to_data_frame(tweets=new_tweets)
df.to_csv('hashtagt_df.csv')
print(df.head())
