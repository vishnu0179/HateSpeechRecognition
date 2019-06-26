import tweepy
import os
import jsonpickle
import pandas
import sys
import json
from fetchtweets import tweetAnalyzer

api_key = 'IgvYuzdJzJHWHSGiubz98UGoe'
api_secret = 'XtFcOGOOVO4rtFHgByIPUYqLt0W9VlkwqfDTAjZovkyvVFqBfl'

auth = tweepy.AppAuthHandler(api_key, api_secret)

api = tweepy.API(auth,wait_on_rate_limit = True,wait_on_rate_limit_notify = True)

if(not api):
    print('Cant Authenticate')
    sys.exit(-1)

choice = 1
if(choice == 1):
    searchQuery = sys.argv[1] # this is what we're searching for
    maxTweets = int(sys.argv[2]) 
    tweetsPerQry = 90
    tweet_count = 0
    
    sinceId = None
    pTweet=[]
    maxId = -1000000    
    
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
        for tweet in new_tweets:
            #t = jsonpickle.encode(tweet._json, unpicklable = False) +'\n'                
            tweet = jsonpickle.encode(tweet._json, unpicklable = False)
            t = json.loads(tweet)
            if('retweeted_status' in t):
                pTweet.append(tweetAnalyzer().clean_tweet(t.get('retweeted_status').get('full_text')))
            else:
                pTweet.append(tweetAnalyzer().clean_tweet(t.get('full_text')))
        tweet_count += len(new_tweets)
        maxId = new_tweets[-1].id    
        #print('Downloaded {0} tweets'.format(tweet_count))
    
    df= pandas.DataFrame(data=pTweet, columns=['tweet'])
    #df=tweetAnalyzer().tweets_to_data_frame(tweets=new_tweets)
    df.to_csv('hashtagt_df.csv')
    df.head(10)

    print('Tweets Fetched')
    sys.stdout.flush()


