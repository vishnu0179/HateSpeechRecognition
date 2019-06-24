import tweepy
import os
import jsonpickle
import sys

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
    maxTweets = sys.argv[2] 
    tweetsPerQry = 50
    fName = 'tweets1.txt'
    tweet_count = 0
    
    sinceId = None
    maxId = -1000000    
    
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

            for tweet in new_tweets:
                f.write(jsonpickle.encode(tweet._json, unpicklable = False) +'\n')
                tweet = jsonpickle.encode(tweet._json, unpicklable = False)
                t = json.loads(tweet)
                if('retweeted_status' in t):
                    f.write(t.get('retweeted_status').get('full_text'))
                    print(t.get('retweeted_status').get('full_text'))
                else:
                    f.write(t.get('full_text'))
                    print(t.get('full_text'))
            tweet_count += len(new_tweets)
            maxId = new_tweets[-1].id    
            print('Downloaded {0} tweets'.format(tweet_count))

