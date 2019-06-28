#!/home/master/anaconda2/envs/base3.7/bin python3
from tweepy import OAuthHandler
from tweepy import API
from tweepy import Cursor
import numpy as np
import pandas as pd
import re
import joblib
import sys


from model.hate_recog import our_model

import KEYS_TOKENS


class twitterClient():
    def __init__(self, twitter_user=None):
        self.auth = twitterAuthentication().authenticate_twitter_app()
        self.twitter_client = API(self.auth)
        self.twitter_user = twitter_user

    def get_twitter_client_api(self):
        return self.twitter_client

    def get_user_timeline_tweets(self, num_tweets):
        tweets = []
        for tweet in Cursor(self.twitter_client.user_timeline, id=self.twitter_user).items(num_tweets):
            tweets.append(tweet)
        return tweets

    def get_friend_list(self, num_friends):
        friend_list = []
        for friend in Cursor(self.twitter_client.friends, id=self.twitter_user).items(num_friends):
            friend_list.append(friend)
        return friend_list


class twitterAuthentication():
    """docstring for twitterAuthentication"""

    def authenticate_twitter_app(self):
        auth = OAuthHandler(KEYS_TOKENS.CONSUMER_KEY,
                            KEYS_TOKENS.CONSUMER_SECRET)
        auth.set_access_token(KEYS_TOKENS.ACCESS_TOKEN,
                              KEYS_TOKENS.ACCESS_TOKEN_SECRET)
        return auth


class tweetAnalyzer():
    """docstring for tweetAnalyzer"""

    def clean_tweet(self, tweet):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

    def tweets_to_data_frame(self, tweets):
        df = pd.DataFrame(
            data=[tweet.created_at for tweet in tweets], columns=['date'])
        df['tweet'] = np.array(
            [tweetAnalyzer().clean_tweet(tweet.full_text) for tweet in tweets])
        df['dirty_tweet'] = np.array([tweet.full_text for tweet in tweets])
        #df['retweets'] = np.array([tweet.retweet_count for tweet in tweets])
        #df['likes'] = np.array([tweet.favorite_count for tweet in tweets])
        #df['source'] = np.array([tweet.source for tweet in tweets])
        #df['lang'] = np.array([tweet.lang for tweet in tweets])
        return df


if __name__ == '__main__':
    twitter_client = twitterClient()
    api = twitter_client.get_twitter_client_api()
    tweet_analyzer = tweetAnalyzer()
    tweets = api.user_timeline(screen_name=sys.argv[1], count=200, tweet_mode="extended")
    df = tweet_analyzer.tweets_to_data_frame(tweets)
    t_df = df['dirty_tweet']
    df.to_csv('userTweets.csv')
    np.savetxt('u_Tweets.txt',t_df.values,fmt='%s')
    print('Done')  
    hey=our_model()
    hey.get_csv(df)
    print('done')

    # print(df.head(10))
    sys.stdout.flush()

    

    # tw_model=joblib.load('Model')
    # vect = joblib.load('vect')

    # # vect = TfidfVectorizer(ngram_range = (1,4)).fit(vect)
    # vect_transformed_test = vect.transform(df['tweet'])

    # a = tw_model.predict(vect_transformed_test)
    # print(a)
