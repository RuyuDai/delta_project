from pandas import read_parquet
import pandas as pd
import re
from collections import Counter, defaultdict
from ast import literal_eval
RAW_TRAIN_FILE = '../data/raw/btc_tweets_train.parquet.gzip'
RAW_TEST_FILE = '../data/raw/btc_tweets_test.parquet.gzip'
EMOJI_LEXICON_FILE = 'emoji_utf8_lexicon.txt'
PROCESSED_FOLDER = '../data/processed/'


# data skewness  Positive / Negative
# 1220 / (1500-1220)

# user behavior

## duplicate content / user


## posting freq


# Emoji / tweet , most commonly used emoji

# URL / tweet

# word / tweet

# Function to check if a tweet contains contractions
def has_contraction(text):
    pattern = r'\b\w+\'\w+\b'
    return bool(re.search(pattern, text))

def _count_unique_hashtags(hashtags_list):
    return len(set(tag.lower() for tag in hashtags_list))

def _extract_urls(text):
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.findall(url_pattern, text)

def calculate_features_for_spam(df):

    user_features = defaultdict(lambda: {'tweet_count': 0, 
                                         'total_words': 0, 
                                         'unique_words': set(), 
                                         'hashtags': set(), 
                                         'url_count': 0,
                                         'total_neutral_score': 0})

    for _, row in df.iterrows():
        username = row['username']
        content = row['content']
        hashtags = literal_eval(row['hashtags']) if isinstance(row['hashtags'], str) else row['hashtags']
        neutral_score = row['vaderNeutral']

        # Word count
        words = content.split()
        word_count = len(words)

        # Update user features
        user_features[username]['tweet_count'] += 1
        user_features[username]['total_words'] += word_count
        user_features[username]['unique_words'].update(words)
        user_features[username]['hashtags'].update(tag.lower() for tag in hashtags)
        user_features[username]['url_count'] += len(_extract_urls(content))
        user_features[username]['total_neutral_score'] += neutral_score

    # Calculate aggregated features
    result = []
    for username, features in user_features.items():
        result.append({
            'username': username,
            'tweet_count': features['tweet_count'],
            'avg_word_count': round(features['total_words'] / features['tweet_count'],1),
            'unique_word_ratio': round(len(features['unique_words']) / features['total_words'] if features['total_words'] > 0 else 0,1),
            'unique_hashtag_count': round(len(features['hashtags']),1),
            'urls_per_tweet': round(features['url_count'] / features['tweet_count'],1),
            'avg_neutral_score': round(features['total_neutral_score'] / features['tweet_count'],1) 
        })

    return pd.DataFrame(result)