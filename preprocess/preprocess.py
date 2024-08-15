import pandas as pd
from pandas import read_parquet
import re
import string
import codecs
from nltk.stem.porter import PorterStemmer
RAW_TRAIN_FILE = '../data/btc_tweets_train.parquet.gzip'
RAW_TEST_FILE = '../data/btc_tweets_test.parquet.gzip'
EMOJI_LEXICON_FILE = 'emoji_utf8_lexicon.txt'
PROCESSED_FOLDER = '../data/processed/'


def expand_contractions(text):
    def expand(match):
        contraction = match.group(0)
        
        # Split the contraction into parts
        parts = re.split(r"'", contraction)
        
        if len(parts) != 2:
            return contraction

        before, after = parts

        # Handle common verb contractions
        if after.lower() in ['s', 're', 've', 'll', 'd', 'm', 't']:
            if after.lower() == 's':
                if before.lower() in ['it', 'that', 'what', 'here']:
                    return f"{before} is"
                else:
                    return f"{before} is"  # This could also be "has" in some cases
            elif after.lower() == 're':
                return f"{before} are"
            elif after.lower() == 've':
                return f"{before} have"
            elif after.lower() == 'll':
                return f"{before} will"
            elif after.lower() == 'd':
                return f"{before} would"  # This could also be "had" in some cases
            elif after.lower() == 'm':
                return f"{before} am"
            elif after.lower() == 't':
                if before.lower() == 'won':
                    return "will not"
                elif before.lower() in ['can', 'don', 'wouldn', 'shouldn', 'couldn', 'didn']:
                    return f"{before}ot"
                else:
                    return f"{before} not"
        
        # If we can't expand it, return the original
        return contraction

    # Apply the expansion to the text
    pattern = r'\b\w+\'\w+\b'  # Matches words with apostrophes
    return re.sub(pattern, expand, text)


def preprocess_word(word):
    # Remove punctuation
    word = word.translate(str.maketrans('', '', string.punctuation))
    # Convert more than 2 letter repetitions to 2 letter
    # funnnnny --> funny
    word = re.sub(r'(.)\1+', r'\1\1', word)
    return word

"""
def _is_valid_word(word):
    # Check if word begins with an alphabet
    return (re.search(r'^[a-zA-Z][a-z0-9A-Z\._]*$', word) is not None)
"""

def handle_emojis(tweet):
    with codecs.open(EMOJI_LEXICON_FILE, encoding='utf-8') as f:
        emoji_full_filepath = f.read()
    emojis = _make_emoji_dict(emoji_full_filepath)
    text_no_emoji = ""
    prev_space = True
    for chr in tweet:
        if chr in emojis:
            # get the textual description
            description = emojis[chr]
            if not prev_space:
                text_no_emoji += ' '
            text_no_emoji += description
            prev_space = False
        else:
            text_no_emoji += chr
            prev_space = chr == ' '
    tweet = text_no_emoji.strip()
    return tweet

def _make_emoji_dict(emoji_full_filepath):
    """
    Convert emoji lexicon file to a dictionary
    """
    emoji_dict = {}
    for line in emoji_full_filepath.rstrip('\n').split('\n'):
        (emoji, description) = line.strip().split('\t')[0:2]
        emoji_dict[emoji] = description
    return emoji_dict

def preprocess_tweet(tweet):
    processed_tweet = []
    # Convert to lower case
    tweet = tweet.lower()
    # expand contractions
    tweet = expand_contractions(tweet)
    # Replaces URLs with the word URL
    tweet = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' URL ', tweet)
    # Replace @handle with the word USER_MENTION
    tweet = re.sub(r'@[\S]+', 'USER_MENTION', tweet)
    # Replaces #hashtag with hashtag
    tweet = re.sub(r'#(\S+)', r' \1 ', tweet)
    # Remove RT (retweet)
    tweet = re.sub(r'\brt\b', '', tweet)
    # Replace 2+ dots with space
    tweet = re.sub(r'\.{2,}', ' ', tweet)
    # Strip space, " and ' from tweet
    tweet = tweet.strip(' "\'')
    # Replace emojis with their description
    tweet = handle_emojis(tweet)
    # Replace multiple spaces with a single space
    tweet = re.sub(r'\s+', ' ', tweet)
    words = tweet.split()

    for word in words:
        word = preprocess_word(word)
        processed_tweet.append(word)
        """
        if _is_valid_word(word):
            if use_stemmer:
                word = str(porter_stemmer.stem(word))
            processed_tweet.append(word)
        """

    return ' '.join(processed_tweet)

#def preprocess_hashtag(hashtag_list):
#    return list(set(hashtag_list))

def preprocess_csv(raw_data_path, processed_file_name, processed_file_path):
    raw_data = pd.read_parquet(raw_data_path)
    # Apply preprocess_tweet function to the 'content' column
    raw_data['content'] = raw_data['content'].apply(preprocess_tweet)
    raw_data.to_csv(processed_file_path, index=False)
    print('\nSaved processed tweets to: %s' % processed_file_name)
    return processed_file_name