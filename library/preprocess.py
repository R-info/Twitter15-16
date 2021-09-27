import gensim
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer, PorterStemmer
from nltk.stem.porter import *
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import numpy as np
import pandas as pd
import sys
from typing import List

np.random.seed(117)

factory = StemmerFactory()
indo_stemmer = factory.create_stemmer()
indo_stopwords =  set(stopwords.words('indonesian'))

def remove_urls(text):
    words = []
    try:
        for word in text.split(' '):
            if "http://" in word or "https://" in word:
                continue
            words.append(word)
        return " ".join(words)
    except:
        print("Error at Removing URLS")
        print(text)
#         sys.exit(1)
        return ""

def remove_stopwords_id(tokens):
    for token in tokens:
        if token in indo_stopwords:
            tokens.remove(token)
    return tokens

def get_stemmed_indo(text):
    return indo_stemmer.stem(text)

def preprocess_id(data, filepath: str = None, is_split = False):
#     print("Checking if preprocess result exists...")
    print("Preprocessing Data of Bahasa Indonesia...")
    results = []
    for i, text in enumerate(data):
#         print(f"Process no {i} - {text}")
        text = remove_urls(text)
        tokens = word_tokenize(text)
        tokens = remove_stopwords_id(tokens)
        result = " ".join(tokens)
        result = get_stemmed_indo(result)
        if is_split:
            result = result.split(" ")
        results.append(result)

        if (i + 1) % 100 == 0:
            print(f"Done with {i+1} data")
    print("Preprocess Done")
    if filepath:
        print("Saving Results...")
        with open(filepath, "w") as f:
            for res in results:
                f.write(res + '\n')

    return results

class TextPreprocess:
    texts: List[str]
    postags: List[str] = []
    is_tokenized: bool = False

    def __init__(self, texts):
        self.texts = texts

    def get_texts(self):
        return self.texts

    def texts_and_postags(self):
        if not self.postags:
            print("Please generate postags dataset first!")
            raise

        if self.is_tokenized:
            print("Please detokenize texts dataset first!")
            raise

        results = []
        for i, text in enumerate(self.texts):
            results.append(text + self.postags[i])
        return results

    def tokenize(self, is_tweet: bool = False):
        if is_tweet:
            text_tokenizer = TweetTokenizer().tokenize
        else:
            text_tokenizer = word_tokenize

        print("Tokenizing Texts")
        for i, text in enumerate(self.texts):
            self.texts[i] = text_tokenizer(text)
        self.is_tokenized = True

        return self

    def detokenize(self):
        print("Detokenizing Texts")
        for i, text in enumerate(self.texts):
            self.texts[i] = " ".join(text)
        self.is_tokenized = False

        return self

    def generate_postags(self):
        if not self.is_tokenized:
            print("Please tokenized the texts dataset first!")
            raise

        self.postags = []
        for tokens in self.texts:
            self.postags.append(" ".join([x[1] for x in nltk.pos_tag(tokens)]))

        return self

    def remove_punctuation(self):
        if not self.is_tokenized:
            print("Please tokenized the texts dataset first!")
            raise

        for i, text in enumerate(self.texts):
            self.texts[i] = [token for token in text if token.isalnum()]
        return self

    def change_urls(self, url_str: str = "$url$"):
        if not self.is_tokenized:
            print("Please tokenized the texts dataset first!")
            raise

        print(f"Change URL to '{url_str}'")
        for i, text in enumerate(self.texts):
            tokens = []
            try:
                for token in text:
                    if "http://" in token or "https://" in token:
                        tokens.append(url_str)
                    else:
                        tokens.append(token)

                self.texts[i] = tokens
            except:
                print(f"Error Removing URLS at index {i}")
                print(text)

        return self

    def change_mentions(self, user_str: str = "$user$"):
        if not self.is_tokenized:
            print("Please tokenized the texts dataset first!")
            raise

        print(f"Change User Mention to '{user_str}'")
        for i, text in enumerate(self.texts):
            tokens = []
            try:
                for token in text:
                    if "@" in token:
                        tokens.append(user_str)
                    else:
                        tokens.append(token)

                self.texts[i] = tokens
            except:
                print(f"Error Removing URLS at index {i}")
                print(text)

        return self

    def clean_tweets(self,
        rm_urls: bool = True,
        rm_hashtags: bool = True,
        rm_rt: bool = True,
        rm_mentions: bool = True
    ):
        if not self.is_tokenized:
            print("Please tokenized the texts dataset first!")
            raise

        print("Cleaning Tweet Texts")
        for i, text in enumerate(self.texts):
            tokens = []
            try:
                for token in text:
                    if rm_urls:
                        if "http://" in token or "https://" in token:
                            continue

                    if rm_hashtags:
                        if "#" in token:
                            continue

                    if rm_rt:
                        if token in ["rt", "RT"]:
                            continue

                    if rm_mentions:
                        if "@" in token:
                            continue
                    else:
                        token = token.replace("_", "-")
                        token = token.replace("@", "")

                    tokens.append(token)

                self.texts[i] = tokens
            except:
                print(f"Error Removing URLS at index {i}")
                print(text)

        return self

    def remove_stopwords(self, lang: str = "id"):
        print(f"Removing {lang.upper()} Stopwords from Texts")
        if lang == 'id':
            lang_stopwords = set(stopwords.words('indonesian'))
        elif lang == 'en':
            lang_stopwords = set(stopwords.words('english'))
        else:
            print("Language not Supported")
            raise

        if not self.is_tokenized:
            print("Please tokenized the texts dataset first!")
            raise

        for i, text in enumerate(self.texts):
            for token in text:
                if token in lang_stopwords:
                    self.texts[i].remove(token)

        return self

    def stemming(self, lang: str = "id"):
        if lang == "id":
            print(f"Stemming 'id' Texts")
            factory = StemmerFactory()
            stemmer = factory.create_stemmer()
        elif lang == 'en':
            print(f"Stemming 'en' Texts")
            # stemmer = SnowballStemmer('english')
            stemmer = PorterStemmer()
        else:
            print("Language not Supported")
            return

        for i, text in enumerate(self.texts):
            self.texts[i] = stemmer.stem(text)

            if (i+1) % 1000 == 0:
                print(f"Stemmed {i+1} Texts")

        return self

    def save(self, filepath):
        print("Saving Texts...")
        with open(filepath, "w") as f:
            for res in self.texts:
                f.write(res + '\n')
