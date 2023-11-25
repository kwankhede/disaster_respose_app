# text_processing.py

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download("punkt")
nltk.download("stopwords")


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    tokens = [WordNetLemmatizer().lemmatize(word).strip() for word in tokens]
    return tokens
