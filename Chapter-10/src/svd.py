import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn import decomposition
from sklearn.feature_extraction.text import TfidfVectorizer

import re
import string

def clean_text(s):
    s = s.split()
    s = " ".join(s)
    s = re.sub(f"[{re.escape(string.punctuation)}]", "", s)
    return s

corpus = pd.read_csv("Chapter-10/input/imdb.csv", nrows=10000)
corpus.loc[:, "review"] = corpus.review.apply(clean_text)
corpus = corpus.review.values

tfv = TfidfVectorizer(tokenizer=word_tokenize, token_pattern=None)
tfv.fit(corpus)

corpus_transformed = tfv.transform(corpus)
svd = decomposition.TruncatedSVD(n_components=10)
corpus_svd = svd.fit(corpus_transformed)

sample_index = 0
feature_scores = dict(
    zip(
        tfv.get_feature_names(),
        corpus_svd.components_[sample_index]
    )
)

N=5
print(sorted(feature_scores, key=feature_scores.get, reverse=True)[:N])