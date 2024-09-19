#@title Import Data { display-mode: "form" }
import math
import os
import numpy as np
from bs4 import BeautifulSoup as bs
import requests
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from torchtext.vocab import GloVe

import pickle

import requests, io, zipfile
!wget -O data.zip 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20Fake%20News%20Detection/inspirit_fake_news_resources%20(1).zip'
!unzip data.zip

#Temp getting glove vectors from another resource - Stanford server shutdown till July 4 2023
!wget http://nlp.uoregon.edu/download/embeddings/glove.6B.300d.txt

basepath = '.'

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

with open(os.path.join(basepath, 'train_val_data.pkl'), 'rb') as f:
  train_data, val_data = pickle.load(f)

print('Number of train examples:', len(train_data))
print('Number of val examples:', len(val_data))

def get_description_from_html(html):
  soup = bs(html)
  description_tag = soup.find('meta', attrs={'name':'og:description'}) or soup.find('meta', attrs={'property':'description'}) or soup.find('meta', attrs={'name':'description'})
  if description_tag:
    description = description_tag.get('content') or ''
  else: # If there is no description, return empty string.
    description = ''
  return description

def scrape_description(url):
  if not url.startswith('http'):
    url = 'http://' + url
  response = requests.get(url, timeout=10)
  html = response.text
  description = get_description_from_html(html)
  return description

print('Description of Google.com:')
print(scrape_description('google.com'))
print(train_data[0])

url = "psg.fr" #@param {type:"string"}

def get_descriptions_from_data(data):
  # A dictionary mapping from url to description for the websites in
  # train_data.
  descriptions = []
  for site in tqdm(data):

    descriptions.append(get_description_from_html(site[1]))


  return descriptions


train_descriptions = get_descriptions_from_data(train_data)
train_urls = [url for (url, html, label) in train_data]

print('\nNYTimes Description:')
print(train_descriptions[train_urls.index('nytimes.com')])

val_descriptions = get_descriptions_from_data(val_data)

vectorizer = CountVectorizer(max_features=300)

vectorizer.fit(train_descriptions)

def vectorize_data_descriptions(descriptions, vectorizer):
  X = vectorizer.transform(descriptions).todense()
  return X

print('\nPreparing train data...')
bow_X_train = vectorize_data_descriptions(train_descriptions, vectorizer)
bow_y_train = [label for url, html, label in train_data]

print('\nPreparing val data...')

bow_X_val = vectorize_data_descriptions(val_descriptions, vectorizer)
bow_y_val = [label for url, html, label in val_data]

bow_X_train, bow_X_val = np.array(bow_X_train), np.array(bow_X_val)

model = LogisticRegression()

model.fit(bow_X_train, bow_y_train)
bow_y_train_pred = model.predict(bow_X_train)
print(accuracy_score(bow_y_train_pred, bow_y_train))

bow_y_val_pred = model.predict(bow_X_val)
print(accuracy_score(bow_y_val_pred, bow_y_val))

VEC_SIZE = 300
glove = GloVe(name='6B', dim=VEC_SIZE)

# Returns word vector for word if it exists, else return None.
def get_word_vector(word):
    try:
      return glove.vectors[glove.stoi[word.lower()]].numpy()
    except KeyError:
      return None

good_vector = get_word_vector("horrible")

#@title Word Similarity { run: "auto", display-mode: "both" }

def cosine_similarity(vec1, vec2):
  return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

word1 = "hotel" #@param {type:"string"}
word2 = "motel" #@param {type:"string"}

print('Word 1:', word1)
print('Word 2:', word2)

def cosine_similarity_of_words(word1, word2):
  vec1 = get_word_vector(word1)
  vec2 = get_word_vector(word2)

  if vec1 is None:
    print(word1, 'is not a valid word. Try another.')
  if vec2 is None:
    print(word2, 'is not a valid word. Try another.')
  if vec1 is None or vec2 is None:
    return None

  return cosine_similarity(vec1, vec2)

def glove_transform_data_descriptions(descriptions):
    X = np.zeros((len(descriptions), VEC_SIZE))
    for i, description in enumerate(descriptions):
        found_words = 0.0
        description = description.strip()
        for word in description.split():
            vec = get_word_vector(word)
            if vec is not None:
                # Increment found_words and add vec to X[i].
                found_words += 1
                X[i] += vec
        # We divide the sum by the number of words added, so we have the
        # average word vector.
        if found_words > 0:
            X[i] /= found_words

    return X

glove_X_train = glove_transform_data_descriptions(train_descriptions)
glove_y_train = [label for (url, html, label) in train_data]

glove_X_val = glove_transform_data_descriptions(val_descriptions)
glove_y_val = [label for (url, html, label) in val_data]

model = LogisticRegression()
model.fit(glove_X_train,glove_y_train)

glove_y_train_pred = model.predict(glove_X_train)
print(accuracy_score(glove_y_train,glove_y_train_pred))

glove_y_val_pred = model.predict(glove_X_val)
print(accuracy_score(glove_y_val,glove_y_val_pred))

def train_model(X_train, y_train, X_val, y_val):
  model = LogisticRegression(solver='liblinear')
  model.fit(X_train, y_train)

  return model


def train_and_evaluate_model(X_train, y_train, X_val, y_val):
  model = train_model(X_train, y_train, X_val, y_val)
  y_train_pred = model.predict(X_train)
  print("Train:", accuracy_score(y_train_pred,y_train))

  y_val_pred = model.predict(X_val)
  print("Val:", accuracy_score(y_val_pred,y_val))

  print(confusion_matrix(y_val,y_val_pred))

  prf = precision_recall_fscore_support(y_val, y_val_pred)

  print('Precision:', prf[0][1])
  print('Recall:', prf[1][1])
  print('F-Score:', prf[2][1])

  return model

def prepare_data(data, featurizer):
    X = []
    y = []
    for datapoint in data:
        url, html, label = datapoint
        # We convert all text in HTML to lowercase, so <p>Hello.</p> is mapped to
        # <p>hello</p>. This will help us later when we extract features from
        # the HTML, as we will be able to rely on the HTML being lowercase.
        html = html.lower()
        y.append(label)

        features = featurizer(url, html)

        # Gets the keys of the dictionary as descriptions, gets the values
        # as the numerical features. Don't worry about exactly what zip does!
        feature_descriptions, feature_values = zip(*features.items())

        X.append(feature_values)

    return X, y, feature_descriptions

# Gets the log count of a phrase/keyword in HTML (transforming the phrase/keyword
# to lowercase).
def get_normalized_count(html, phrase):
    return math.log(1 + html.count(phrase.lower()))

# Returns a dictionary mapping from plaintext feature descriptions to numerical
# features for a (url, html) pair.
def keyword_featurizer(url, html):
    features = {}

    # Same as before.
    features['.com domain'] = url.endswith('.com')
    features['.org domain'] = url.endswith('.org')
    features['.net domain'] = url.endswith('.net')
    features['.info domain'] = url.endswith('.info')
    features['.org domain'] = url.endswith('.org')
    features['.biz domain'] = url.endswith('.biz')
    features['.ru domain'] = url.endswith('.ru')
    features['.co.uk domain'] = url.endswith('.co.uk')
    features['.co domain'] = url.endswith('.co')
    features['.tv domain'] = url.endswith('.tv')
    features['.news domain'] = url.endswith('.news')

    keywords = ['trump', 'biden', 'clinton', 'sports', 'finance', 'election', 'covid']

    for keyword in keywords:
      features[keyword + ' keyword'] = get_normalized_count(html, keyword)

    return features

keyword_X_train, y_train, feature_descriptions = prepare_data(train_data, keyword_featurizer)
keyword_X_val, y_val, feature_descriptions = prepare_data(val_data, keyword_featurizer)

train_and_evaluate_model(keyword_X_train, y_train, keyword_X_val, y_val)

vectorizer = CountVectorizer(max_features=300)

vectorizer.fit(train_descriptions)

def vectorize_data_descriptions(data_descriptions, vectorizer):
  X = vectorizer.transform(data_descriptions).todense()
  return X

bow_X_train = vectorize_data_descriptions(train_descriptions, vectorizer)
bow_X_val = vectorize_data_descriptions(val_descriptions, vectorizer)
bow_X_train, bow_X_val = np.array(bow_X_train), np.array(bow_X_val)

train_and_evaluate_model(bow_X_train, y_train, bow_X_val, y_val)

VEC_SIZE = 300
glove = GloVe(name='6B', dim=VEC_SIZE)

# Returns word vector for word if it exists, else return None.
def get_word_vector(word):
    try:
      return glove.vectors[glove.stoi[word.lower()]].numpy()
    except KeyError:
      return None

def glove_transform_data_descriptions(descriptions):
    X = np.zeros((len(descriptions), VEC_SIZE))
    for i, description in enumerate(descriptions):
        found_words = 0.0
        description = description.strip()
        for word in description.split():
            vec = get_word_vector(word)
            if vec is not None:
                # Increment found_words and add vec to X[i].
                found_words += 1
                X[i] += vec
        # We divide the sum by the number of words added, so we have the
        # average word vector.
        if found_words > 0:
            X[i] /= found_words

    return X

# Note that you can use y_train and y_val from before, since these are the
# same for both the keyword approach and the BOW approach.

glove_X_train = glove_transform_data_descriptions(train_descriptions)
glove_X_val = glove_transform_data_descriptions(val_descriptions)

train_and_evaluate_model(glove_X_train, y_train, glove_X_val, y_val)

def combine_features(X_list):
  return np.concatenate(X_list, axis=1)
# First, produce combined_X_train and combined_X_val using 2 calls to
# combine_features, using keyword_X_train, bow_X_train, glove_X_train
# and keyword_X_val, bow_X_val, glove_X_val from before.

combined_X_train = combine_features([keyword_X_train, bow_X_train, glove_X_train])
combined_X_val = combine_features([keyword_X_val, bow_X_val, glove_X_val])

model = train_and_evaluate_model(combined_X_train, y_train, combined_X_val, y_val)

#@title Live Fake News Classification Demo { run: "auto", vertical-output: true, display-mode: "both" }
def get_data_pair(url):
  if not url.startswith('http'):
      url = 'http://' + url
  url_pretty = url
  if url_pretty.startswith('http://'):
      url_pretty = url_pretty[7:]
  if url_pretty.startswith('https://'):
      url_pretty = url_pretty[8:]

  # Scrape website for HTML
  response = requests.get(url, timeout=10)
  htmltext = response.text

  return url_pretty, htmltext

curr_url = "www.yahoo.com" #@param {type:"string"}

url, html = get_data_pair(curr_url)

# Call on the output of *keyword_featurizer* or something similar
# to transform it into a format that allows for concatenation. See
# example below.
def dict_to_features(features_dict):
  X = np.array(list(features_dict.values())).astype('float')
  X = X[np.newaxis, :]
  return X
def featurize_data_pair(url, html):
  # Approach 1.
  keyword_X = dict_to_features(keyword_featurizer(url, html))
  # Approach 2.
  description = get_description_from_html(html)

  bow_X = vectorize_data_descriptions([description], vectorizer)

  # Approach 3.
  glove_X = glove_transform_data_descriptions([description])

  X = combine_features([keyword_X, bow_X, glove_X])

  return X

curr_X = np.array(featurize_data_pair(url, html))

model = train_model(combined_X_train, y_train, combined_X_val, y_val)

curr_y = model.predict(curr_X)[0]


if curr_y < .5:
  print(curr_url, 'appears to be real.')
else:
  print(curr_url, 'appears to be fake.')
