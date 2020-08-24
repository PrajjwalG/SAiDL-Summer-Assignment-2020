# -*- coding: utf-8 -*-
"""Untitled

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18bjdFETDBN3DxFw4sfXEKvhBUMWIUMt9
"""

from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.layers.merge import Concatenate

import pandas as pd
import numpy as np


import matplotlib.pyplot as plt

# importing data
train = pd.read_csv("train.csv")
print(train.shape)
train.head()

test = pd.read_csv("val.csv")
test.head()

# dropping data that is null
filter = train["Post"] != ""
train = train[filter]
train = train.dropna()

filter = test["Post"] != ""
test = test[filter]
test = test.dropna()
filter = train["Title"] != ""
train = train[filter]
train = train.dropna()

filter = test["Title"] != ""
test = test[filter]
test = test.dropna()

# data pre processing
def preprocess_text(sen):
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sen)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence

import re
X_train = []
sentences = list(train["Post"])
for sen in sentences:
    X_train.append(preprocess_text(sen))

y_train = train['Flair']
X_train2 = []
sentences = list(train["Title"])
for sen in sentences:
    X_train2.append(preprocess_text(sen))

import re
X_test = []
sentences = list(test["Post"])
for sen in sentences:
    X_test.append(preprocess_text(sen))

y_test = test['Flair']
X_test2 = []
sentences = list(test["Title"])
for sen in sentences:
    X_test2.append(preprocess_text(sen))

from sklearn import preprocessing

# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()

# Encode labels in column 'flair'.
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit_transform(y_test)

from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# tokenizer 
tokenizer= Tokenizer(num_words=9000,
                     filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                     lower=True,
                     split=" ",
                     char_level=True)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
X_train2 = tokenizer.texts_to_sequences(X_train2)
X_test2 = tokenizer.texts_to_sequences(X_test2)
vocab_size = len(tokenizer.word_index) + 1

maxlen = 450

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
X_train2 = pad_sequences(X_train2, padding='post', maxlen=100)
X_test2 = pad_sequences(X_test2, padding='post', maxlen=100)

# word embeddings
from numpy import array
from numpy import asarray
from numpy import zeros

embeddings_dictionary = dict()

glove_file = open('/content/drive/My Drive/glove.6B.100d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions
glove_file.close()

embedding_matrix = zeros((vocab_size, 100))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

# stacking title and post
X_train3 = np.hstack((X_train2, X_train))
X_test3 = np.hstack((X_test2, X_test))

# model
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D,BatchNormalization,Dense
from keras_self_attention import SeqSelfAttention
model = Sequential([
Embedding(vocab_size, 100, weights=[embedding_matrix],input_length=maxlen+100,trainable=False),
LSTM(128, return_sequences=True),
SeqSelfAttention(attention_activation='sigmoid'),
Conv1D(32,3,padding = 'same',activation = 'relu'),
MaxPooling1D(3),
GlobalMaxPooling1D(),
BatchNormalization(),
Dense(150, activation='relu'),
Dense(100, activation='relu'),
Dense(50, activation='relu'),
Dense(15, activation='sigmoid'),
])

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])


# training 
history = model.fit(X_train3 , y_train, batch_size=128, epochs=15, verbose=1, validation_data=(X_test3, y_test))

# prediction 
y_pred = model.predict(X_test3, batch_size=64, verbose=1)

y_pred = np.array(y_pred)
y_true = np.array(y_test)
y_pred = np.argmax(y_pred, axis=1) # assuming you have n-by-5 class_prob
y_true = np.argmax(y_true, axis=1)

from sklearn.metrics import f1_score
f1_score(y_true, y_pred, average='macro')
f1_score(y_true, y_pred, average='micro')

# for testing dataset put name of csv in Test
Test = pd.read_csv(" .csv")
filter = Test["Post"] != ""
Test = Test[filter]
Test = Test.dropna()

filter = Test["Title"] != ""
Test = Test[filter]
Test = Test.dropna()

X_Test = []
sentences = list(Test["Post"])
for sen in sentences:
    X_Test.append(preprocess_text(sen))

y_Test = Test['Flair']
X_Test2 = []
sentences = list(Test["Title"])
for sen in sentences:
    X_Test2.append(preprocess_text(sen))

y_Test = label_encoder.fit_transform(y_Test)
y_Test = to_categorical(y_Test)
X_Test = tokenizer.texts_to_sequences(X_Test)
X_Test2 = tokenizer.texts_to_sequences(X_Test2)
X_Test = pad_sequences(X_Test, padding='post', maxlen=maxlen)
X_Test2 = pad_sequences(X_Test2, padding='post', maxlen=100)
X_Test3 = np.hstack((X_Test2, X_Test))
y_pred = model.predict(X_Test3, batch_size=64, verbose=1)
y_pred = np.array(y_pred)
y_true = np.array(y_Test)
y_pred = np.argmax(y_pred, axis=1) 
y_true = np.argmax(y_true, axis=1)
f1_score(y_true, y_pred, average='micro')
f1_score(y_true, y_pred, average='macro')