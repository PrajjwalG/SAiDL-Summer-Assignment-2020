#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


# importing data
train = pd.read_csv("D:/OneDrive/Desktop/Summer-Induction-Assignment-2020/Question 4.2/NLP Data/data/train.csv")
print(train.shape)
train.head() 


# In[3]:


test = pd.read_csv("D:/OneDrive/Desktop/Summer-Induction-Assignment-2020/Question 4.2/NLP Data/data/val.csv")
test.head()


# In[4]:


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


# In[5]:


# data pre processing
def preprocess_text(sen):
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sen)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence


# In[6]:


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


# In[7]:


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



# In[8]:


from sklearn import preprocessing

# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()

# Encode labels in column 'flair'.
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit_transform(y_test)


# In[9]:


from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# In[10]:

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


# In[11]:


# word embeddings
from numpy import array
from numpy import asarray
from numpy import zeros

embeddings_dictionary = dict()

glove_file = open('D:/OneDrive/Desktop/New folder (3)/glove.6B.100d.txt', encoding="utf8")

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


# In[12]:

# stacking title and post 
X_train3 = np.hstack((X_train2, X_train))
X_test3 = np.hstack((X_test2, X_test))


# In[13]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D,BatchNormalization,Dense
model = Sequential([

tf.keras.layers.Embedding(vocab_size , 100, weights=[embedding_matrix], input_length=maxlen+100, trainable=True),
tf.keras.layers.LSTM(128,return_sequences = True),
# Add Convolutional layer
Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'),
# Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'),
MaxPooling1D(3),
GlobalMaxPooling1D(),
BatchNormalization(),
# Add fully connected layers
Dense(150, activation='relu'),
Dense(100, activation='relu'),
Dense(50, activation='relu'),
Dense(15, activation='sigmoid')])
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])





# In[15]:


# training 
history = model.fit(X_train3 , y_train, batch_size=128, epochs=6, verbose=1, validation_data=(X_test3, y_test))


# In[16]:

# prediction 
y_pred = model.predict(X_test3, batch_size=64, verbose=1)


# In[17]:


y_pred = np.array(y_pred)
y_true = np.array(y_test)
y_pred = np.argmax(y_pred, axis=1) # assuming you have n-by-5 class_prob
y_true = np.argmax(y_true, axis=1) 


# In[19]:


from sklearn.metrics import f1_score
f1_score(y_true, y_pred, average='macro')
f1_score(y_true, y_pred, average='micro')





