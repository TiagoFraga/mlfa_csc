# -*- coding: utf-8 -*-
"""
Created on Wed May  8 15:45:23 2019

@author: joelt
"""
import tensorflow as tf
import numpy as np
from random import randint
from numpy import array
from numpy import argmax


def generate_sequence(length,nr_features):
    return[randint(0,nr_features-1) for _ in range(length)]
    
def one_hot_encode(sequence,nr_features):
    encoded = list()
    for value in sequence:
        one_hot_encoded = np.zeros(nr_features)
        one_hot_encoded[value] = 1
        encoded.append(one_hot_encoded)
    return array(encoded)

def one_hot_decode(encoded_seq):
    return[argmax(value) for value in encoded_seq]   
    
def generate_sample(length,nr_features,out_index):
    sequence = generate_sequence(length,nr_features)
    encoded = one_hot_encode(sequence,nr_features)
    X = encoded.reshape((1,length,nr_features))
    y = encoded[out_index].reshape(1,nr_features)
    return X,y

length = 5
nr_features = 10
out_index = 2

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(15,input_shape=(length,nr_features)))
model.add(tf.keras.layers.Dense(nr_features,activation='softmax'))
model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=tf.train.AdamOptimizer(0.001),
        metrics=['accuracy'])

print(model.summary())

for i in range(10000):
    X,y=generate_sample(length,nr_features,out_index)
    model.fit(X,y,shuffle=False,epochs=1,verbose=2)
    
correct=0
for i in range(100):
    X,y=generate_sample(length,nr_features,out_index)
    yhat=model.predict(X)
    if one_hot_decode(yhat)==one_hot_decode(y):
        correct+=1
print('Accuracy:%f'%((correct/100)*100.0)) 

X,x=generate_sample(length,nr_features,out_index)
yhat=model.predict(X)

print('Sequence:%s'%[one_hot_decode(x) for x in X])
print('Expected:%s'%one_hot_decode(y))
print('Predicted:%s'%one_hot_decode(yhat))       