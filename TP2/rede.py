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
import pandas as pd
from sklearn.model_selection import train_test_split


# Converter para supervised learning
def to_supervised(df, tamanho, intervalo_previsao):
    data = df.values
    X, y = list(),list()
    
    for pos in range(len(data)):
        
        tamanho_fim = pos + tamanho
        
        inicio_previsao = tamanho_fim
        
        fim_previsao = tamanho_fim + intervalo_previsao
    
    
        if fim_previsao < len(data):
            X.append(data[pos:tamanho_fim,:])
            y.append(data[inicio_previsao:fim_previsao,5])
        
        
    
    X = np.reshape(np.array(X),(len(X),tamanho,11))
    y = np.reshape(np.array(y),(len(y),1))
        
    return X, y



###################################################################################################
###############################              MAIN              ####################################
###################################################################################################

data = pd.read_csv('Porto_Data/WeatherTraffic/Rotunda_da_Boavista.csv', header=0, sep=',' )
print(data.shape)
data_train = data[0:2600]
data_test = data[2600:3600]


length = 24
nr_features = 11
multi_steps = 1


X,y = to_supervised(data_train, length, multi_steps)


model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(15,input_shape=(length,nr_features)))
model.add(tf.keras.layers.Dense(multi_steps))
model.compile(
        loss=tf.keras.losses.mse,
        optimizer=tf.train.AdamOptimizer(0.001),
        metrics=['accuracy','mae'])

print(model.summary())

model.fit(X,y,shuffle=False,epochs=10,verbose=2)
 
'''  
correct=0
for i in range(100):
    yhat=model.predict(X)
    correct+=1
        
print('Accuracy:%f'%((correct/100)*100.0)) 


X,x=generate_sample(length,nr_features,out_index)
'''

X_test,y_test = to_supervised(data_test,length,multi_steps)


yhat=model.predict(X_test)

print('Expected:%s'% y_test[0])
print('Predicted:%s'% yhat[0])       

