#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import csv
import pandas

###################### CLEAN DATASETS ########################
# -*- coding: utf-8 -*-

import sklearn
from random import randint
from numpy import array
from numpy import array,argmax
from pandas import DataFrame
from pandas import concat
from pandas import Series
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

df = pandas.read_csv('Rotunda_da_Boavista.csv')

df['hour'] = df.hour.astype(int)

print(df.dtypes)

#print(df)

# Normalizar os dados

series = df['speed_diff']

values = series.values.reshape((len(series.values),1))
scaler = MinMaxScaler(feature_range=(0,1))
normalized = scaler.fit_transform(values)
inversed = scaler.inverse_transform(values)
#print(inversed)
# One hot-encoding

label_encoder = LabelEncoder()
label_encoded = label_encoder.fit_transform(df['road_name'])
#print(label_encoded)

onehot_encoder = OneHotEncoder(sparse=False,categories='auto')
label_encoded = label_encoded.reshape(len(label_encoded),1)
onehot = onehot_encoder.fit_transform(label_encoded)
#print(onehot)



# Passar para Supervised Learning
"""
def to_supervised(sequence, n_in, n_out):
	# create lag copies of the sequence
	sl = DataFrame(sequence)
	sl = concat([df.shift(n_in-i-1) for i in range(n_in)], axis=1)
	# drop rows with missing values
	sl.dropna(inplace=True)
	# specify columns for input and output pairs
	values = sl.values
	width = sequence.shape[1]
	X = values.reshape(len(values), n_in, width)
	y = values[:, 0:(n_out*width)].reshape(len(values), n_out, width)
	return X, y
"""
#Esta é a maneira que o stor fez, mas parece rafado, vamos tentar meter a de cima a funcionar

df = DataFrame(df['hour'])
df['t-1'] = df['hour'].shift(1)
df['t+1'] = df['hour'].shift(-1)
print(df)


# (Samples,Timesteps,Features)

model = Sequential()
model.add(LSTM(5,batch_input_shape=(13449,5,1)))

"""
# prepare data for the LSTM
def get_data(n_in, n_out):
	# generate random sequence
	sequence = generate_sequence()
	# one hot encode
	encoded = one_hot_encode(sequence)
	# convert to X,y pairs
	X,y = to_supervised(encoded, n_in, n_out)
	return X,y
 
# define LSTM
n_in = 5
n_out = 5
encoded_length = 100
batch_size = 7
model = Sequential()
model.add(LSTM(20, batch_input_shape=(batch_size, n_in, encoded_length), return_sequences=True, stateful=True))
model.add(TimeDistributed(Dense(encoded_length, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# train LSTM
for epoch in range(500):
	# generate new random sequence
	X,y = get_data(n_in, n_out)
	# fit model for one epoch on this sequence
	model.fit(X, y, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
	model.reset_states()
# evaluate LSTM
X,y = get_data(n_in, n_out)
yhat = model.predict(X, batch_size=batch_size, verbose=0)
# decode all pairs
for i in range(len(X)):
	print('Expected:', one_hot_decode(y[i]), 'Predicted', one_hot_decode(yhat[i]))


















############################### MAIN ##################################################    

weather = "Porto_Data/weather.csv"
incidents = "Porto_Data/incidents.csv"
traffic = "Porto_Data/traffic.csv"



final_weather = "Porto_Data/weather_result.csv"
final_incidents = "Porto_Data/incidents_result.csv"
final_traffic = "Porto_Data/traffic_result.csv"

hour_weather = "Porto_Data/weather_hour.csv"


cleanWeather(weather,final_weather)
cleanIncidents(incidents, final_incidents)
cleanTraffic(traffic, final_traffic)

streets = {}
streets = getAllStreets(traffic)

getDataFromStreet(streets, final_traffic)

cleanDateStreets(streets)

cleanDate(final_weather)

joinStreetWeather(streets,hour_weather)

#joinFiles(streets,final_weather)

# generate a sequence of random numbers in [0, 99]
def generate_sequence(length=25):
	return [randint(0, 99) for _ in range(length)]
 
# one hot encode sequence
def one_hot_encode(sequence, n_unique=100):
	encoding = list()
	for value in sequence:
		vector = [0 for _ in range(n_unique)]
		vector[value] = 1
		encoding.append(vector)
	return array(encoding)
 
# decode a one hot encoded string
def one_hot_decode(encoded_seq):
	return [argmax(vector) for vector in encoded_seq]
 
# convert encoded sequence to supervised learning
def to_supervised(sequence, n_in, n_out):
	# create lag copies of the sequence
	df = DataFrame(sequence)
	df = concat([df.shift(n_in-i-1) for i in range(n_in)], axis=1)
	# drop rows with missing values
	df.dropna(inplace=True)
	# specify columns for input and output pairs
	values = df.values
	width = sequence.shape[1]
	X = values.reshape(len(values), n_in, width)
	y = values[:, 0:(n_out*width)].reshape(len(values), n_out, width)
	return X, y
 
# prepare data for the LSTM
def get_data(n_in, n_out):
	# generate random sequence
	sequence = generate_sequence()
	# one hot encode
	encoded = one_hot_encode(sequence)
	# convert to X,y pairs
	X,y = to_supervised(encoded, n_in, n_out)
	return X,y
 
# define LSTM
n_in = 5
n_out = 5
encoded_length = 100
batch_size = 7
model = Sequential()
model.add(LSTM(20, batch_input_shape=(batch_size, n_in, encoded_length), return_sequences=True, stateful=True))
model.add(TimeDistributed(Dense(encoded_length, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# train LSTM
for epoch in range(500):
	# generate new random sequence
	X,y = get_data(n_in, n_out)
	# fit model for one epoch on this sequence
	model.fit(X, y, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
	model.reset_states()
# evaluate LSTM
X,y = get_data(n_in, n_out)
yhat = model.predict(X, batch_size=batch_size, verbose=0)
# decode all pairs
for i in range(len(X)):
	print('Expected:', one_hot_decode(y[i]), 'Predicted', one_hot_decode(yhat[i]))
        
  """      
        
        
        
        
        
        
