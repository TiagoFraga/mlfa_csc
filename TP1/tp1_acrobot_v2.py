#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 16:03:47 2019

@author: tiagofraga
"""

import gym
import random
import numpy as np
import keras
from statistics import mean, median
from collections import Counter


from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense
from keras.optimizers import Adam
LR = 1e-3


###########################################################################################
# MODEL PREPARATION
###########################################################################################

def showGame(nr = 10):

    for _ in range(nr):
        env.reset()
        
        s2s, s3s = [], []
        while True:
            #env.render()
            
            #action = 2
            action = random.randrange(-1,2)
            observation, reward, done, info = env.step(action)
            #print (observation, reward)
            #print (env.state)
            _,_,_,_, s2,s3 = observation
            s2s.append(s2)
            s3s.append(s3)
            if done:  break
        #print(np.mean(np.array(s2s)))
        #print(np.mean(np.array(s3s)))




def saveGoodGames(nr=10000):
    observations = []
    actions = []
    minReward = -100

    for i in range(nr):
        env.reset()
        action = env.action_space.sample()
        
        obserVationList = []
        actionList = []
        score = 0

        while True:


            env.render()
            
            observation, reward, done, info = env.step(action)
            print(observation)
            action = env.action_space.sample()
            obserVationList.append(observation)
            if action == 1:
                actionList.append([0,1,0] )
            elif action == 0:
                actionList.append([1,0,0])
            else:
                actionList.append([0,0,1])
            
            score += reward
            if done:  break

       
        if score > minReward:
            print(score)
            observations.extend(obserVationList)
            actions.extend(actionList)
    observations = np.array(observations)
    actions = np.array(actions)
    return observations, actions


def trainModell(observations, actions):
    
    model = Sequential()
    model.add(Dense(128,  activation='relu'))
    model.add(Dense(256,  activation='relu'))
    model.add(Dense(256,  activation='relu'))
    model.add(Dense(2,  activation='softmax'))

    model.compile(optimizer=Adam())

    model.fit(observations, actions, epochs=10)
    return model


def playGames(nr, ai):

    observations = []
    actions = []
    minReward = 70
    scores=0
    scores = []

    for i in range(nr):
        env.reset()
        action = env.action_space.sample()
        
        obserVationList = []
        actionList = []
        score=0
        while True:
            #env.render()
            
            observation, reward, done, info = env.step(action)
            action = np.argmax(ai.predict(observation.reshape(1,4)))
            obserVationList.append(observation)
            if action == 1:
                actionList.append([0,1] )
            elif action == 0:
                actionList.append([1,0])
            score += 1
            #score += reward
            if done:  break

        
        print(score)
        scores.append(score)
        if score > minReward:
            observations.extend(obserVationList)
            actions.extend(actionList)
    observations = np.array(observations)
    actions = np.array(actions)
    print (np.mean(scores))
    return observations, actions


###########################################################################################
# MAIN
###########################################################################################

gym.envs.register(
        id='Acrobot-v2',entry_point='gym.envs.classic_control:AcrobotEnv',max_episode_steps=1000   
        )

env = gym.make('Acrobot-v2')
env.reset()

showGame(1)
print("Loading ...")
obs, acts = saveGoodGames()
print ('training 1st modell')
#firstModel = trainModell(obs, acts)
print("Playing the games ...")
#obs, acts = playGames(1000, firstModel)


