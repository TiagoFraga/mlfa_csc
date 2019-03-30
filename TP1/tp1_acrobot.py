#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 14:19:51 2019

@author: tiagofraga
"""

import gym
import random
import numpy as np
from tensorflow import keras
from keras.models     import Sequential
from keras.layers     import Dense
from keras.optimizers import Adam

###########################################################################################
# MODEL PREPARATION
###########################################################################################

def play_a_random_game_first():
    for step_index in range(goal_steps):
        #env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print("Step {}:".format(step_index))
        print("action: {}".format(action))
        print("observation: {}".format(observation))
        print("reward: {}".format(reward))
        print("done: {}".format(done))
        print("info: {}".format(info))
        if done:
            break
    env.reset()
    


def model_data_preparation():
    training_data = []
    accepted_scores = []

    for game_index in range(intial_games):
        #env.render()
        score = 0
        game_memory = []
        previous_observation = []
        observationList = []
        for step_index in range(goal_steps):
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            observationList.append(observation[4])
            if len(previous_observation) > 0:
                game_memory.append([previous_observation, action])
                
            previous_observation = observation
            score += reward
            if done:
                break
        
        eixo = np.average(observationList)
        print(eixo)
        if (score >= score_requirement or eixo > 2):

            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0, 1, 0]
                elif data[1] == 0:
                    output = [1, 0 ,0]
                else: 
                    output = [0, 0, 1]  
                training_data.append([data[0], output])
        
        env.reset()

    print(accepted_scores)
    
    return training_data



def build_model(input_size, output_size):
    model = Sequential()
    model.add(Dense(128, input_dim=input_size, activation='relu'))
    model.add(Dense(52, activation='relu'))
    model.add(Dense(output_size, activation='softmax'))
    model.compile(loss='mse', optimizer=Adam())
    return model


def train_model(training_data):
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]))
    y = np.array([i[1] for i in training_data]).reshape(-1, len(training_data[0][1]))
    model = build_model(input_size=len(X[0]), output_size=len(y[0]))
    
    model.fit(X, y, epochs=1)
    return model


###########################################################################################
# MAIN
###########################################################################################

gym.envs.register(
        id='Acrobot-v2',entry_point='gym.envs.classic_control:AcrobotEnv',max_episode_steps=1000   
        )


env = gym.make('Acrobot-v2')
env.reset()
goal_steps = 1000
score_requirement = -100
intial_games = 1000

#play_a_random_game_first()
training_data = model_data_preparation()
trained_model = train_model(training_data)


scores = []
choices = []
for each_game in range(1000):
    score = 0
    prev_obs = []
    for step_index in range(goal_steps):
        env.render()
        if len(prev_obs)==0:
            action = env.action_space.sample()
        else:
            print("prev_obs")
            print(trained_model.predict(prev_obs.reshape(-1, len(prev_obs)))[0])
            action = np.argmax(trained_model.predict(prev_obs.reshape(-1, len(prev_obs)))[0])
            print("action")
            print(action)
        
        choices.append(action)
        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        score+=reward
        if done:
            break

    env.reset()
    scores.append(score)

print(scores)
print('Average Score:',sum(scores)/len(scores))
print('choice 1:{}  choice 0:{}  choice 2:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices),choices.count(2)/len(choices)))


