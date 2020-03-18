import gym
import numpy as np
import random
import tensorflow.keras as keras
import seaborn as sns

env = gym.make('CarRacing-v0')

new_action_space = [np.array([-1, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 0]), np.array([0, 0, 1])]

for i_episode in range(2):
    observation = env.reset()
    for t in range(200):
        env.render()        
        action = random.choice(new_action_space)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
