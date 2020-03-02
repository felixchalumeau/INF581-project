import gym
import numpy as np
import random

env = gym.make('CarRacing-v0')
env.seed(42)

new_action_space = [np.array([-1, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 0]), np.array([0, 0, 1])]

for i_episode in range(2):
    observation = env.reset()
    for t in range(200):
        env.render()
        print(observation)
        
        action = random.choice(new_action_space)
        action = np.array([0, 0, 0])
        if t%2 ==0 : 
            action = np.array([0, 1, 0])

        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
 
env.close()

print("State space dimension is:", env.observation_space.shape[0])
print("State upper bounds:", env.observation_space.high)
print("State lower bounds:", env.observation_space.low)

# print("Number of actions is:", env.action_space.n)
