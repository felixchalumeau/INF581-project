import numpy as np
import matplotlib.pyplot as plt
import gym
import random

import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical

# TODO: explain following
ENV_NAME = "CarRacing-v0"
EPISODE_DURATION = 15000
EPISODE_DURATION_AUGM = 25
ALPHA_INIT = 0.1 # (can also decay this over time..)
SCORE = 900
TEST_TIME = 1
TEST_TIME_AUGM = 1
LEFT = np.array([-1, 0, 0])
RIGHT = np.array([1, 0, 0])

LAMBDA = 1 # to smooth decisions

EPSILON = 0.2

VERBOSE = True

#########################################################################################

# First: we accelerate once every two frames and choose left or right each time
DISCRETE_ACTION_SPACE = [np.array([-1, 0, 0]), np.array([1, 0, 0])]

# [np.array([-1, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 0]), np.array([0, 0, 1])]

#########################################################################################

# from complex to simple image (observation)
def preprocess(rgb):
    '''
    Simplifie l'image. PAsse de RGB à gray et enlève le base inutile.
    '''
    end_img = 84
    
    gray = np.dot(rgb[...,:3], [0.0, 0.5, 0.5])
    gray[gray>150] = 180
    return gray[:84]

# get useful informations from an observation
def capteur(observation):
    '''
    Position du nez de la voiture codé en dur pour l'instant.
    Cette fonction renvoie les quatres distances d'intérêt !
    '''
    i_nose = 67
    i_back = 76
    j_left = 46
    j_right = 49
    
    grass_color = 180
    road_color = 102

    # informations horizontales
    # nose

    hori = []
    for i_ in [i_nose, i_back]:
        horizontal = np.where(observation[i_] == grass_color)[0]
        try:
            hori_g = 46 - horizontal[horizontal < 46][-1]
        except:
            hori_g = 46
        try:
            hori_d = horizontal[horizontal > 49][0] - 49
        except:
            hori_d = 46
        hori += [hori_g, hori_d]


    # informations verticales
    vertical_gauche = np.where(observation[:, 46] == grass_color)[0]
    try:
        verti_gauche = 67 - vertical_gauche[vertical_gauche < 67][-1] 
    except:
        verti_gauche = 67

    vertical_droite = np.where(observation[:, 49] == grass_color)[0]
    try:
        verti_droite = 67 - vertical_droite[vertical_droite < 67][-1]
    except:
        verti_droite = 67
    
    res = np.array(hori + [verti_gauche, verti_droite])

    res[res == 1] = -1

    return res/20

# a kind of complete preprocessing of the images observed
def useful_from_observation(rgb):
    '''
    From an observation (or a state), which is a 96x96 RGB image, we return 4 distances, which will be used by our classifier.
    '''
    gray = preprocess(rgb)
    return capteur(gray)

#################################################################################
learning_rate = 0.01
gamma = 0.99

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.state_space = 12
        self.action_space = 2
        
        self.l1 = nn.Linear(self.state_space, 24, bias=False)
        self.l2 = nn.Linear(24, self.action_space, bias=False)
        
        self.gamma = gamma
        
        # Episode policy and reward history 
        self.policy_history = Variable(torch.Tensor()) 
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []

    def forward(self, x):    
        model = torch.nn.Sequential(
            self.l1,
            nn.Dropout(p=0.6),
            nn.ReLU(),
            self.l2,
            nn.Softmax(dim=-1)
        )
        return model(x)

policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

def select_action(state):
    #Select an action (0 or 1) by running policy model and choosing based on the probabilities in state
    state = torch.from_numpy(state).type(torch.FloatTensor)
    state = policy(Variable(state))
    c = Categorical(state)
    action = c.sample()
    
    # Add log probability of our chosen action to our history    
    if policy.policy_history.dim() != 0:
        policy.policy_history = torch.cat([policy.policy_history, c.log_prob(action)])
    else:
        policy.policy_history = (c.log_prob(action))
    return action



#################################################################################

# transparent
def sigmoid(x, lmbd=LAMBDA):
    return 1.0 / (1.0 + np.exp(-lmbd*x))

# Return policy
def get_policy(s, theta):

    p_right = sigmoid(np.dot(s, np.transpose(theta)), lmbd=LAMBDA)
    pi = [1-p_right, p_right]
    return pi
    # return 2*p_right - 1

# Draw an action according to current policy
def act_with_policy(s, theta):
    p_right = get_policy(s, theta)[1]
    #p_right = get_policy(s, theta)
    #r=0
    #if np.random.rand() < EPSILON:
    #    r = np.random.rand()
    #    if p_right >0:
    #        r = -r

    r = np.random.rand()
    if r < p_right:
        return np.array([1.0, 0, 0])
    else:
        return np.array([-1.0, 0, 0])

    # turn = np.clip(np.array([p_right + r*10*EPSILON]), a_min = -1, a_max = 1)[0][0]
    #print([p_right, r, turn])
    #return np.array([turn, 0.0, 0.0])

# Generate an episode
def gen_rollout(env, theta, max_episode_length=EPISODE_DURATION, render=False):

    s_t = env.reset()
    s_t = useful_from_observation(s_t)
    s_t = np.concatenate((s_t, np.array([0, 0, 0, 0, 0, 0])))

    episode_states = []
    episode_actions = []
    episode_rewards = []
    episode_states.append(s_t)

    to_stop = 25

    for t in range(max_episode_length):

        if render:
            env.render()

        a_t = act_with_policy(s_t, theta)

        if t%4 == 0:
            a_t += np.array([0.0, 1.0, 0.0])

        s_t, r_t, done, info = env.step(a_t)
        
        # transform the state to a usable one
        s_t = useful_from_observation(s_t)
        old_s_t = episode_states[-1][:6]
        s_t = np.concatenate((s_t, s_t-old_s_t))
        print(s_t)
        # working on the reward
        # r_t += (- s_t[0]**2 - s_t[1]**2)
        if ((s_t[:2]==np.array([-0.05, -0.05])).all()):
            r_t = r_t - 2
            to_stop = to_stop - 1
        elif (s_t[0]==-0.05 or s_t[1]==-0.05):
            r_t = r_t - 1

        print(r_t)
        episode_states.append(s_t)
        episode_actions.append(a_t)
        episode_rewards.append(r_t)


        if done or to_stop==0:
            break

    return episode_states, episode_actions, episode_rewards

def test_policy(env, theta, score = SCORE, num_episodes = TEST_TIME , max_episode_length=EPISODE_DURATION, render=False):
    
    num_success = 0
    average_return = 0

    for i_episode in range(num_episodes):
        _, _, episode_rewards = gen_rollout(env, theta, max_episode_length, render)

        total_rewards = sum(episode_rewards)

        if total_rewards > score:
            num_success+=1

        average_return += (1.0 / num_episodes) * total_rewards

        if render:
            print("Test Episode {0}: Total Reward = {1} - Success = {2}".format(i_episode,total_rewards,total_rewards>score))


    if average_return > score:
        success = True
    else:
        success = False

    return success, num_success, average_return

# Returns policy gradient for a given episode 
def compute_PG(episode_states, episode_actions, episode_rewards, theta):

    H = len(episode_rewards)
    PG = 0

    for t in range(H):
        pi = get_policy(episode_states[t], theta)
        a_t = episode_actions[t]
        R_t = sum(episode_rewards[t::])

        if (a_t[0] == 1):
            g_theta_log_pi = - pi[0] * episode_states[t] * R_t
        else:
            g_theta_log_pi = pi[1] * episode_states[t] * R_t
        
        # g_theta_log_pi = pi * episode_states[t] * R_t
        # g_theta_log_pi = LAMBDA * a_t[0] * episode_states[t] * R_t # * np.exp(-np.dot(episode_states[t], np.transpose(theta)))
        PG += g_theta_log_pi

    return PG

# Train until average_return is larger than SCORE
def train(env, theta_init, max_episode_length = EPISODE_DURATION, alpha_init = ALPHA_INIT):

    theta = theta_init
    i_episode = 0
    average_returns = []

    success, _, R = test_policy(env, theta, render=True)
    average_returns.append(R)

    # Train until success
    while (not success):

        # Rollout
        episode_states, episode_actions, episode_rewards = gen_rollout(env, theta, max_episode_length, render=True)

        # Schedule step size
        alpha = alpha_init / (1 + i_episode)

        # Compute gradient
        PG = compute_PG(episode_states, episode_actions, episode_rewards, theta)

        # Do gradient ascent 
        theta += alpha * PG

        # Normalize theta
        norm = theta@theta.T
        norm = norm.flatten()
        norm = norm[0]**0.5

        # theta = theta/norm

        # Test new policy 
        success, _, R = test_policy(env, theta, score=SCORE, max_episode_length= max_episode_length, render=True)

        # Monitor 
        average_returns.append(R)

        i_episode += 1

        if VERBOSE:
            print("Episode {0}, average return: {1}".format(i_episode, R))

    return theta, i_episode, average_returns


def main():

    env = gym.make(ENV_NAME)

    # dim = env.observation_space.shape[0]

    # Init parameters to random
    theta_init = np.random.randn(1, 12)

    # Train agent
    theta, i, average_returns = train(env, theta_init)

    print("Solved after {} iterations".format(i))

    # Test final policy
    test_policy(env,theta, num_episodes=10, render=True)

    # Show training curve
    plt.plot(range(len(average_returns)),average_returns)
    plt.title("Average reward on 100 episodes")
    plt.xlabel("Training Steps")
    plt.ylabel("Reward")

    plt.show()

if __name__ == "__main__":

    main()
