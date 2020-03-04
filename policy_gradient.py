import numpy as np
import matplotlib.pyplot as plt
import gym
import random

# TODO: explain following
ENV_NAME = "CarRacing-v0"
EPISODE_DURATION = 15000
EPISODE_DURATION_AUGM = 25
ALPHA_INIT = 0.01 # (can also decay this over time..)
SCORE = 300
TEST_TIME = 0
TEST_TIME_AUGM = 1
LEFT = np.array([-1, 0, 0])
RIGHT = np.array([1, 0, 0])

GAMMA = 0.99

EPSILON = 0.05

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
    j_left = 46
    j_right = 49
    
    grass_color = 180
    road_color = 102

    # informations horizontales
    horizontal = np.where(observation[67] == grass_color)[0]
    try:
        hori_gauche = 46 - horizontal[horizontal < 46][-1]
    except:
        hori_gauche = 46
    try:
        hori_droite = horizontal[horizontal > 49][0] - 49
    except:
        hori_droite = 46
    
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

    res = np.array([hori_gauche, hori_droite, verti_gauche, verti_droite])
    res[res == 1] == 0

    return res/20

# a kind of complete preprocessing of the images observed
def useful_from_observation(rgb):
    '''
    From an observation (or a state), which is a 96x96 RGB image, we return 4 distances, which will be used by our classifier.
    '''
    gray = preprocess(rgb)
    return capteur(gray)

#################################################################################

# transparent
def sigmoid(x):
    x = np.clip(x, -100, 100)
    return 1.0 / (1.0 + np.exp(-x))

# Return policy
def get_policy(s, theta):

    p_right = sigmoid(np.dot(s, np.transpose(theta)))
    # pi = [1-p_right, p_right]
    # return pi
    return 2*p_right - 1

# Draw an action according to current policy
def act_with_policy(s, theta):
    # p_right = get_policy(s, theta)[1]
    p_right = get_policy(s, theta)
    r = np.random.rand()
    # if r < EPSILON:
    #    return np.array([1, 0, 0])
    #else:
    #    return np.array([-1, 0, 0])
    turn = np.clip(np.array([p_right + r*EPSILON]), a_min = -1, a_max = 1)[0][0]
    return np.array([turn, 0.0, 0.0])

# Generate an episode
def gen_rollout(env, theta, max_episode_length=EPISODE_DURATION, render=False):

    s_t = env.reset()
    s_t = useful_from_observation(s_t)
    s_t = np.concatenate((s_t, np.array([0, 0, 0, 0])))

    episode_states = []
    episode_actions = []
    episode_rewards = []
    episode_states.append(s_t)

    to_stop = 15

    for t in range(max_episode_length):

        if render:
            env.render()

        a_t = act_with_policy(s_t, theta)

        if t%6 == 0:
            a_t += np.array([0.0, 1.0, 0.0])

        s_t, r_t, done, info = env.step(a_t)
        
        # transform the state to a usable one
        s_t = useful_from_observation(s_t)
        old_s_t = episode_states[-1][:4]
        s_t = np.concatenate((s_t, s_t-old_s_t))
        
        # working on the reward
        # r_t += -(s_t[0]**2 + s_t[1]**2)*100

        if ((s_t[:4]==np.array([0.05, 0.05, 0.05, 0.05])).all()):
            r_t = r_t - 1
            to_stop = to_stop - 1
        elif (s_t[0]==0.05 or s_t[1]==0.05):
            r_t = r_t - 0.5
        else:
            r_t += 0.2-(s_t[0]**2 + s_t[1]**2)*100

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

    H2 = 50

    for t in range(H):
        if H == 1000 and H - t < 100:
            break
        pi = get_policy(episode_states[t], theta)
        a_t = episode_actions[t]
        
        # R_t = sum(np.array(episode_rewards[t::])*np.array([GAMMA**i for i in range(0, H-t)]))
        R_t = sum(episode_rewards[t::t+H2])


        #if (a_t[0] == 1):
        #    g_theta_log_pi = - pi[1] * episode_states[t] * R_t
        #else:
        #    g_theta_log_pi = pi[0] * episode_states[t] * R_t
        
        g_theta_log_pi = pi * episode_states[t] * R_t
        PG += g_theta_log_pi

    return PG

# Train until average_return is larger than SCORE
def train(env, theta_init, max_episode_length = EPISODE_DURATION, alpha_init = ALPHA_INIT):

    theta = theta_init
    i_episode = 0
    average_returns = []

    success, _, R = test_policy(env, theta, render=True)
    average_returns.append(R)

    new_test_time = TEST_TIME

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

        # Test new policy 
        # new_test_time += TEST_TIME_AUGM
        success, _, R = test_policy(env, theta, score=SCORE, num_episodes=new_test_time, max_episode_length= max_episode_length, render=True)

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
    theta_init = np.random.randn(1, 8)

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