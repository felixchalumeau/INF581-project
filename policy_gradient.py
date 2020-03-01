import numpy as np
import matplotlib.pyplot as plt
import gym

# TODO: explain following
ENV_NAME = "CartPole-v0"
EPISODE_DURATION = 300
ALPHA_INIT = 0.1 # (can also decay this over time..)
SCORE = 195.0
TEST_TIME = 100
LEFT = 0
RIGHT = 1

VERBOSE = True

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# Return policy
def get_policy(s, theta):

    p_right = sigmoid(np.dot(s, np.transpose(theta)))
    pi = [1-p_right, p_right]
    return pi

# Draw an action according to current policy
def act_with_policy(s, theta):
    p_right = get_policy(s, theta)[1]
    r = np.random.rand()
    if r < p_right:
        return 1
    else:
        return 0

# Generate an episode
def gen_rollout(env, theta, max_episode_length=EPISODE_DURATION, render=False):

    s_t = env.reset()
    episode_states = []
    episode_actions = []
    episode_rewards = []
    episode_states.append(s_t)

    for t in range(max_episode_length):

        if render:
            env.render()

        a_t = act_with_policy(s_t, theta)
        s_t, r_t, done, info = env.step(a_t)
        episode_states.append(s_t)
        episode_actions.append(a_t)
        episode_rewards.append(r_t)

        if done:
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

        if a_t == LEFT:
            g_theta_log_pi = - pi[RIGHT] * episode_states[t] * R_t
        else:
            g_theta_log_pi = pi[LEFT] * episode_states[t] * R_t
        
        PG += g_theta_log_pi

    return PG

# Train until average_return is larger than SCORE
def train(env, theta_init, max_episode_length = EPISODE_DURATION, alpha_init = ALPHA_INIT):

    theta = theta_init
    i_episode = 0
    average_returns = []

    success, _, R = test_policy(env, theta)
    average_returns.append(R)

    # Train until success
    while (not success):

        # Rollout
        episode_states, episode_actions, episode_rewards = gen_rollout(env, theta, max_episode_length)

        # Schedule step size
        alpha = alpha_init / (1 + i_episode)

        # Compute gradient
        PG = compute_PG(episode_states, episode_actions, episode_rewards, theta)

        # Do gradient ascent 
        theta += alpha * PG

        # Test new policy 
        success, _, R = test_policy(env, theta, render=False)

        # Monitor 
        average_returns.append(R)

        i_episode += 1

        if VERBOSE:
            print("Episode {0}, average return: {1}".format(i_episode, R))

    return theta, i_episode, average_returns


def main():

    env = gym.make(ENV_NAME)

    dim = env.observation_space.shape[0]

    # Init parameters to random
    theta_init = np.random.randn(1,dim)

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
