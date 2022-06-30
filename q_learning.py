import gym
import numpy as np
import pickle as pkl

# Creates an environment
cliffEnv = gym.make("CliffWalking-v0")

# Initializing Q table
q_table = np.zeros(shape=(48, 4))

# Parameters
EPSILON = 0.1
ALPHA = 0.1
GAMMA = 0.9
NUM_EPISODES = 500


# epsilon-greedy policy
def policy(state, explore=0.0):
    action = int(np.argmax(q_table[state]))
    if np.random.random() <= explore:
        action = int(np.random.randint(low=0, high=4, size=1))
    return action


# Training for 500 episodes
for episode in range(NUM_EPISODES):

    # Initializing episode
    done = False
    total_reward = 0
    episode_length = 0

    state = cliffEnv.reset()

    # For every step of the episode
    while not done:
        # Select an action according to policy
        action = policy(state, EPSILON)

        # Take the action in the environment
        next_state, reward, done, _ = cliffEnv.step(action)

        # Select the optimal action of the next state
        next_action = policy(next_state)

        # Q-Learning update
        q_table[state][action] += ALPHA * (reward + GAMMA * q_table[next_state][next_action] - q_table[state][action])

        state = next_state

        total_reward += reward
        episode_length += 1
    print("Episode:", episode, "Length:", episode_length, "Total Reward: ", total_reward)
cliffEnv.close()
pkl.dump(q_table, open("q_learning_q_table.pkl", "wb"))
