import gym
import numpy as np
import pickle as pkl

# Creates the environment
cliffEnv = gym.make("CliffWalking-v0")

# Initializing Q Table
q_table = np.zeros(shape=(48, 4))


# epsilon-greedy policy
def policy(state, explore=0.0):
    action = int(np.argmax(q_table[state]))
    if np.random.random() <= explore:
        action = int(np.random.randint(low=0, high=4, size=1))
    return action


# Parameters
EPSILON = 0.1
ALPHA = 0.1
GAMMA = 0.9
NUM_EPISODES = 500

# Training for 500 episodes
for episode in range(NUM_EPISODES):

    # Initializing episode
    done = False
    total_reward = 0
    episode_length = 0

    state = cliffEnv.reset()

    # Selecting an action according to our policy
    action = policy(state, EPSILON)

    # For every step of the episode
    while not done:
        # Take an action in the environment
        next_state, reward, done, _ = cliffEnv.step(action)

        # Select the next action
        next_action = policy(next_state, EPSILON)

        # SARSA update
        q_table[state][action] += ALPHA * (reward + GAMMA * q_table[next_state][next_action] - q_table[state][action])

        state = next_state
        action = next_action

        total_reward += reward
        episode_length += 1
    print("Episode:", episode, "Episode Length:", episode_length, "Total Reward: ", total_reward)
cliffEnv.close()

pkl.dump(q_table, open("sarsa_q_table.pkl", "wb"))
print("Training Complete. Q Table Saved")
