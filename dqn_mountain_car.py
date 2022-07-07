import gym
import pandas as pd
import tensorflow as tf
from keras import Model, Input
from keras.models import clone_model
from keras.layers import Dense
from keras.losses import Huber

env = gym.make("MountainCar-v0")

# Online Network
net_input = Input(shape=(2,))
x = Dense(64, activation="relu")(net_input)
x = Dense(32, activation="relu")(x)
output = Dense(3, activation="linear")(x)
q_net = Model(inputs=net_input, outputs=output)
q_net.compile(optimizer="adam")
loss_fn = Huber()

# Target Network
target_net = clone_model(q_net)

# Parameters
EPSILON = 1.0
EPSILON_DECAY = 1.005
GAMMA = 0.99
MAX_TRANSITIONS = 1_00_000
NUM_EPISODES = 1500
BATCH_SIZE = 64
TARGET_UPDATE_AFTER = 1000
LEARN_AFTER_STEPS = 3

REPLAY_BUFFER = []


def insert_transition(transition):
    if len(REPLAY_BUFFER) >= MAX_TRANSITIONS:
        REPLAY_BUFFER.pop(0)
    REPLAY_BUFFER.append(transition)


def sample_transitions(batch_size=16):
    random_indices = tf.random.uniform(shape=(batch_size,), minval=0, maxval=len(REPLAY_BUFFER), dtype=tf.int32)
    sampled_current_states = []
    sampled_actions = []
    sampled_rewards = []
    sampled_next_states = []
    sampled_terminals = []

    for index in random_indices:
        sampled_current_states.append(REPLAY_BUFFER[index][0])
        sampled_actions.append(REPLAY_BUFFER[index][1])
        sampled_rewards.append(REPLAY_BUFFER[index][2])
        sampled_next_states.append(REPLAY_BUFFER[index][3])
        sampled_terminals.append(REPLAY_BUFFER[index][4])

    return tf.convert_to_tensor(sampled_current_states), tf.convert_to_tensor(sampled_actions), tf.convert_to_tensor(
        sampled_rewards), tf.convert_to_tensor(sampled_next_states), tf.convert_to_tensor(sampled_terminals)


def policy(state, explore=0.0):
    action = tf.argmax(q_net(tf.expand_dims(state, axis=0))[0], output_type=tf.int32)
    if tf.random.uniform(shape=(), maxval=1) <= explore:
        action = tf.random.uniform(shape=(), maxval=3, dtype=tf.int32)
    return action


random_states = []
done = False
i = 0
state = env.reset()
while i < 20 and not done:
    random_states.append(state)
    state, _, done, _ = env.step(policy(state).numpy())
    i += 1

random_states = tf.convert_to_tensor(random_states)


def get_q_values(states):
    return tf.reduce_max(q_net(states), axis=1)


step_counter = 0
metric = {"episode": [], "length": [], "total_reward": [], "avg_q": [], "exploration": []}
for episode in range(NUM_EPISODES):
    done = False
    total_rewards = 0
    episode_length = 0
    state = env.reset()
    while not done:
        action = policy(state, EPSILON)
        next_state, reward, done, _ = env.step(action.numpy())
        insert_transition([state, action, reward, next_state, done])
        state = next_state
        step_counter += 1

        if step_counter % LEARN_AFTER_STEPS == 0:
            current_states, actions, rewards, next_states, terminals = sample_transitions(BATCH_SIZE)

            next_action_values = tf.reduce_max(target_net(next_states), axis=1)
            targets = tf.where(terminals, rewards, rewards + GAMMA * next_action_values)

            with tf.GradientTape() as tape:
                preds = q_net(current_states)
                batch_nums = tf.range(0, limit=BATCH_SIZE)
                indices = tf.stack((batch_nums, actions), axis=1)
                current_values = tf.gather_nd(preds, indices)
                loss = loss_fn(targets, current_values)
            grads = tape.gradient(loss, q_net.trainable_weights)
            q_net.optimizer.apply_gradients(zip(grads, q_net.trainable_weights))

        if step_counter % TARGET_UPDATE_AFTER == 0:
            target_net.set_weights(q_net.get_weights())

        total_rewards += reward
        episode_length += 1

    avg_q = tf.reduce_mean(get_q_values(random_states)).numpy()
    metric["episode"].append(episode)
    metric["length"].append(episode_length)
    metric["total_reward"].append(total_rewards)
    metric["avg_q"].append(avg_q)
    metric["exploration"].append(EPSILON)
    EPSILON /= EPSILON_DECAY

    pd.DataFrame(metric).to_csv("mountain_metric.csv", index=False)
env.close()
q_net.save("dqn_mountain_q_net")
