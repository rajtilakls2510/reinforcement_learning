import gym, cv2
import tensorflow as tf
from keras.models import load_model

env = gym.make("MountainCar-v0")
q_net = load_model("dqn_mountain_q_net")


def policy(state, explore=0.0):
    action = tf.argmax(q_net(state)[0], output_type=tf.int32)
    if tf.random.uniform(shape=(), maxval=1) <= explore:
        action = tf.random.uniform(shape=(), minval=0, maxval=3, dtype=tf.int32)
    return action


for episode in range(5):
    done = False
    state = tf.convert_to_tensor([env.reset()])
    while not done:
        frame = env.render(mode="rgb_array")
        cv2.imshow("Mountain Car", frame)
        cv2.waitKey(20)
        action = policy(state)
        state, reward, done, _ = env.step(action.numpy())
        state = tf.convert_to_tensor([state])
env.close()
