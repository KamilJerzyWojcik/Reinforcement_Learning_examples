"""
Bot 3 -- Use Q-learning network to train bot
"""

from typing import List
import gym
import numpy as np
import random
import tensorflow as tf

random.seed(0)  # make result reproducible
np.random.seed(0)
tf.set_random_seed(0)

num_episodes = 4000
discount_factor = 0.99
learning_rate = 0.15
report_interval = 500
exploration_probability = lambda episode: 50. / (episode + 10)
report = '100-ep Average: %.2f . Best 100-ep Average: %.2f . Average: ' \
         '%.2f (Episode %d)'


def one_hot(i: int, n: int) -> np.array:
    """Implements one-hot encoding by selecting the ith standard basis vector"""
    return np.identity(n)[i].reshape((1, -1))


def print_report(rewards: List, episode: int):
    """Print rewards for current episode
    - Average for last 100 episodes
    - Best 100-episode average across all time
    - Average for all episodes across time
    """
    print(report % (np.mean(rewards[-100:]),
                    max([np.mean(rewards[i:i+100]) for i in range(len(rewards) - 100)]),
                    np.mean(rewards),
                    episode
                    ))


def main():
    env = gym.make('FrozenLake-v0')
    env.seed(0)
    rewards = []

    # 1. Setup Placeholders
    n_observations, n_actions = env.observation_space.n, env.action_space.n
    observations_time_ph = tf.placeholder(shape=[1, n_observations], dtype=tf.float32)
    observations_time_plus_1_ph = tf.placeholder(shape=[1, n_observations], dtype=tf.float32)
    actions_ph = tf.placeholder(shape=(), dtype=tf.int32)
    rewards_ph = tf.placeholder(shape=(), dtype=tf.float32)
    q_target_ph = tf.placeholder(shape=[1, n_actions], dtype=tf.float32)

    # 2. Setup computation graph
    W = tf.Variable(tf.random_uniform([n_observations, n_actions], 0, 0.01))
    q_current = tf.matmul(observations_time_ph, W)
    q_target = tf.matmul(observations_time_plus_1_ph, W)

    q_target_max = tf.reduce_max(q_target_ph, axis=1)
    q_target_sa = rewards_ph + discount_factor * q_target_max
    q_current_sa = q_current[0, actions_ph]
    error = tf.reduce_sum(tf.square(q_target_sa - q_current_sa))
    prediction_action_ph = tf.argmax(q_current, 1)

    # 3. Setup optimization
    trainer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    update_model = trainer.minimize(error)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        for episode in range(1, num_episodes + 1):
            observations_time = env.reset()
            episode_reward = 0

            while True:
                # 4. Take step using best action or random action
                observations_time_one_hot = one_hot(observations_time, n_observations)
                action = session.run(prediction_action_ph, feed_dict={
                            observations_time_ph: observations_time_one_hot
                        })[0]
                if np.random.rand(1) < exploration_probability(episode):
                    action = env.action_space.sample()
                observations_time_plus_1, reward, done, _ = env.step(action)

                # 5. Train model
                observations_time_plus_1_one_hot = one_hot(observations_time_plus_1,
                                                           n_observations)
                q_target_value = session.run(q_target, feed_dict={
                    observations_time_plus_1_ph: observations_time_plus_1_one_hot
                })

                session.run(update_model, feed_dict={
                    observations_time_ph: observations_time_one_hot,
                    rewards_ph: reward,
                    q_target_ph: q_target_value,
                    actions_ph: action
                })
                episode_reward += reward
                observations_time = observations_time_plus_1
                if episode == 4000:
                    print(env.render())
                    input()
                if done:
                    rewards.append(episode_reward)
                    if episode % report_interval == 0:
                        print_report(rewards, episode)
                    break
        print_report(rewards, -1)


if __name__ == '__main__':
    main()

