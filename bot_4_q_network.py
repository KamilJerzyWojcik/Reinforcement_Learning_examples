"""
Bot 3 -- Use Q-learning network to train bot
"""

from typing import List
import gym
import numpy as np
import random
import tensorflow as tf

random.seed(0) # make result reproducible
np.random.seed(0)
tf.set_random_seed(0)

num_episodes = 4000
discount_factor = 0.8
learning_rate = 0.9
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
    env = gym.make('FrozenLake-v0') # create game
    env.seed(0)
    rewards = []

    Q = np.zeros((env.observation_space.n, env.action_space.n)) # powierzchnia, l_akcji
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        episode_reward = 0

        while True:
            noise = np.random.random((1, env.action_space.n)) / (episode ** 2)
            action = np.argmax(Q[state, :] + noise)
            state2, reward, done, _ = env.step(action)
            Qtarget = reward + discount_factor * np.max(Q[state2, :])
            Q[state, action] = (1 - learning_rate) * Q[state, action] + learning_rate * Qtarget
            episode_reward += reward
            state = state2
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

