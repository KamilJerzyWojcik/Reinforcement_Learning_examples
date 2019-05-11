"""
Bot 2 -- Make a random, baseline agent for the SpaceInvaders game.
"""

import gym
import random

random.seed(0) # make result reproducible
num_episodes = 10

def main():
    env_space_invaders = gym.make('SpaceInvaders-v0')
    env_space_invaders.seed(0) # make result reproducible
    rewards = []

    for _ in range(num_episodes):
        env_space_invaders.reset()

        episode_reward = 0
        while True:
            action = env_space_invaders.action_space.sample()
            _, reward, done, _ = env_space_invaders.step(action) # random action
            episode_reward += reward
            if done:
                print('Reward: %s' % episode_reward)
                rewards.append(episode_reward)
                break
    print(f'Average reward: {(sum(rewards)/len(rewards))}')

if __name__ == '__main__':
    main()

