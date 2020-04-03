import gym
import numpy as np
import random

'''
(S: starting point, safe) 
(F: frozen surface, safe)  
(H: hole, fall to your doom) 
(G: goal, where the frisbee is located).
'''


def q_table_generator():
    env = gym.make("FrozenLake-v0")
    action_size = env.action_space.n
    state_size = env.observation_space.n
    q_table = np.zeros(shape=(state_size, action_size))
    learning_rate = 0.1
    gamma = 0.95
    epsilon = 1.0
    max_steps = 100
    max_epsilon = 1.0
    min_epsilon = 0.01
    decay_rate = 0.01
    total_episodes = 10000
    rewards = []
    for episode in range(total_episodes):
        state = env.reset()
        done = False
        total_rewards = 0
        for step in range(max_steps):
            exp_tradaeoff = random.uniform(0, 0.8)
            if episode > 5000:
                action = np.argmax(q_table[state, :])
            else:
                action = np.random.randint(low=0, high=4, size=1)[0]
            new_state, reward, done, info = env.step(action=action)
            print(action)
            q_table[state, action] = q_table[state, action] + learning_rate * (
                    reward + gamma * (np.max(q_table[new_state, :]) - q_table[state, action]))
            state = new_state
            total_rewards += reward
            if done:
                break
        rewards.append(total_rewards)
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    print(q_table)
    return q_table
