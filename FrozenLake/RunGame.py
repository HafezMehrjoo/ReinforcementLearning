import gym
import numpy as np


def run_game(q_table):
    env = gym.make("FrozenLake-v0")
    for episode in range(1):
        state = env.reset()
        steps = 100
        done = False
        for step in range(steps):
            env.render()
            new_state, reward, done, info = env.step(action=np.argmax(q_table[state, :]))
            if done:
                print('step', step + 1)
                print('episode', episode + 1)
                env.render()
                break
            state = new_state
    env.close()
