import gym
import random
import numpy as np


#
# def test_bot(model):
#     env = gym.make('CartPole-v1')
#     env.reset()
#     previous_observation = 0
#     for i in range(1000):
#         env.render()
#         if previous_observation > 0:
#             observation, reward, info, done = env.step(action=model.predict(previous_observation))
#             previous_observation = observation
#         else:
#             env.step(random.randrange(0, 2))
def test_bot(model):
    env = gym.make('CartPole-v1')
    env.reset()
    for each_game in range(20000):
        prev_obs = []
        env.render()
        if len(prev_obs) == 0:
            action = random.randrange(0, 2)
        else:
            action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs)))[0])
        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        env.reset()
        if done:
            break
