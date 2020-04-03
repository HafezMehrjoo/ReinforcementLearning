import gym
import random


# [position of cart, velocity of cart, angle of pole, rotation rate of pole]

def model_data_preparation():
    env = gym.make('CartPole-v1')
    env.reset()
    goal_steps = 500
    score_requirement = 60
    initial_games = 4000
    training_data = []
    accepted_score = []
    for game_index in range(initial_games):
        score = 0
        game_memory = []
        previous_observation = []
        for step_index in range(goal_steps):
            action = random.randrange(0, 2)
            observation, reward, info, done = env.step(action)
            if len(previous_observation) > 0:
                game_memory.append([observation, action])
            previous_observation = observation
            score += reward
            if done:
                break
        output = None
        if score >= score_requirement:
            accepted_score.append(score)
            for data in game_memory:
                if data[1] == 0:
                    output = [0, 1]
                elif data[1] == 1:
                    output = [1, 0]
                training_data.append([data[0], output])

        env.reset()
    print(accepted_score)
    return training_data
