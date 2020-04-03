import gym
import keras.models as keras_model
import keras.layers as layer
from keras.optimizers import Adam
import numpy as np
import random


class MountainCarDQL():
    def __init__(self):
        self.env = gym.make('MountainCar-v0')
        self.gamma = 0.95
        self.learning_rate = 0.1
        self.epsilon = 1.0
        self.min_epsilon = 0.01
        self.epsilon_decay = 0.001
        self.model = self.create_model()
        self.memory = []
        self.target_model = self.create_model()
        self.tau = 0.125

    def create_model(self):
        model = keras_model.Sequential()
        input_shape = self.env.observation_space.shape
        model.add(layer.Dense(units=32, input_shape=input_shape, activation='relu'))
        model.add(layer.Dense(units=32, activation='relu'))
        model.add(layer.Dense(units=self.env.action_space.n))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def get_action(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.min_epsilon, self.epsilon)
        if np.random.uniform(0, 1, size=1)[0] < self.epsilon:
            action = np.random.randint(low=0, high=3, size=1)[0]
        else:
            # print(self.model.predict(state))
            action = np.argmax(self.model.predict(state)[0])
        return action

    def replay(self):
        batch_size = 128
        if len(self.memory) < batch_size:
            return
        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target_train = self.target_model.predict(state)
            if done:
                target_train = reward
            else:
                target_train[0][action] = reward + self.learning_rate * (
                        reward + self.gamma * np.argmax(self.model.predict(new_state)[0]))

            self.model.fit(state, target_train, epochs=1, verbose=0)
            # if done:
            #     target[0][action] = reward
            # else:
            #     Q_future = max(self.target_model.predict(new_state)[0])
            #     target[0][action] = reward + Q_future * self.gamma
            #     self.model.fit(state, target, epochs=1, verbose=0)

    def target_model(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self):
        self.model.save('MountanCarBot.h5')
