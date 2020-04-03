import gym
import numpy as np
from keras.models import load_model

'''
actions    0 -> left
           1 -> netural
           2 -> right

state      velocity = (-0.07,0.07)
           position = (-1.2 , 6.0)     
           continue state
For every time step:
reward += -1          
'''


def mountain_car_bot():
    env = gym.make('MountainCar-v0')
    model = load_model('MountainCarBot00.h5')
    for _ in range(10):
        state = env.reset()
        for _ in range(100):
            env.render()
            state = np.array([[-1.14692759, -0.02637298]]).reshape(1, 2)
            action = np.argmax(model.predict(state)[0])
            new_state, reward, done, info = env.step(action=action)
            print(action)
            state = new_state
        print(_)


mountain_car_bot()
