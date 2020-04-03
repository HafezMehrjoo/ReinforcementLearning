from MountainCar import MountainCarDQL
from MountainCar import MountainCarBot


def main():
    mountain_car_dql = MountainCarDQL.MountainCarDQL()
    episode = 100
    steps = 100
    for _ in range(episode):
        state = mountain_car_dql.env.reset().reshape(1, 2)
        for step in range(steps):
            action = mountain_car_dql.get_action(state)
            new_state, reward, done, info = mountain_car_dql.env.step(action=action)
            new_state = new_state.reshape(1, 2)
            mountain_car_dql.remember(state=state, action=action, reward=reward, new_state=new_state, done=done)
            mountain_car_dql.replay()
            state = new_state

    mountain_car_dql.save_model()
    MountainCarBot.mountain_car_bot()


if __name__ == '__main__':
    main()
