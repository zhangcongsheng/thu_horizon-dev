import numpy as np

import environment  # noqa: F401
import gym

env = gym.make("HorizonCrossing-v0", training_task2=("EE6", "ES8"), num_future_data=0)

def test_crossing():

    # env = HorizonCrossingEnv(training_task2=['EN4', 'EN3', 'EE2'], num_future_data=0)

    obs = env.reset()
    i = 0
    while i < 1:
        for j in range(100):
            i += 1
            # action=2*np.random.random(2)-1
            action = env.action_space.sample()
            action = np.array([0.0, 0.3])
            obs, reward, done, info = env.step(action)

            # obses, actions = obs[np.newaxis, :], action[np.newaxis, :]
            # # extract infos for each kind of participants
            # start = 0
            # end = (
            #     start
            #     + env.ego_info_dim
            #     + env.per_tracking_info_dim * (env.num_future_data + 1)
            # )
            # obses_ego = obses[:, start:end]
            # print(obses_ego)
            # env.render()

            if done:
                break
    env.close()


if __name__ == "__main__":
    test_crossing()
