import gym
from gym.envs.registration import register

from constants import default_params


register(
    id='PredatorVictim-v0',
    entry_point='PredatorVictim:PredatorVictim',
)
env = gym.make('PredatorVictim-v0', params=default_params)

print(env.observation_space.sample())

print(env.reset())
for i in range(1000):
    print(env.step(
        {
            "predator": (1, 1),
            "victim": (1, 1),
        }
    )[0])