import gym
from gym.envs.registration import register

from constants import default_params_for_multiple_victims


register(
    id='PredatorMultipleVictims-v0',
    entry_point='PredatorMultipleVictims:PredatorMultipleVictims',
)
env = gym.make('PredatorMultipleVictims-v0', params=default_params_for_multiple_victims)

print(env.reset())
for i in range(5):
    print(env.step(
        {
            "predator": [1, 1, 1, 1],
            "victims": [1, 1, 1, 1]
        }
    ))
