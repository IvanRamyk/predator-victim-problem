import gym
from ray.rllib.agents.ppo import ppo
from ray.tune import register_env

from PredatorVictim.PredatorVictim import PredatorVictim
from PredatorVictim.constants import default_params

register_env("PredatorVictim-v0", lambda c: PredatorVictim(params=default_params))
PVEnv = gym.make("PredatorVictim-v0", params=default_params)


def gen_policy(PVEnv, i):
    return None, PVEnv.observation_space, PVEnv.action_space, {}


def policy_mapping_fn(agent_id):
    if agent_id == 'predator':
        return "policy_predator"
    elif agent_id == 'victim':
        return "policy_victim"

trainer = ppo.PPOTrainer(
    env='PredatorVictim-v0',
    config={
        "multiagent": {
            "policies": {"policy_predator": gen_policy(PVEnv, 0),
                         "policy_victim": gen_policy(PVEnv, 1)},
            "policy_mapping_fn": policy_mapping_fn,
        },
    }
)
while True:
    print("Training", trainer.step())
