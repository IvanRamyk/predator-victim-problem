import gym
from ray.rllib.agents import a3c
from ray.tune import register_env

from PredatorVictim.PredatorMultipleVictims import PredatorMultipleVictims
from PredatorVictim.constants import default_params_for_multiple_victims


def test_without_custom_model():
    register_env("PredatorMultipleVictims-v0", lambda c: PredatorMultipleVictims(params=default_params_for_multiple_victims))
    PVEnv = gym.make("PredatorMultipleVictims-v0", params=default_params_for_multiple_victims)

    print(PVEnv.action_space.sample())

    def gen_policy(PVEnv, i):
        return None, PVEnv.observation_space, PVEnv.action_space, {}

    def policy_mapping_fn(agent_id):
        if agent_id == 'predator':
            return "policy_predator"
        elif agent_id == 'victims':
            return "policy_victim"

    trainer = a3c.A3CTrainer(
        env='PredatorMultipleVictims-v0',
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


if __name__ == "__main__":
    test_without_custom_model()
