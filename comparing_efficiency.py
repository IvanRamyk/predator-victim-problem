import cv2
import numpy as np
import os
import pickle
import gym
import ray
import ray.rllib.agents.a3c as a3c
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.tune.registry import register_env
import tensorflow as tf
from PredatorVictim.PredatorMultipleVictims import PredatorMultipleVictims
from PredatorVictim.constants import default_params_for_multiple_victims
from evaluate import evaluate


def victim_policy_go_from_predator(observation):
    p_x, p_y = observation[0], observation[1]
    v1_x, v1_y = observation[4], observation[5]
    v2_x, v2_y = observation[9], observation[10]
    return np.array([v1_x - p_x, v1_y - p_y, v2_x - p_x, v2_y - p_y])


def predator_policy_go_to_victim(observation):
    p_x, p_y = observation[0], observation[1]
    v1_x, v1_y, v1_a = observation[4], observation[5], observation[8]
    v2_x, v2_y, v2_a = observation[9], observation[10], observation[13]
    if v1_a > 0.1:
        return np.array([v1_x - p_x, v1_y - p_y, 0, 0])
    return np.array([v2_x - p_x, v2_y - p_y, 0, 0])


def victim_custom(trainer, PVEnv, n_iter=10, video_file=None):
    if video_file is not None:
        video = cv2.VideoWriter("../videos/Predator_Victim.avi", 0, 60, (PVEnv.screen_wh, PVEnv.screen_wh))
    for i in range(n_iter):
        obs = PVEnv.reset()
        done = False
        while not done:
            # action_predator = trainer.compute_action(obs['predator'], policy_id="policy_predator", explore=False)
            action_victim = trainer.compute_action(obs['victims'], policy_id="policy_victims", explore=False)
            # print(action_predator)
            action_predator = predator_policy_go_to_victim(obs["predator"])
            # print(action_victim)
            obs, rewards, dones, info = PVEnv.step({"predator": action_predator, "victims": action_victim})
            done = dones['__all__']
            frame = PVEnv.render(mode='rgb_array')
            if video_file is not None:
                video.write(frame[..., ::-1])
        PVEnv.close()
    if video_file is not None:
        video.release()


def evaluate_length(trainer, PVEnv, file, n_iter=100):
    f = open(file, "w")
    for i in range(n_iter):
        obs = PVEnv.reset()
        done = False
        step = 0
        while not done:
            action_predator = trainer.compute_action(obs['predator'], policy_id="policy_predator", explore=False)
            # action_predator = predator_policy_go_to_victim(obs["predator"])
            # action_victim = trainer.compute_action(obs['victims'], policy_id="policy_victims", explore=False)
            action_victim = victim_policy_go_from_predator(obs['victims'])
            obs, rewards, dones, info = PVEnv.step({"predator": action_predator, "victims": action_victim})
            done = dones['__all__']
            step += 1
        PVEnv.close()
        print(i, step)
        f.write("{} {}\n".format(i, step))

    f.close()


ready_to_exit = False


def press_key_exit(_q):
    global ready_to_exit
    ready_to_exit = True

def gen_policy(PVEnv, i):
    ModelCatalog.register_custom_model("PredatorVictimModel_{}".format(i), PredatorVictimModel)
    config = {
        "model": {"custom_model": "PredatorVictimModel_{}".format(i)},
    }
    print(PVEnv.action_space, PVEnv.observation_space)
    return None, PVEnv.observation_space, PVEnv.action_space, config
    # return None, PVEnv.observation_space, PVEnv.action_space, {}


def policy_mapping_fn(agent_id):
    if agent_id == 'predator':
        return "policy_predator"
    elif agent_id == 'victims':
        return "policy_victims"


if __name__ == "__main__":
    model_file_to_read = 'PredatorVictim_action_dist-v3.pickle'
    model_file_to_read = 'PredatorVictim_action_dist-v3-from-policy.pickle'

    # ray.init(include_dashboard=False)
    register_env("PredatorMultipleVictims-v0", lambda c: PredatorMultipleVictims(params=default_params_for_multiple_victims))
    PVEnv = gym.make("PredatorMultipleVictims-v0", params=default_params_for_multiple_victims)

    trainer = a3c.A3CTrainer(
        env="PredatorMultipleVictims-v0",
        config={
            "multiagent": {
                "policies": {
                    "policy_predator": gen_policy(PVEnv, 0),
                    "policy_victims": gen_policy(PVEnv, 1)
                },
                "policy_mapping_fn": policy_mapping_fn,
            },
        }
    )

    if os.path.isfile(model_file_to_read):
        weights = pickle.load(open(model_file_to_read, "rb"))
        trainer.restore_from_object(weights)
        print(trainer)
        print("model restored!")

    # victim_custom(trainer, PVEnv, n_iter=5)
    evaluate_length(trainer, PVEnv, ".txt")
