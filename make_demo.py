import datetime
import os
import pickle

import cv2
import gym
import numpy as np
import ray
from numpy import float32
from ray.rllib.agents.a3c import a3c
from ray.rllib.models import ModelCatalog
from ray.tune import register_env

from PredatorVictim.PredatorMultipleVictims import PredatorMultipleVictims
from PredatorVictim.constants import default_params_for_multiple_victims
from action_dist import VictimActionDist, PredatorActionDist, PredatorVictimModel


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


def episode_from_point(trainer, PVEnv, n_iter=1, video_file=f"demo-video-{datetime.date.today()}", point=None):
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter("Predator_Victim.avi", fourcc, 60, (PVEnv.screen_wh, PVEnv.screen_wh))
    for i in range(n_iter):
        obs = PVEnv.reset()
        if point is not None:
            PVEnv.entities["predator"]["pos"] = point["predator"]["pos"]
            PVEnv.entities["predator"]["vel"] = point["predator"]["vel"]
            PVEnv.entities["victims-1"]["pos"] = point["victims-1"]["pos"]
            PVEnv.entities["victims-1"]["vel"] = point["victims-1"]["vel"]
            PVEnv.entities["victims-0"]["pos"] = point["victims-0"]["pos"]
            PVEnv.entities["victims-0"]["vel"] = point["victims-0"]["vel"]
        done = False
        step = 0
        while not done:
            step += 1
            # action_predator = trainer.compute_action(obs['predator'], policy_id="policy_predator", explore=False)
            action_predator = predator_policy_go_to_victim(obs["predator"])
            action_victim = trainer.compute_action(obs['victims'], policy_id="policy_victims", explore=False)
            obs, rewards, dones, info = PVEnv.step({"predator": action_predator, "victims": action_victim})
            done = dones['__all__']
            frame = PVEnv.render(mode='rgb_array')
            video.write(frame[..., ::-1])
        PVEnv.close()
        print(step)
    video.release()


def episode_from_point_for_policy_from_file(
    trainer_file, point=None
):
    def gen_policy(PVEnv, i):
        ModelCatalog.register_custom_model("PredatorVictimModel_{}".format(i), PredatorVictimModel)
        config = {
            "model": {
                "custom_model": "PredatorVictimModel_{}".format(i),
            },
        }
        if i == 1:
            ModelCatalog.register_custom_action_dist("VictimActionDist".format(i), VictimActionDist)
            config["model"]["custom_action_dist"] = "VictimActionDist"
        else:
            ModelCatalog.register_custom_action_dist("PredatorActionDist".format(i), PredatorActionDist)
            config["model"]["custom_action_dist"] = "PredatorActionDist"
        print(config)
        return None, PVEnv.observation_space, PVEnv.action_space, config

    def policy_mapping_fn(agent_id):
        if agent_id == 'predator':
            return "policy_predator"
        elif agent_id == 'victims':
            return "policy_victims"


    ray.init(include_dashboard=True)
    register_env("PredatorMultipleVictims-v0",
                 lambda c: PredatorMultipleVictims(params=default_params_for_multiple_victims))
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

    if os.path.isfile(trainer_file):
        weights = pickle.load(open(trainer_file, "rb"))
        trainer.restore_from_object(weights)
        print(trainer)
        print("model restored!")

    episode_from_point(
        trainer,
        PVEnv,
        point=point
    )


if __name__ == "__main__":
    episode_from_point_for_policy_from_file(
        "PredatorVictim_action_dist-v3-demo-1.pickle",
        point={
            "predator": {
                "pos": np.array([-0, -0], dtype=float32),
                "vel": np.array([0, 0], dtype=float32),
            },
            "victims-0": {
                "pos": np.array([-0.25, 0.03], dtype=float32),
                "vel": np.array([-0.005, -0.001], dtype=float32),
            },
            "victims-1": {
                "pos": np.array([0.25, -0], dtype=float32),
                "vel": np.array([0.005, 0.001], dtype=float32),
            }
        }
    )