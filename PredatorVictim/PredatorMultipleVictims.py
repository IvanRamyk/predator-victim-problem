from math import sqrt

from gym import spaces
import numpy as np
import gym
from gym.utils import seeding
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class PredatorMultipleVictims(MultiAgentEnv, gym.Env):

    metadata = {"render.modes": ["human"]}

    @staticmethod
    def _get_observation_space_size(n_victims: int) -> int:
        """Returns a number of the coordinates in the observation space"""
        return 4 + n_victims * 5

    @staticmethod
    def _get_action_space_size(n_victims: int) -> int:
        """Returns a number of the coordinates in the action space"""
        return 2 * n_victims

    def __init__(self, **kwargs):
        super().__init__()
        self.params = kwargs.get("params")
        self.agents = ["predator", "victims"]
        self._agent_ids = ["predator", "victims"]
        self._n_victims = self.params["n_victims"]
        observation_space_size = self._get_observation_space_size(self._n_victims)
        self.observation_space = spaces.Box(
            -np.ones(observation_space_size),
            np.ones(observation_space_size),
            shape=(observation_space_size,),
        )
        action_space_size = self._get_action_space_size(self._n_victims)
        self.action_space = spaces.Box(low=-1, high=1, shape=(action_space_size,))
        self.entities = dict()
        self.n_steps = 0
        self.seed()
        self.viewer = None
        self.screen_wh = 600

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def create_entity(self, name, color):
        res = dict()
        entity_class = self._get_agent_class_by_entity_name(name)
        res["pos"] = 2 * self.np_random.rand(2) - 1
        res["vel"] = 2 * self.np_random.rand(2) - 1
        res["vel"] /= np.linalg.norm(res["vel"])
        res["vel"] *= self.np_random.rand() * self.params[entity_class]["max_vel"]
        res["color"] = color
        res["alive"] = True
        return res

    def make_obs(self):
        observation = np.array([])
        observation = np.append(observation, self.entities["predator"]["pos"])
        observation = np.append(observation, self.entities["predator"]["vel"] / self.params["predator"]["max_vel"])

        for i in range(self._n_victims):
            observation = np.append(observation, self.entities[f"victims-{i}"]["pos"])
            observation = np.append(observation, self.entities[f"victims-{i}"]["vel"] / self.params["victims"]["max_vel"])
            observation = np.append(observation, [1 if self.entities[f"victims-{i}"]["alive"] else 0])

        return observation

    def reset(self):
        self.n_steps = 0
        self.entities["predator"] = self.create_entity("predator", (1, 0, 0))
        for i in range(self._n_victims):
            self.entities[f"victims-{i}"] = self.create_entity(f"victims-{i}", (0, 1, 0))

        observation = self.make_obs()
        obs = {"predator": observation, "victims": observation}
        return obs

    @staticmethod
    def action_to_vector_continuous(action):
        if np.isnan(action).any():
            action = np.zeros(
                PredatorMultipleVictims._get_action_space_size(2)
            )
        if np.linalg.norm(action) > 1:
            return action / np.linalg.norm(action)
        else:
            return action

    @staticmethod
    def action_to_vector_discrete(action):
        if action == 0:
            return np.array([1, 0])
        elif action == 1:
            return np.array([-1, 0])
        elif action == 2:
            return np.array([0, 1])
        elif action == 3:
            return np.array([0, -1])

    def step(self, action_dict):
        done = False
        self.n_steps += 1
        rewards = {"predator": 0, "victims": 0}
        for entity_name in self.entities:
            act = None
            entity_class = self._get_agent_class_by_entity_name(entity_name)
            if self._is_agent_victim(entity_name):
                victim_number = self._get_victim_number_by_name(entity_name)
                act = self.action_to_vector_continuous(action_dict["victims"][victim_number * 2:victim_number * 2 + 2])
            else:
                act = self.action_to_vector_continuous(action_dict[entity_name][:2])
            self.entities[entity_name]["vel"] += self.params[entity_class]["max_acceleration"] * act
            vel_abs = np.linalg.norm(self.entities[entity_name]["vel"])
            if vel_abs > self.params[entity_class]["max_vel"]:
                self.entities[entity_name]["vel"] /= vel_abs
                self.entities[entity_name]["vel"] *= self.params[entity_class]["max_vel"]
            # THIS LINE SHOULDN'T BE IN PROD
            # if entity_name == "predator":
            if self.entities[entity_name]["alive"]:
                self.entities[entity_name]["pos"] += self.entities[entity_name]["vel"]
            for i in range(2):
                if (
                    self.entities[entity_name]["pos"][i] > 1
                    or self.entities[entity_name]["pos"][i] < -1
                ):
                    self.entities[entity_name]["pos"][i] = np.sign(self.entities[entity_name]["pos"][i])
                    self.entities[entity_name]["vel"][i] *= -1

        done = True
        for victim_i in range(self._n_victims):
            victim_name = f"victims-{victim_i}"
            if self.entities[victim_name]["alive"]:
                dist_between = np.linalg.norm(
                    self.entities["predator"]["pos"] - self.entities[f"victims-{victim_i}"]["pos"]
                )
                rewards["victims"] = sqrt(dist_between) * self.params["reward_scale"]
                rewards["predator"] = -sqrt(dist_between) * self.params["reward_scale"]

                if dist_between < self.params["catch_distance"]:
                    rewards["predator"] += 1500 / self.n_steps
                    rewards["victims"] -= 1500 / self.n_steps
                    self.entities[victim_name]["alive"] = False
                    self.entities[victim_name]["color"] = (1, 1, 1)

                else:
                    done = False

        if self.n_steps > self.params["max_steps"]:
            done = True

        observation = self.make_obs()
        obs = {"predator": observation, "victims": observation}
        dones = {"__all__": done}

        return obs, rewards, dones, {}

    def render(self, mode="human"):
        radius = self.params["catch_distance"] * self.screen_wh / 4
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(self.screen_wh, self.screen_wh)
            for key in self.entities:
                self.entities[key]["trans"] = rendering.Transform()
                self.entities[key]["geom"] = rendering.make_circle(radius=radius)
                self.entities[key]["geom"].add_attr(self.entities[key]["trans"])
                self.entities[key]["geom"].set_color(*self.entities[key]["color"])
                if self.entities[key]["alive"]:
                    self.viewer.add_geom(self.entities[key]["geom"])
                else:
                    self.entities[key]["geom"].set_color(1, 1, 1)
                    self.viewer.add_geom(self.entities[key]["geom"])
        for key in self.entities:
            pos = (self.entities[key]["pos"] + 1) * self.screen_wh / 2
            self.entities[key]["trans"].set_translation(pos[0], pos[1])
        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def action_space_sample(self, agent_ids: list = None):
        return {
            "predator": self.action_space.sample(),
            "victims": self.action_space.sample(),
        }

    def observation_space_sample(self, agent_ids: list = None):
        return {
            "predator": self.observation_space.sample(),
            "victims": self.observation_space.sample(),
        }

    @staticmethod
    def _get_agent_class_by_entity_name(name) -> str:
        return name.split("-")[0]

    @staticmethod
    def _is_agent_victim(name) -> bool:
        return PredatorMultipleVictims._get_agent_class_by_entity_name(name) == "victims"

    @staticmethod
    def _get_victim_number_by_name(name) -> int:
        return int(name.split("-")[1])