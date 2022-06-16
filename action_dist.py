import random
from typing import List, Union

import gym
import numpy as np
from ray.rllib.models import ActionDistribution
from ray.rllib.models.tf import TFModelV2
from ray.rllib.models.tf.tf_action_dist import TFActionDistribution
from ray.rllib.utils.typing import ModelConfigDict
from torch import TensorType
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



class PredatorActionDist(TFActionDistribution):

    def _build_sample_op(self) -> TensorType:
        return self.mean + self.std * tf.random.normal(tf.shape(self.mean))

    @staticmethod
    def predator_policy_go_to_victim(observation):
        p_x, p_y = observation[0], observation[1]
        v1_x, v1_y, v1_a = observation[4], observation[5], observation[8]
        v2_x, v2_y, v2_a = observation[9], observation[10], observation[13]
        # if v1_a > 0.1:
        return np.array([v1_x - p_x, v1_y - p_y, 0, 0])
        # return np.array([v2_x - p_x, v2_y - p_y, 0, 0])

    def _go_to_victim(self):
        dx1 = tf.reshape(self.observation[0][4] - self.observation[0][0], [1, 1])
        dy1 = tf.reshape(self.observation[0][5] - self.observation[0][1], [1, 1])
        dx2 = tf.reshape(self.observation[0][9] - self.observation[0][0], [1, 1])
        dy2 = tf.reshape(self.observation[0][10] - self.observation[0][1], [1, 1])
        tn = tf.multiply(tf.concat([dx1, dy1, dx2, dy2], 1), 100)  # vector v-p
        return tf.reshape(tn, [1, 4])

    def _go_from_victim(self):
        return -self._go_to_victim()

    def __init__(self, inputs: List[TensorType], model: TFModelV2):
        mean, log_std, observations = tf.split(inputs, tf.constant([4, 4, 14]), axis=1)
        self.mean = mean
        self.log_std = log_std
        self.std = tf.exp(log_std)
        self.observation = observations
        super().__init__(inputs, model)

    def _get_uniform_dist_value(self):
        random.seed()
        return random.random()

    def deterministic_sample(self) -> TensorType:
        prob = self._get_uniform_dist_value()
        if prob < 0:
            return self._go_to_victim()
        elif prob < 0:
            return self._go_from_victim()
        return self.mean

    def sample(self):
        prob = self._get_uniform_dist_value()
        if prob < 0:
            return self._go_to_victim()
        elif prob < 0.0:
           return self._go_from_victim()
        return self._build_sample_op()

    def logp(self, x: TensorType) -> TensorType:
        return -0.5 * tf.reduce_sum(
            tf.math.square((tf.cast(x, tf.float32) - self.mean) / self.std),
            axis=1
        ) - 0.5 * np.log(2.0 * np.pi) * tf.cast(tf.shape(x)[1], tf.float32) - \
            tf.reduce_sum(self.log_std, axis=1)

    def kl(self, other: ActionDistribution) -> TensorType:
        assert isinstance(other, VictimActionDist)
        return tf.reduce_sum(
            other.log_std - self.log_std +
            (tf.math.square(self.std) + tf.math.square(self.mean - other.mean))
            / (2.0 * tf.math.square(other.std)) - 0.5,
            axis=1)

    def entropy(self) -> TensorType:
        return tf.reduce_sum(
            self.log_std + .5 * np.log(2.0 * np.pi * np.e), axis=1)

    @staticmethod
    def required_model_output_shape(
            action_space: gym.Space,
            model_config: ModelConfigDict) -> Union[int, np.ndarray]:
        return 18


class VictimActionDist(TFActionDistribution):
    def _build_sample_op(self) -> TensorType:
        return self.mean + self.std * tf.random.normal(tf.shape(self.mean))

    def _go_to_predator(self):
        dx1 = tf.reshape(self.observation[0][4] - self.observation[0][0], [1, 1])
        dy1 = tf.reshape(self.observation[0][5] - self.observation[0][1], [1, 1])
        dx2 = tf.reshape(self.observation[0][9] - self.observation[0][0], [1, 1])
        dy2 = tf.reshape(self.observation[0][10] - self.observation[0][1], [1, 1])
        tn = tf.multiply(tf.concat([dx1, dy1, dx2, dy2], 1), 100)  # vector v-p
        return -tf.reshape(tn, [1, 4])

    def _go_from_predator(self):
        return -self._go_to_predator()

    def __init__(self, inputs: List[TensorType], model: TFModelV2):
        mean, log_std, observations = tf.split(inputs, tf.constant([4, 4, 14]), axis=1)
        self.mean = mean
        self.log_std = log_std
        self.std = tf.exp(log_std)
        self.observation = observations
        super().__init__(inputs, model)

    def _get_uniform_dist_value(self):
        random.seed()
        return random.random()

    def deterministic_sample(self) -> TensorType:
        prob = self._get_uniform_dist_value()
        if prob < 0.1:
            return self._go_to_predator()
        elif prob < 0.2:
            return self._go_from_predator()
        return self.mean

    def sample(self):
        prob = self._get_uniform_dist_value()
        if prob < 0.1:
            return self._go_to_predator()
        elif prob < 0.2:
            return self._go_from_predator()
        return self._build_sample_op()

    def logp(self, x: TensorType) -> TensorType:
        return -0.5 * tf.reduce_sum(
            tf.math.square((tf.cast(x, tf.float32) - self.mean) / self.std),
            axis=1
        ) - 0.5 * np.log(2.0 * np.pi) * tf.cast(tf.shape(x)[1], tf.float32) - \
            tf.reduce_sum(self.log_std, axis=1)

    def kl(self, other: ActionDistribution) -> TensorType:
        assert isinstance(other, VictimActionDist)
        return tf.reduce_sum(
            other.log_std - self.log_std +
            (tf.math.square(self.std) + tf.math.square(self.mean - other.mean))
            / (2.0 * tf.math.square(other.std)) - 0.5,
            axis=1)

    def entropy(self) -> TensorType:
        return tf.reduce_sum(
            self.log_std + .5 * np.log(2.0 * np.pi * np.e), axis=1)

    @staticmethod
    def required_model_output_shape(
            action_space: gym.Space,
            model_config: ModelConfigDict) -> Union[int, np.ndarray]:
        return 18



ready_to_exit = False


def press_key_exit(_q):
    global ready_to_exit
    ready_to_exit = True


class PredatorVictimModel(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(PredatorVictimModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        input_layer = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
        hidden_layer1 = tf.keras.layers.Dense(500, activation='relu')(input_layer)
        hidden_layer2 = tf.keras.layers.Dense(500, activation='relu')(hidden_layer1)
        hidden_layer3 = tf.keras.layers.Dense(500, activation='relu')(hidden_layer2)
        output_mean = tf.keras.layers.Dense(action_space.shape[0], activation='tanh')(hidden_layer3)
        output_std = tf.keras.layers.Dense(action_space.shape[0], activation='sigmoid')(hidden_layer3)
        output_layer = tf.keras.layers.Concatenate(axis=1)([output_mean, output_std, input_layer])
        value_layer = tf.keras.layers.Dense(1)(hidden_layer3)
        self.base_model = tf.keras.Model(input_layer, [output_layer, value_layer])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


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

if __name__ == "__main__":
    model_file_to_read = 'PredatorVictim_action_dist-v3-demo.pickle'
    model_file_to_write = 'PredatorVictim_action_dist-v3-demo-1.pickle'

    ray.init(include_dashboard=True)
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


    # evaluate(trainer, PVEnv, video_file='../videos/Predator_Victim_A3C')
    # # evaluate_length(trainer, PVEnv, "custom_victim-v2.txt")
    #
    #
    while True:
        try:
            for i in range(100):
                rest = trainer.train()
                print(rest['policy_reward_mean'])
            weights = trainer.save_to_object()
            pickle.dump(weights, open(model_file_to_write, 'wb'))
            print('Model saved')

        except KeyboardInterrupt:
            break

    weights = trainer.save_to_object()
    pickle.dump(weights, open(model_file_to_write, 'wb'))
    print('Model saved')


    evaluate(trainer, PVEnv, video_file='videos/Predator_Victim_A3C')
