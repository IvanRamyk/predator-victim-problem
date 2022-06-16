import os
import pickle
import keyboard
import numpy as np
import gym
import ray
import ray.rllib.agents.a3c as a3c
import ray.rllib.agents.sac as sac
import ray.rllib.agents.pg as pg
import ray.rllib.agents.ddpg as ddpg
from ray.rllib.models import ModelCatalog
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.tune.registry import register_env
import tensorflow as tf
from PredatorVictim import PredatorVictim
from evaluate import evaluate, victim_custom
from ray.rllib.models.tf.tf_action_dist import TFActionDistribution
from ray.rllib.utils.typing import TensorType, List, Union, \
    Tuple, ModelConfigDict
import random
import ray.rllib.rollout


class VictimActionDist(TFActionDistribution):

    def _build_sample_op(self) -> TensorType:
        return self.mean + self.std * tf.random.normal(tf.shape(self.mean))

    def _go_to_predator(self):
        dx = self.observation[0][0] - self.observation[0][4]  # Predator.x - Victim.x
        dy = self.observation[0][1] - self.observation[0][5]  # py - vy
        dx = tf.reshape(dx, [1, 1])
        dy = tf.reshape(dy, [1, 1])
        tn = tf.multiply(tf.concat([dx, dy], 1), 100)  # vector v-p
        return tf.reshape(tn, [1, 2])

    def _go_from_predator(self):
        dx = self.observation[0][4] - self.observation[0][0]  # Predator.x - Victim.x
        dy = self.observation[0][5] - self.observation[0][1]  # py - vy
        dx = tf.reshape(dx, [1, 1])
        dy = tf.reshape(dy, [1, 1])
        tn = tf.multiply(tf.concat([dx, dy], 1), 100)  # vector v-p
        return tf.reshape(tn, [1, 2])

    def __init__(self, inputs: List[TensorType], model: TFModelV2):
        mean, log_std, observations = tf.split(inputs, tf.constant([2, 2, 8]), axis=1)
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
        elif prob < 1:
            return self._go_from_predator()
        return self.mean

    def sample(self):
        prob = self._get_uniform_dist_value()
        if prob < 0.1:
            return self._go_to_predator()
        elif prob < 1:
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

    def _build_sample_op(self) -> TensorType:
        return self.mean + self.std * tf.random.normal(tf.shape(self.mean))

    @staticmethod
    def required_model_output_shape(
            action_space: gym.Space,
            model_config: ModelConfigDict) -> Union[int, np.ndarray]:
        return 12


# class PredatorActionDist(TFActionDistribution):

#     def _up(self):

#         tn =  tf.constant([-1, 0], dtype=tf.float32)
#         return tf.reshape(tn, [1, 2])

#     def __init__(self, inputs: List[TensorType], model: TFModelV2):
#         mean, log_std, observations = tf.split(inputs, tf.constant([2, 2, 8]), axis=1)
#         self.mean = mean
#         self.log_std = log_std
#         self.std = tf.exp(log_std)
#         self.observation = observations
#         super().__init__(inputs, model)

#     def deterministic_sample(self) -> TensorType:
#         return self._up()

#     def sample(self):
#         return self._up()

#     def logp(self, x: TensorType) -> TensorType:
#         return -0.5 * tf.reduce_sum(
#             tf.math.square((tf.cast(x, tf.float32) - self.mean) / self.std),
#             axis=1
#         ) - 0.5 * np.log(2.0 * np.pi) * tf.cast(tf.shape(x)[1], tf.float32) - \
#             tf.reduce_sum(self.log_std, axis=1)

#     def kl(self, other: ActionDistribution) -> TensorType:
#         assert isinstance(other, PredatorActionDist)
#         return tf.reduce_sum(
#             other.log_std - self.log_std +
#             (tf.math.square(self.std) + tf.math.square(self.mean - other.mean))
#             / (2.0 * tf.math.square(other.std)) - 0.5,
#             axis=1)

#     def entropy(self) -> TensorType:
#         return tf.reduce_sum(
#             self.log_std + .5 * np.log(2.0 * np.pi * np.e), axis=1)

#     def _build_sample_op(self) -> TensorType:
#         return self.mean + self.std * tf.random.normal(tf.shape(self.mean))

#     @staticmethod
#     def required_model_output_shape(
#             action_space: gym.Space,
#             model_config: ModelConfigDict) -> Union[int, np.ndarray]:
#         return 12        


class PredatorModel(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        # raise Exception("Hack Pentagon")
        super(PredatorModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        input_layer = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
        hidden_layer1 = tf.keras.layers.Dense(512, activation='relu')(input_layer)
        hidden_layer2 = tf.keras.layers.Dense(512, activation='relu')(hidden_layer1)
        hidden_layer3 = tf.keras.layers.Dense(512, activation='relu')(hidden_layer2)
        output_mean = tf.keras.layers.Dense(2, activation='tanh')(hidden_layer3)
        output_std = tf.keras.layers.Dense(2, activation='sigmoid')(hidden_layer3)
        output_layer = tf.keras.layers.Concatenate(axis=1)([output_mean, output_std])
        value_layer = tf.keras.layers.Dense(1)(hidden_layer3)
        self.base_model = tf.keras.Model(input_layer, [output_layer, value_layer])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


# class PredatorModel12(TFModelV2):

#     def __init__(self, obs_space, action_space, num_outputs, model_config, name):
#         super(PredatorModel12, self).__init__(obs_space, action_space, num_outputs, model_config,name)
#         input_layer = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
#         hidden_layer1 = tf.keras.layers.Dense(512, activation='relu')(input_layer)
#         hidden_layer2 = tf.keras.layers.Dense(512, activation='relu')(hidden_layer1)
#         hidden_layer3 = tf.keras.layers.Dense(512, activation='relu')(hidden_layer2)
#         output_mean = tf.keras.layers.Dense(2, activation='tanh')(hidden_layer3)
#         output_std = tf.keras.layers.Dense(2, activation='sigmoid')(hidden_layer3)
#         output_layer = tf.keras.layers.Concatenate(axis=1)([output_mean, output_std, input_layer])
#         value_layer = tf.keras.layers.Dense(1)(hidden_layer3)
#         self.base_model = tf.keras.Model(input_layer, [output_layer, value_layer])
#         self.register_variables(self.base_model.variables)

#     def forward(self, input_dict, state, seq_lens):
#         model_out, self._value_out = self.base_model(input_dict["obs"])
#         return model_out, state

#     def value_function(self):
#         return tf.reshape(self._value_out, [-1])


class VictimModel(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(VictimModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        input_layer = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
        hidden_layer1 = tf.keras.layers.Dense(512, activation='relu')(input_layer)
        hidden_layer2 = tf.keras.layers.Dense(512, activation='relu')(hidden_layer1)
        hidden_layer3 = tf.keras.layers.Dense(512, activation='relu')(hidden_layer2)
        output_mean = tf.keras.layers.Dense(2, activation='tanh')(hidden_layer3)
        output_std = tf.keras.layers.Dense(2, activation='sigmoid')(hidden_layer3)
        output_layer = tf.keras.layers.Concatenate(axis=1)([output_mean, output_std, input_layer])
        value_layer = tf.keras.layers.Dense(1)(hidden_layer3)
        self.base_model = tf.keras.Model(input_layer, [output_layer, value_layer])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


def gen_policy(PVEnv, i, isVictim):
    config = {}
    if isVictim:
        ModelCatalog.register_custom_action_dist("victim_dist", VictimActionDist)
        ModelCatalog.register_custom_model("victim_model", VictimModel)
        config1 = {
            "model": {
                "custom_model": "victim_model",
                "custom_action_dist": "victim_dist",
            },
        }
        return None, PVEnv.observation_space, PVEnv.action_space, config1
    else:
        ModelCatalog.register_custom_model("predator_model", PredatorModel)
        config2 = {
            "model": {
                "custom_model": "predator_model",
                # "custom_action_dist": "predator_dist",
            },
        }
        return None, PVEnv.observation_space, PVEnv.action_space, config2


def policy_mapping_fn(agent_id):
    if agent_id == 'predator':
        return "policy_predator"
    elif agent_id == 'victim':
        return "policy_victim"


model_file = 'PV10-10-80.pickle'
params = {'predator': {'max_vel': 0.01, 'max_acceleration': 0.001},
          'victim': {'max_vel': 0.01, 'max_acceleration': 0.001},
          'reward_scale': 0.1,
          'max_steps': 2000,
          'is_continuous': True,
          'catch_distance': 0.1}

ray.init(include_dashboard=False)
PVEnv = gym.make("PredatorVictim-v0", params=params)
register_env("PredatorVictimEnv", lambda _: PVEnv)

trainer = a3c.A3CTrainer(env="PredatorVictimEnv", config={
    "multiagent": {
        "policies": {
            "policy_predator": gen_policy(PVEnv, 0, False),
            "policy_victim": gen_policy(PVEnv, 1, True),
        },
        "policy_mapping_fn": policy_mapping_fn,
    },
})
# trainer.export_policy_model()

if os.path.isfile(model_file):
    weights = pickle.load(open(model_file, "rb"))
    trainer.restore_from_object(weights)
    print("model restored!")

str = input()
try:
    if str == "1":
        while True:
            for i in range(1000):
                rest = trainer.train()
                print(rest['policy_reward_mean'])
            weights = trainer.save_to_object()
            pickle.dump(weights, open(model_file, 'wb'))
            print('Model saved')
    elif str == "2":
        evaluate(trainer, PVEnv, video_file='../videos/Predator_Victim_A3C')
    elif str == "3":
        victim_custom(trainer, PVEnv, video_file='../videos/Predator_Victim_A3C')
except Exception as e:
    print(e)
    weights = trainer.save_toobject()
    pickle.dump(weights, open(model_file, 'wb'))
    print('Model saved')
