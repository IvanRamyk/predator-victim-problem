default_params = {
    "predator": {"max_vel": 0.01, "max_acceleration": 0.001},
    "victim": {"max_vel": 0.01, "max_acceleration": 0.001},
    "reward_scale": 0.1,
    "max_steps": 2000,
    "is_continuous": True,
    "catch_distance": 0.1,
}

default_params_for_multiple_victims = {
    "predator": {"max_vel": 0.01, "max_acceleration": 0.001},
    "victims": {"max_vel": 0.005, "max_acceleration": 0.001},
    "n_victims": 2,
    "reward_scale": 0.1,
    "max_steps": 4000,
    "is_continuous": True,
    "catch_distance": 0.1,
}

