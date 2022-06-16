from gym.envs.registration import register

register(
    id='PredatorVictim-v0',
    entry_point='PredatorVictim.PredatorVictim:PredatorVictim',
)
register(
    id='PredatorMultipleVictims-v0',
    entry_point='PredatorVictim.PredatorMultipleVictims:PredatorMultipleVictims',
)