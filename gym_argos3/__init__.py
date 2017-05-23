from gym.envs.registration import register

register(
        id='argos3-v0',
        entry_point='gym_argos3.envs:Argos3Env',
        )
