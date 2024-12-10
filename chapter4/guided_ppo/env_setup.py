import gymnasium as gym
import numpy as np

def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        # Environment setup logic as before...
        if idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
        else:
            env = gym.make(env_id)
        
        env = wrap_env(env, capture_video, run_name, gamma)
        return env
    return thunk

def wrap_env(env, capture_video, run_name, gamma):
    # Apply all wrappers here, making this function flexible and configurable
    if capture_video:
        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    env = gym.wrappers.NormalizeReward(env, gamma=gamma)
    env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    return env