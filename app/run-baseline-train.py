from os.path import exists
from pathlib import Path
import uuid
import random
from ddenv import DDEnv
from stable_baselines3 import A2C, PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback
from typing import Callable, Union
from PIL import Image


def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: (float or str)
    :return: (function)
    """
    # Force conversion to float
    initial_value_ = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value_

    return func

def make_env(rank, env_conf, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = DDEnv(env_conf)
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init

if __name__ == '__main__':


    ep_length = 2048 * 32
    sess_path = Path(f'session_{str(uuid.uuid4())[:8]}')

    env_config = {
                'headless': True, 'save_final_state': True, 'early_stop': False,
                'action_freq': 5, 'init_state': 'ignored/dd.gb.state', 'max_steps': ep_length, 
                'print_rewards': True, 'save_video': False, 'fast_video': False, 'session_path': sess_path,
                'gb_path': 'ignored/dd.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0, 
                'use_screen_explore': True, 'extra_buttons': False
            }
    
    
    num_cpu = 20 #64 #46  # Also sets the number of episodes per training iteration
    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])
    
    checkpoint_callback = CheckpointCallback(save_freq=ep_length, save_path=sess_path,
                                     name_prefix='dd')
    #env_checker.check_env(env)
    file_name = '../sessions/no_points_reward/'
    
    if exists(file_name + '.zip'):
        print('\nloading checkpoint')
        model = PPO.load(file_name, env=env, device="cpu", tensorboard_log="./double_dragon_logs/")
        model.n_steps = ep_length
        model.ent_coef = 0.05
        model.learning_rate = linear_schedule(0.001)
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = ep_length
        model.rollout_buffer.n_envs = num_cpu
        model.verbose = 1
        model.gamma = 0.977
        model.batch_size = 512
        model.rollout_buffer.reset()
    else:
        print("Knowledge is power")
        model = PPO('CnnPolicy', env, verbose=1, learning_rate = linear_schedule(0.001), n_steps=ep_length, batch_size=512, n_epochs=3, gamma=0.999)
    
    model.learn(total_timesteps=(ep_length)*num_cpu*1000, callback=checkpoint_callback)