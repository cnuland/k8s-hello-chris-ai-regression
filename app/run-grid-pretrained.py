from os.path import exists
from pathlib import Path
import sys
import uuid
from ddenv import DDEnv
from stable_baselines3 import A2C, PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback

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
        #env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

def run_save(save):
    save = Path(save)
    ep_length = 2048 * 12
    sess_path = f'grid_renders/session_{save.stem}'
    env_config = {
                'headless': True, 'save_final_state': True, 'early_stop': False,
                'action_freq': 9, 'init_state': ["ignored/dd.gb.state", "ignored/lvl2-dd.gb.state", "ignored/lvl3-dd.gb.state", "ignored/lvl3.5-dd.gb.state","ignored/lvl-1.5.dd.gb.state"], 'max_steps': ep_length, 
                'print_rewards': True, 'save_video': True, 'fast_video': False, 'session_path': sess_path,
                'gb_path': 'ignored/dd.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0
            }
    num_cpu = 15  # Also sets the number of episodes per training iteration
    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])

    checkpoint_callback = CheckpointCallback(save_freq=ep_length, save_path=sess_path,
                                     name_prefix='dd')
    #env_checker.check_env(env)
    learn_steps = 1
    file_name = save
    if exists(file_name):
        print('\nloading checkpoint')
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
            "n_steps": ep_length
        }
        model = PPO.load(file_name, env=env, custom_objects=custom_objects)
        model.n_steps = ep_length
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = ep_length
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()
    else:
        print('initializing new policy')
        model = PPO('CnnPolicy', env, verbose=1, n_steps=ep_length, batch_size=512, n_epochs=1, gamma=0.999)

    model.learn(total_timesteps=(ep_length)*num_cpu, callback=checkpoint_callback)


if __name__ == '__main__':
    run_save(sys.argv[1])
    
        
