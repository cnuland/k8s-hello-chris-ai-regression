from os.path import exists
from pathlib import Path
import uuid
import random
from ddenv import DDEnv
from stable_baselines3 import PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from tensorboard_callback import TensorboardCallback

def make_env(rank, env_conf, seed=0):

    def _init():
        env = DDEnv(env_conf)
        return env
    set_random_seed(seed)
    return _init

if __name__ == '__main__':

    use_wandb_logging = True
    ep_length = 2048 * 10
    sess_id = str(uuid.uuid4())[:8]
    sess_path = Path(f'session_{sess_id}')

    env_config = {
                'headless': False, 'save_final_state': True, 'early_stop': False,
                'action_freq': 24, 'init_state': 'ignored/dd.gb.state', 'max_steps': ep_length, 
                'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
                'gb_path': 'ignored/dd.gb', 'debug': True, 'sim_frame_dist': 2_000_000.0, 
                'use_screen_explore': True, 'reward_scale': 4, 'extra_buttons': False,
                'explore_weight': 3 # 2.5
            }
    
    print(env_config)
    
    num_cpu = 1  # Also sets the number of episodes per training iteration
    #env = DummyVecEnv([make_env(i, env_config) for i in range(num_cpu)])
    env = make_env(0, env_config)() #SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])
        
    #keyboard.on_press_key("M", toggle_agent)
    obs, info = env.reset()
    while True:
        actions = [0, 1, 2, 3, 4, 5] # pass action
        action = random.choice(actions)
        obs, rewards, terminated, truncated, info = env.step(action)
        env.render()