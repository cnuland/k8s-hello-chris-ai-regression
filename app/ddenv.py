import sys
import os
from math import floor, sqrt
import uuid
import numpy as np
from einops import rearrange
import matplotlib.pyplot as plt
from skimage.transform import resize
from pyboy import PyBoy
from pyboy.logger import log_level
import pandas as pd

from gymnasium import Env, spaces
from pyboy.utils import WindowEvent
# Makes us able to import PyBoy from the directory below
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + "/../..")

class DDEnv(Env):


    def __init__(
        self, config=None):

        # Check for ROM
        if len(sys.argv) > 1:
            filename = sys.argv[1]
        else:
            print("Usage: python wrapper-dd.py ROM")
            exit(1)

        self.debug = config['debug']
        self.s_path = config['session_path']
        self.save_final_state = config['save_final_state']
        self.print_rewards = config['print_rewards']
        self.vec_dim = 42*42*3 #4320 #1000
        self.headless = config['headless']
        self.num_elements = 20000 # max
        self.init_state = config['init_state']
        self.act_freq = config['action_freq']
        self.max_steps = config['max_steps']
        self.early_stopping = config['early_stop']
        self.save_video = config['save_video']
        self.fast_video = config['fast_video']
        self.video_interval = 256 * self.act_freq
        self.downsample_factor = 2
        self.frame_stacks = 3
        self.similar_frame_dist = config['sim_frame_dist']
        self.reset_count = 0
        self.instance_id = str(uuid.uuid4())[:8]
        self.s_path.mkdir(exist_ok=True)
        self.all_runs = []

        # Set this in SOME subclasses
        self.metadata = {"render.modes": []}
        self.reward_range = (0, 15000)

        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            99, # A and B for the jump kick
        ]
        
        self.extra_buttons: [
                WindowEvent.PRESS_BUTTON_START,
                WindowEvent.PASS
            ]

        self.release_arrow = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP
        ]

        self.release_button = [
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B
        ]

        self.output_shape = (42,42,3)#(36, 40, 3)
        self.mem_padding = 2
        self.memory_height = 8
        self.col_steps = 16
        self.output_full = (
            self.output_shape[0] * self.frame_stacks + 2 * (self.mem_padding + self.memory_height),
                            self.output_shape[1],
                            self.output_shape[2]
        )

        head = 'headless' if config['headless'] else 'SDL2'

         # Set these in ALL subclasses
        self.action_space = spaces.Discrete(len(self.valid_actions))
        self.observation_space = spaces.Box(low=0, high=255, shape=self.output_shape, dtype=np.uint8)

        
        self.pyboy = PyBoy(
                "ignored/"+filename,
                debugging=True,
                disable_input=False,
                window_type=head,
                hide_window='--quiet' in sys.argv,     
                       
                )

        self.screen = self.pyboy.botsupport_manager().screen()

        if not config['headless']:
            self.pyboy.set_emulation_speed(6)
        print("Lets get this party started")

    def reset(self, *, seed=None, options=None):
        # restart game, skipping credits
        with open(self.init_state, "rb") as f:
            self.pyboy.load_state(f)

        self.levels_satisfied = False
        self.base_explore = 0
        self.last_lives = 3
        self.total_lives_rew = 3
        self.last_score = 0
        self.total_score_rew = 0
        self.step_count = 0
        self.reset_count += 1
        return self.render(), {}
    
    def render(self, reduce_res=True):
        game_pixels_render = self.screen.screen_ndarray() # (144, 160, 3)
        if reduce_res:
            game_pixels_render = (255*resize(game_pixels_render, self.output_shape)).astype(np.uint8)
        return game_pixels_render
    
    def step(self, action):
        self.run_action_on_emulator(action)
        obs_memory = self.render()
        new_reward = self.update_reward()
        self.step_count += 1

        step_limit_reached = self.check_if_done()
        if step_limit_reached:
            print("DEATH")

        return obs_memory, new_reward*0.1, False, step_limit_reached, {}
    
    def run_action_on_emulator(self, action):
        # special move
        if self.valid_actions[action] == 99:
            self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
            self.pyboy.send_input(WindowEvent.PRESS_BUTTON_B)
            self.pyboy.tick()
            self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
            self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_B)
            return
        # press button then release after some steps
        self.pyboy.send_input(self.valid_actions[action])
        for i in range(self.act_freq):
            # release action, so they are stateless
            if i == 8:
                if action < 4:
                    # release arrow
                    self.pyboy.send_input(self.release_arrow[action])
                if action > 3 and action < 6:
                    # release button 
                    self.pyboy.send_input(self.release_button[action - 4])
                if self.valid_actions[action] == WindowEvent.PRESS_BUTTON_START:
                    self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)

            self.pyboy.tick()
    
    def get_score(self):
        scoreArray = [PyBoy.get_memory_value(self.pyboy,0xC645), PyBoy.get_memory_value(self.pyboy,0xC644), PyBoy.get_memory_value(self.pyboy,0xC643), PyBoy.get_memory_value(self.pyboy,0xC642), PyBoy.get_memory_value(self.pyboy,0xC641), PyBoy.get_memory_value(self.pyboy,0xC640)]
        scoreText = "".join(str(x) for x in scoreArray)
        score = int(scoreText)
        return score
    
    def get_lives(self):
        return PyBoy.get_memory_value(self.pyboy,0xC499)
    
    def get_score_reward(self):
        new_score = self.get_score()
        if self.last_score != new_score:
            difference = new_score - self.last_score
            self.last_score = self.total_score_rew
            self.total_score_rew = new_score
            return difference
        else:
            return 0
    
    def get_lives_reward(self):
        new_lives = self.get_lives()
        if self.last_lives != new_lives:
            difference = new_lives - self.last_lives
            self.last_health = self.total_lives_rew
            self.total_lives_rew = new_lives
            if new_lives == 0: # putting this here because we need to update the lives for other functions
                return -10 # Let's make dying bad
            return difference
        else:
            return 0
    
    def update_reward(self):
        # compute reward
        self.progress_reward = self.get_game_state_reward()
        new_total = sum([val for _, val in self.progress_reward.items()]) #sqrt(self.explore_reward * self.progress_reward)
        return new_total


    def get_game_state_reward(self, print_stats=True):

        state_scores = {
            'lives': self.get_lives_reward() * 15,  
            'score': int(self.get_score_reward() // 10), 
        }

        if print_stats and self.step_count % 20 == 0:
            print("CURRENT SCORE:"+str(self.total_score_rew))
        
        return state_scores
    
    def check_if_done(self):
        if self.total_lives_rew == 0:
            return True
        done = self.step_count >= self.max_steps
        return done
