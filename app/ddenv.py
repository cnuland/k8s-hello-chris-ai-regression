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
from pathlib import Path
import mediapy as media
from PIL import Image
from collections import deque



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
        self.gb_path = config['gb_path']
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
        Path(self.s_path).mkdir(parents=True, exist_ok=True)

        self.all_runs = []
        self.old_pos = []
        self.total_score_rew = 0

        # Set this in SOME subclasses
        self.metadata = {"render.modes": []}
        self.reward_range = (0, 100000)

        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            #WindowEvent.PRESS_BUTTON_B,
            97,
            98,
            #99, # A and B for the jump kick
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
            #WindowEvent.RELEASE_BUTTON_B
        ]

        self.output_shape = (36, 40, 3)
        self.mem_padding = 2
        self.memory_height = 8
        self.col_steps = 16
        self.output_full = (
            self.output_shape[0] * self.frame_stacks + 2 * (self.mem_padding + self.memory_height),
                            self.output_shape[1],
                            self.output_shape[2])

        head = 'headless' if config['headless'] else 'SDL2'

         # Set these in ALL subclasses
        self.action_space = spaces.Discrete(len(self.valid_actions))
        self.observation_space = spaces.Box(low=0, high=255, shape=self.output_shape, dtype=np.uint8)

        
        self.pyboy = PyBoy(
                self.gb_path,
                debugging=True,
                disable_input=False,
                window_type=head,
                hide_window='--quiet' in sys.argv,     
                       
                )

        self.screen = self.pyboy.botsupport_manager().screen()

        if not config['headless']:
            self.pyboy.set_emulation_speed(20)
        print("Starting Instance")

    def reset(self, *, seed=None, options=None):
        # restart game, skipping credits
        with open(self.init_state, "rb") as f:
            self.pyboy.load_state(f)

        if self.save_video:
            base_dir = self.s_path / Path('rollouts')
            base_dir.mkdir(exist_ok=True)
            full_name = Path(f'full_reset_{self.reset_count}_id{self.instance_id}').with_suffix('.mp4')
            model_name = Path(f'model_reset_{self.reset_count}_id{self.instance_id}').with_suffix('.mp4')
            self.full_frame_writer = media.VideoWriter(base_dir / full_name, (144, 160), fps=60)
            self.full_frame_writer.__enter__()
            #self.model_frame_writer = media.VideoWriter(base_dir / model_name, self.output_full[:2], fps=60)
            #self.model_frame_writer.__enter__()

        #self.recent_frames = deque()
        #self.recent_frame = np.zeros(
            #(self.frame_stacks, self.output_shape[0], 
            # self.output_shape[1], self.output_shape[2]),
            #dtype=np.uint8)
        #self.recent_memory = np.zeros((self.output_shape[1]*self.memory_height, 3), dtype=np.uint8)

        self.old_x_pos = []
        self.old_y_pos = []
        self.visited_x = []
        self.visited_y = []

        self.levels_satisfied = False
        self.base_explore = 0
        self.last_lives = 3
        self.kick_penality = False
        self.levels = 0
        self.total_lives_rew = 3
        self.last_score = 0
        self.last_level = 0
        self.step_count = 0
        self.movement_reward = 0
        self.reset_count += 1
        self.locations = {
            1: False,
            2: False,
            3: False,
            4: False,
            5: False,
            6: False,
            7: False,
        }

        self.progress_reward = self.get_game_state_reward()
        self.total_reward = sum([val for _, val in self.progress_reward.items()])
        return self.render(), {}
    
    def render(self, reduce_res=True, add_memory=False, update_mem=True):
        game_pixels_render = self.screen.screen_ndarray() # (144, 160, 3)
        if reduce_res:
            game_pixels_render = (255*resize(game_pixels_render, self.output_shape)).astype(np.uint8)
            #self.recent_frames.append(game_pixels_render)
            #self.recent_frame[0] = game_pixels_render

            #if self.step_count < 4:
            #    self.recent_frame[0] = game_pixels_render
            #else:
            #    self.recent_frame[0] = self.recent_frames.pop()
                    
            if add_memory:
                pad = np.zeros(
                    shape=(self.mem_padding, self.output_shape[1], 3), 
                    dtype=np.uint8)
                game_pixels_render = np.concatenate(
                    (
                        self.create_exploration_memory(), 
                        pad,
                        self.create_recent_memory(),
                        pad,
                        rearrange(self.recent_frame, 'f h w c -> (f h) w c')
                    ),
                    axis=0)
        return game_pixels_render
    def step(self, action):
        self.run_action_on_emulator(action)
        #self.recent_frame = np.roll(self.recent_frame, 1, axis=0)
        obs_memory = self.render()
        self.step_count += 1

        new_reward, new_prog = self.update_reward()
        
        # shift over short term reward memory
        #self.recent_memory = np.roll(self.recent_memory, 3)
        #self.recent_memory[0, 0] = min(new_prog[0] * 64, 255)
        #self.recent_memory[0, 1] = min(new_prog[1] * 64, 255)
        #self.recent_memory[0, 2] = min(new_prog[2] * 128, 255)

        step_limit_reached = self.check_if_done()
        if step_limit_reached:
            print("INSTANCE:"+str(self.instance_id))
            print("FINAL SCORE:"+str(self.total_score_rew))
            print("FINAL LEVEL:"+str(self.levels))
            print("\n")
            self.total_score_rew = 0 # needs to reset here as there are times the reset command has already started running before this goes, not sure why this is...

        #img = Image.fromarray(obs_memory, 'RGB')
        #img.save('data.png')
        return obs_memory, new_reward*0.1, False, step_limit_reached, {}
    
    def run_action_on_emulator(self, action):
     # special move
        if self.valid_actions[action] == 99:
            self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
            self.pyboy.send_input(WindowEvent.PRESS_BUTTON_B)
            self.pyboy.tick()
            self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
            self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_B)
            self.kick_penality = True
            return
        elif self.valid_actions[action] == 97:
            self.pyboy.send_input(WindowEvent.PRESS_ARROW_LEFT)
            self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
            self.pyboy.send_input(WindowEvent.PRESS_BUTTON_B)
            self.pyboy.tick()
            self.pyboy.send_input(WindowEvent.RELEASE_ARROW_LEFT)
            self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
            self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_B)
            self.kick_penality = True

            return
        elif self.valid_actions[action] == 98:
            self.pyboy.send_input(WindowEvent.PRESS_ARROW_RIGHT)
            self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
            self.pyboy.send_input(WindowEvent.PRESS_BUTTON_B)
            self.pyboy.tick()
            self.pyboy.send_input(WindowEvent.RELEASE_ARROW_RIGHT)
            self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
            self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_B)
            self.kick_penality = True
            return
        # press button then release after some steps
        self.pyboy.send_input(self.valid_actions[action])
        for i in range(self.act_freq):
            # release action, so they are stateless
            if i == 7:
                if action > 3 and action < 6:
                    # release button 
                    self.pyboy.send_input(self.release_button[action - 4])
                if action < 4:
                    # release arrow
                    self.pyboy.send_input(self.release_arrow[action])
                if self.valid_actions[action] == WindowEvent.PRESS_BUTTON_START:
                    self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
            if self.save_video and not self.fast_video:
                self.add_video_frame()
            self.pyboy.tick()
        if self.save_video and self.fast_video:
            self.add_video_frame()
    
    def create_recent_memory(self):
        return rearrange(
            self.recent_memory, 
            '(w h) c -> h w c', 
            h=self.memory_height)
     
    def add_video_frame(self):
        self.full_frame_writer.add_image(self.render(reduce_res=False))
        #self.model_frame_writer.add_image(self.render(reduce_res=True))

    def get_score(self):
        scoreArray = [PyBoy.get_memory_value(self.pyboy,0xC645), PyBoy.get_memory_value(self.pyboy,0xC644), PyBoy.get_memory_value(self.pyboy,0xC643), PyBoy.get_memory_value(self.pyboy,0xC642), PyBoy.get_memory_value(self.pyboy,0xC641), PyBoy.get_memory_value(self.pyboy,0xC640)]
        scoreText = "".join(str(x) for x in scoreArray)
        score = int(scoreText)
        return score
    
    #This function looks at the background and makes sure there are changes happening. Promotes that the AI keeps moving and doesn't get stuck
    def get_screen_x_position(self):
        return [PyBoy.get_memory_value(self.pyboy,0xE100), PyBoy.get_memory_value(self.pyboy,0xE101), PyBoy.get_memory_value(self.pyboy,0xE102), PyBoy.get_memory_value(self.pyboy,0xE103), PyBoy.get_memory_value(self.pyboy,0xE104), PyBoy.get_memory_value(self.pyboy,0xE105), PyBoy.get_memory_value(self.pyboy,0xE106), PyBoy.get_memory_value(self.pyboy,0xE107), PyBoy.get_memory_value(self.pyboy,0xE108), PyBoy.get_memory_value(self.pyboy,0xE109), PyBoy.get_memory_value(self.pyboy,0xE10A), PyBoy.get_memory_value(self.pyboy,0xE10B), PyBoy.get_memory_value(self.pyboy,0xE10C), PyBoy.get_memory_value(self.pyboy,0xE10D), PyBoy.get_memory_value(self.pyboy,0xE10E), PyBoy.get_memory_value(self.pyboy,0xE10F)]

    def get_screen_y_position(self):
        return [PyBoy.get_memory_value(self.pyboy,0xE210), PyBoy.get_memory_value(self.pyboy,0xE211), PyBoy.get_memory_value(self.pyboy,0xE212), PyBoy.get_memory_value(self.pyboy,0xE213), PyBoy.get_memory_value(self.pyboy,0xE214), PyBoy.get_memory_value(self.pyboy,0xE215), PyBoy.get_memory_value(self.pyboy,0xE216), PyBoy.get_memory_value(self.pyboy,0xE217), PyBoy.get_memory_value(self.pyboy,0xE218), PyBoy.get_memory_value(self.pyboy,0xE219), PyBoy.get_memory_value(self.pyboy,0xE21A), PyBoy.get_memory_value(self.pyboy,0xE21B), PyBoy.get_memory_value(self.pyboy,0xE21C), PyBoy.get_memory_value(self.pyboy,0xE21D), PyBoy.get_memory_value(self.pyboy,0xE21E), PyBoy.get_memory_value(self.pyboy,0xE21F)]

    def get_lives(self):
        return PyBoy.get_memory_value(self.pyboy,0xC499)
    
    def get_level(self):
        return PyBoy.get_memory_value(self.pyboy,0xE110)
    
    def get_position_reward(self):
        pos_x = self.get_screen_x_position()
        pos_y = self.get_screen_y_position()
        if pos_x != self.old_x_pos or pos_y != self.old_y_pos: # Moving into a new frame, add a positioning reward
            self.old_x_pos = pos_x
            self.old_y_pos = pos_y
            return (.5 * self.levels)
        else:
            self.old_x_pos = pos_x
            self.old_y_pos = pos_y
            return (-1 * self.levels)


    def get_score_reward(self):
        new_score = self.get_score()
        if self.last_score != new_score:
            difference = new_score - self.last_score
            self.last_score = new_score
            self.total_score_rew += difference
            return (difference * self.levels)
        else:
            return 0
    
    def get_level_reward(self):
        new_level = self.get_level()
        if self.last_level != new_level: # Not sure if it's part of the gameboy or just a bug but the game shuffles between a few levels when moving to a new level, so there's a need to call out the specific level
            if new_level == 15 and self.locations[1] == False: # starting level
                self.locations[1] = True
                self.last_level = new_level
                self.levels+=1
                return 0
            elif new_level == 84 and self.locations[2] == False: # starting level
                self.locations[2] = True
                self.levels+=1
                self.last_level = new_level
                return 3000
            elif new_level == 48 and self.locations[3] == False: # starting level
                self.locations[3] = True
                self.levels+=1
                self.last_level = new_level
                return 6000         
            elif new_level == 89 and self.locations[4] == False: # starting level
                self.locations[4] = True
                self.levels+=1
                self.last_level = new_level
                return 1000
            elif new_level == 11 and self.locations[5] == False: # starting level
                self.locations[5] = True
                self.levels+=1
                self.last_level = new_level
                return 12000
            else:
                return 0
        else:
            return 0

    def get_lives_reward(self):
        new_lives = self.get_lives()
        if self.last_lives != new_lives:
            difference = new_lives - self.last_lives
            self.last_health = self.total_lives_rew
            self.total_lives_rew = new_lives
            if new_lives == 0: # putting this here because we need to update the lives for other functions
                return -100 # Let's make dying bad
            return difference
        else:
            return 0

    def get_moves_penality(self):
        if self.kick_penality:
            self.kick_penality = False
            return -30 # Let's make dying bad
        else:
            return 0

    def update_reward(self):
        # compute reward
        old_prog = self.group_rewards()
        self.progress_reward = self.get_game_state_reward()
        new_prog = self.group_rewards()
        new_total = sum([val for _, val in self.progress_reward.items()]) #sqrt(self.explore_reward * self.progress_reward)
        new_step = new_total - self.total_reward
    
        self.total_reward = new_total
        return (new_step, 
                   (new_prog[0]-old_prog[0], 
                    new_prog[1]-old_prog[1], 
                    new_prog[2]-old_prog[2],
                    new_prog[3]-old_prog[3],
                    )
               )

    def group_rewards(self):
        prog = self.progress_reward
        # these values are only used by memory
        return (
            prog['score'],
            prog['lives'],
            prog['pos'],
            prog['level'],
            )

    def create_exploration_memory(self):
        w = self.output_shape[1]
        h = self.memory_height
        
        def make_reward_channel(r_val):
            col_steps = self.col_steps
            max_r_val = (w-1) * h * col_steps
            # truncate progress bar. if hitting this
            # you should scale down the reward in group_rewards!
            r_val = min(r_val, max_r_val)
            row = floor(r_val / (h * col_steps))
            memory = np.zeros(shape=(h, w), dtype=np.uint8)
            memory[:, :row] = 255
            row_covered = row * h * col_steps
            col = floor((r_val - row_covered) / col_steps)
            memory[:col, row] = 255
            col_covered = col * col_steps
            last_pixel = floor(r_val - row_covered - col_covered) 
            memory[col, row] = last_pixel * (255 // col_steps)
            return memory
        
        score, pos, level, lives = self.group_rewards()
        full_memory = np.stack((
            make_reward_channel(level),
            make_reward_channel(pos),
            make_reward_channel(pos),

        ), axis=-1)

        return full_memory       

    def get_game_state_reward(self, print_stats=True):
        #score = self.get_score_reward()
        #lives = self.get_lives_reward() # we aren't using it but its important to calculate to tell when the game is done
        state_scores = {
            'score': int(self.get_score_reward()  *  0.001),
            'pos': int(self.get_position_reward() *  0.2 ),
            'level': int(self.get_level_reward()  *  0.1),
            'lives': int(self.get_lives_reward()  *  0.1),
            'moves': int(self.get_moves_penality()),
        }

        return state_scores
    
    def check_if_done(self):
        done = False
        done = self.step_count >= self.max_steps
        if self.total_lives_rew == 0:
            done = True
        if self.save_video and done:
            self.full_frame_writer.close()
        return done