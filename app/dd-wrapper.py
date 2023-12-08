
#!python

import sys
import os
from pyboy import PyBoy, WindowEvent

# Makes us able to import PyBoy from the directory below
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + "/../..")

# Check for ROM
if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    print("Usage: python wrapper-dd.py ROM")
    exit(1)

quiet = "--quiet" in sys.argv
pyboy = PyBoy("ignored/"+filename, window_type="headless" if quiet else "SDL2", window_scale=3, debug=not quiet, game_wrapper=False)
pyboy.set_emulation_speed(0)

dd = pyboy.game_wrapper()

print(dd)

pyboy.send_input(WindowEvent.PRESS_ARROW_RIGHT)
while(True):
    health = PyBoy.get_memory_value(pyboy,0xC499)
    scoreArray = [PyBoy.get_memory_value(pyboy,0xC645), PyBoy.get_memory_value(pyboy,0xC644), PyBoy.get_memory_value(pyboy,0xC643), PyBoy.get_memory_value(pyboy,0xC642), PyBoy.get_memory_value(pyboy,0xC641), PyBoy.get_memory_value(pyboy,0xC640)]
    scoreText = "".join(str(x) for x in scoreArray)
    score = int(scoreText)
    print(health)
    print(score)
    pyboy.tick()

dd.reset_game()

pyboy.stop()