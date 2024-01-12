import retro
from utils import *
import numpy as np

env = retro.make(game="SonicTheHedgehog-Genesis", state="LabyrinthZone.Act1")

env.reset()

done = False

while not done:
    env.render()

    action = POSSIBLE_ACTIONS[np.random.randint(0, len(POSSIBLE_ACTIONS))]

    ob, rew, done, info = env.step(action)

    

    