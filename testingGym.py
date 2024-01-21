from utils import *
import numpy as np
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.utils import play

env = get_env(game="mario", level="SuperMarioBros-1-1-v0", action_space="SIMPLE_MOVEMENT")
# (game="sonic", level="GreenHillZone.Act1")

env.reset()

while True:
    env.render()

    action_index = np.random.randint(0, get_action_space_size(env))

    ob, rew, done, info = env.step(get_action(action_index, env))
    
    print(rew)

    if done:
        break