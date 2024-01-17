import retro
from utils import *
import numpy as np
from gym.wrappers import FrameStack
from wrappers import *

env = retro.make(game="SonicTheHedgehog-Genesis", state="LabyrinthZone.Act1")
env.reset()

next_state, reward, done, info = env.step(POSSIBLE_ACTIONS[np.random.randint(0, len(POSSIBLE_ACTIONS))])
print(f"{next_state.shape},\n {reward},\n {done},\n {info}")

# done = False

# while not done:
#     env.render()

#     action = POSSIBLE_ACTIONS[np.random.randint(0, len(POSSIBLE_ACTIONS))]

#     ob, rew, done, info = env.step(action)
    
#     print(f"{ob.shape},{rew},{done},{info}")
#     break

# Apply Wrappers to environment
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=(84, 84))
env = FrameStack(env, num_stack=4)

env.reset()

next_state, reward, done, info = env.step(POSSIBLE_ACTIONS[np.random.randint(0, len(POSSIBLE_ACTIONS))])
print(f"{next_state.shape},\n {reward},\n {done},\n {info}")

    