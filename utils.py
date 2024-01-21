from gym.wrappers import FrameStack
from wrappers import *
import retro
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY, COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
import matplotlib.pyplot as plt

MAPPING_ACTION_SPACE_MARIO ={
    "SIMPLE_MOVEMENT": SIMPLE_MOVEMENT,
    "RIGHT_ONLY": RIGHT_ONLY,
    "COMPLEX_MOVEMENT": COMPLEX_MOVEMENT
}

POSSIBLE_ACTIONS_SONIC = {
    # No Operation
    0: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # Left
    1: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    # Right
    2: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    # Left, Down
    3: [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    # Right, Down
    4: [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    # Down
    5: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    # Down, B
    6: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    # B
    7: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}

MAP_IDS_TO_NAME_SONIC = ["None", "Left", "Right", "Left, Down", "Right, Down", "Down", "Down, B", "B"]

def apply_wrappers(env, skip=4, gray_scale = True, shape=(156, 156), num_stack=4):
    if skip:
        env = SkipFrame(env, skip=skip)
    if gray_scale:
        env = GrayScaleObservation(env)
    if shape:
        env = ResizeObservation(env, shape=shape)
    if num_stack:
        env = FrameStack(env, num_stack=num_stack)
    return env

def get_env(game="mario", level="SuperMarioBros-1-1-v0", action_space="COMPLEX_MOVEMENT"):
    if game=="sonic":
        env = retro.make(game="SonicTheHedgehog-Genesis", state=level)
        return env
    elif game=="mario":
        env = gym.make(level)
        env = JoypadSpace(env, MAPPING_ACTION_SPACE_MARIO[action_space])
        return env
    else:
        return None

def get_action(index, env): 
    try:
        if (env.spec.id=="SuperMarioBros-1-1-v0"):        
            return index
    except:
        if (env.gamename=="SonicTheHedgehog-Genesis"):
            return POSSIBLE_ACTIONS_SONIC[index]
        return None

def get_action_space_size(env): 
    try:
        if (env.spec.id=="SuperMarioBros-1-1-v0"):        
            return env.action_space.n
    except:
        if (env.gamename=="SonicTheHedgehog-Genesis"):
            return len(POSSIBLE_ACTIONS_SONIC)
        return None
    
def get_action_sample(env):
    action_index = np.random.randint(0, get_action_space_size(env))

    return get_action(action_index, env)

def plot_sequence_observations(next_state):
    fig=plt.figure(figsize=(28, 15))

    for i in range(len(next_state)):
        ax = fig.add_subplot(1, len(next_state), i+1)
        ax.imshow(next_state[i]) #Converting to matplotlib format
        ax.set_title(f"Frame {i+1}")


