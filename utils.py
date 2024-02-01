from gym.wrappers import FrameStack, TimeLimit
from wrappers import *
import retro
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY, COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
import matplotlib.pyplot as plt
import os
import yaml
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

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

def apply_wrappers(env, skip="max_and_skip", gray_scale = True, shape=[84, 84], num_stack=4):
    if skip=="max_and_skip":
        env = MaxAndSkipEnv(env, 4) #Returns only ith frame, same action for i frames, observation returned is the maxpooling over last 2 frames
    else:
        if skip:
            env = SkipFrame(env, skip=skip)

    if shape==[84, 84]:
        env = ProcessFrame84(env)
    else:
        if gray_scale:
            env = GrayScaleObservation(env)
        if shape:
            env = ResizeObservation(env, shape=shape)

    if num_stack:
        env = FrameStack(env, num_stack=num_stack)

    return env

def get_env(game="mario", level="SuperMarioBros-1-1-v0", action_space="COMPLEX_MOVEMENT"):
    if game=="sonic":
        env = retro.make(game="SonicTheHedgehog-Genesis", state=level, scenario='contest')
        env = SonicActionSpace(env, POSSIBLE_ACTIONS_SONIC)
        env = TimeLimit(env, max_episode_steps=30000)
        return env
    elif game=="mario":
        env = gym.make(level)
        env = JoypadSpace(env, MAPPING_ACTION_SPACE_MARIO[action_space])
        return env
    else:
        return None
    
def get_env_name(env): 
    try:
        return env.spec.id
    except:
        if (env.gamename=="SonicTheHedgehog-Genesis"):
            return env.gamename
        return None

def get_action(index, env): 
    try:
        if (env.spec.id=="SuperMarioBros-1-1-v0"):        
            return index
    except:
        if (env.gamename=="SonicTheHedgehog-Genesis"):
            return index
        return None

def get_action_space_size(env): 
    return env.action_space.n
    
def get_action_sample(env):
    action_index = np.random.randint(0, get_action_space_size(env))

    return get_action(action_index, env)

def plot_sequence_observations(next_state):
    fig=plt.figure(figsize=(28, 15))

    for i in range(len(next_state)):
        ax = fig.add_subplot(1, len(next_state), i+1)
        ax.imshow(next_state[i]) #Converting to matplotlib format
        ax.set_title(f"Frame {i+1}")

def create_folder_path(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def get_params():
    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)
    return params["environment"], params["logging"], params["hyperparameters"]

def save_hyperparameters(hp_dict, path):
    file_path = os.path.join(path, "hyperparameters.yaml")
    with open(file_path, "w") as file:
        yaml.dump(hp_dict, file, default_flow_style=False)
    
def plot_average_reward(data, title="Episodes trained vs. Average Rewards", n_average=500):
    plt.title(title)
    plt.plot([0 for _ in range(n_average)] + 
            np.convolve(data, np.ones((n_average,))/n_average, mode="valid").tolist())
    plt.show()
        