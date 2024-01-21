import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym.utils import play
import argparse
from utils import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', default='SuperMarioBros-v0')
    parser.add_argument('--state', default="COMPLEX_MOVEMENT")
    args = parser.parse_args()

    env = gym_super_mario_bros.make(args.game)
    env = JoypadSpace(env, MAPPING_ACTION_SPACE_MARIO[args.state])
    play.play(env, zoom=3, fps=60)

if __name__ == '__main__':
    main()