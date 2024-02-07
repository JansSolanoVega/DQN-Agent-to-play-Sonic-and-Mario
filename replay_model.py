from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from utils import *
import time
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"

def run_model(game="sonic"):
    #Loading parameters
    ep, lp, hp = get_params(game) #params["environment"], params["logging"], params["hyperparameters"]

    # Load the trained model
    model = DQN.load(r"best_models/best_model_"+game+".zip")

    env = get_env(game=ep["game"], level=ep["level"], action_space=ep["action_space"])
    env = apply_wrappers(env, skip=ep["skip"], gray_scale=ep["gray_scale"], shape=ep["frame_shape"], num_stack=ep["num_stack"])# Test the model

    env = gym.wrappers.RecordVideo(env, 'results/video_'+game)
    try:
        while True:
            obs = env.reset()
            done = False
            while not done:
                action, _ = model.predict(np.array(obs), deterministic=True)
                #print(action)
                obs, _, done, _ = env.step(int(action))
                env.render()
                time.sleep(0.001)
    except:
        env.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', default='sonic')
    args = parser.parse_args()

    run_model(args.game)

if __name__ == '__main__':
    main()
