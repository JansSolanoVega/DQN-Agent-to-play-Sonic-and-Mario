#Code adapted from https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html and https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/blob/master/Chapter06/lib/wrappers.py
import gym
from gym.spaces import Box
import numpy as np
import torch
from torchvision import transforms as T
import cv2
from skimage import transform

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

class SonicActionSpace(gym.Wrapper):
    def __init__(self, env, actions):
        super().__init__(env)
        self.actions = actions
        self.n_actions = len(actions)
        self.action_space = gym.spaces.Discrete(self.n_actions)

    def step(self, action):
        return self.env.step(self.actions[action])

class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        resize_obs = transform.resize(observation, self.shape)
        # cast float back to uint8
        resize_obs *= 255
        resize_obs = resize_obs.astype(np.uint8)
        return resize_obs
    
class ProcessFrame84(gym.ObservationWrapper):
    """
    Downsamples image to 84x84
    Greyscales image

    Returns numpy array
    """
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 240 * 256 * 3:#mario
            img = np.reshape(frame, [240, 256, 3]).astype(np.float32)
            img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
            resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
            x_t = resized_screen[18:102, :]
            x_t = np.reshape(x_t, [84, 84])
        elif frame.size == 224 * 320 * 3:#sonic
            img = np.reshape(frame, [224, 320, 3]).astype(np.float32)
            img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
            resized_screen = cv2.resize(img, (84, 95), interpolation=cv2.INTER_AREA)
            x_t = resized_screen[0:84, :]
            x_t = np.reshape(x_t, [84, 84])
        else:
            assert False, "Unknown resolution."
        return x_t.astype(np.uint8)