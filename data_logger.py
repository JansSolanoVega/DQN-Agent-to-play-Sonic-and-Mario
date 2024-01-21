from torch.utils.tensorboard import SummaryWriter
import os
from utils import *
import torch
from datetime import datetime

class DataLogger:
    def __init__(self, env, agent, model="DDQN"):
        name_logging = datetime.now().strftime('%Y%m%d%H%M%S')
        folder_path_train = os.path.join("logs", get_env_name(env), model, name_logging,"train")
        self.folder_path_models = os.path.join("logs", get_env_name(env), model, name_logging, "checkpoints")
        create_folder_path(folder_path_train); create_folder_path(self.folder_path_models)
        
        self.writer = SummaryWriter(folder_path_train)
        
        #Per episodes
        self.sum_reward_per_episode = 0.0
        self.sum_reward_per_n_episodes = 0.0
        self.sum_win_per_n_episodes = 0.0
        self.episode = 0
        self.best_episode_reward = 0.0

        #Per time_steps
        self.average_reward_per_n_time_steps = 0.0
        self.time_step_count = 0

        self.agent = agent

    def episode_step(self, info, n_episodes_average = 5):
        self.episode += 1
        self.sum_reward_per_n_episodes += self.sum_reward_per_episode

        if self.episode % n_episodes_average == 0:
            self.writer.add_scalar('average_reward_per_n_episodes', float(self.sum_reward_per_n_episodes/n_episodes_average), self.episode)
            self.writer.add_scalar('average_wins_per_n_episodes', float(self.sum_win_per_n_episodes/n_episodes_average), self.episode)
            self.sum_reward_per_n_episodes = 0.0
            self.sum_win_per_n_episodes = 0.0
        
        if self.sum_reward_per_episode > self.best_episode_reward:
            save_path = os.path.join(self.folder_path_models, "best_model.pth")
            torch.save(self.agent.net.state_dict(), save_path)
            print(f"Best model saved to {save_path} at episode {self.episode}")
            self.best_episode_reward = self.sum_reward_per_episode

        if info["flag_get"]:
            self.sum_win_per_n_episodes += 1.0
        
        self.sum_reward_per_episode = 0.0
    
    def time_step(self, reward, n_time_steps_save_model=10000):
        self.time_step_count += 1
        self.average_reward_per_n_time_steps += reward
        self.sum_reward_per_episode += reward

        if self.time_step_count % n_time_steps_save_model == 0:
            save_path = os.path.join(self.folder_path_models, f"{self.time_step_count}.pth")
            torch.save(self.agent.net.state_dict(), save_path)
            print(f"Model saved to {save_path} at step {self.time_step_count}")

    def close(self):
        self.writer.close()