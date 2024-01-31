from torch.utils.tensorboard import SummaryWriter
import os
from utils import *
import torch
from datetime import datetime
import pickle

class DataLogger:
    def __init__(self, env, hp, agent=None, model=None):
        name_logging = datetime.now().strftime('%Y%m%d%H%M%S')
        self.folder_path_train = os.path.join("logs", get_env_name(env), model, name_logging,"train")
        self.folder_path_models = os.path.join("logs", get_env_name(env), model, name_logging, "checkpoints")
        create_folder_path(self.folder_path_train); create_folder_path(self.folder_path_models)
        save_hyperparameters(hp, self.folder_path_train)
        
        if agent is not None:
            self.writer = SummaryWriter(self.folder_path_train)
        
        #Per episodes
        self.sum_reward_per_episode = 0.0
        self.sum_loss_per_episode = 0.0
        self.win_per_episode = 0.0
        self.episode = 0
        self.best_episode_reward = 0.0

        #Per time_steps
        self.time_step_count = 0

        self.agent = agent

        #Save
        self.total_rewards = []
        self.wins = []

    def episode_step(self, info):
        self.episode += 1
        self.win_per_episode += float(info["flag_get"])
        
        self.total_rewards.append(self.sum_reward_per_episode)
        self.wins.append(self.win_per_episode)

        self.writer.add_scalar('mean_rew', float(self.sum_reward_per_episode), self.episode)
        self.writer.add_scalar('win_rate', float(self.win_per_episode), self.episode)
        self.writer.add_scalar('online_loss', float(self.sum_loss_per_episode), self.episode)
        
        if self.sum_reward_per_episode > self.best_episode_reward:
            save_path = os.path.join(self.folder_path_models, "best_model.pth")
            torch.save({"model": self.agent.net.state_dict(), "exploration_rate": self.agent.epsilon}, save_path)
            print(f"Best model saved to {save_path} at episode {self.episode}")
            self.best_episode_reward = self.sum_reward_per_episode
        
        self.sum_reward_per_episode = 0.0
        self.sum_loss_per_episode = 0.0
        self.win_per_episode = 0.0
    
    def time_step(self, reward, loss, n_time_steps_save_model=10000):
        self.sum_reward_per_episode += reward
        if loss:
            self.sum_loss_per_episode += loss

        if self.time_step_count % n_time_steps_save_model == 0:
            save_path = os.path.join(self.folder_path_models, f"{self.time_step_count}.pth")
            torch.save({"model": self.agent.net.state_dict(), "exploration_rate": self.agent.epsilon}, save_path)
            print(f"Model saved to {save_path} at step {self.time_step_count}")

        self.writer.add_scalar('exploration_rate', float(self.agent.epsilon), self.time_step_count)
        
        self.time_step_count += 1

    def close(self):
        with open(os.path.join(self.folder_path_train, "total_rewards"), "wb") as f:
            pickle.dump(self.total_rewards, f)
        with open(os.path.join(self.folder_path_train, "total_wins"), "wb") as f:
            pickle.dump(self.wins, f)
        
        if self.agent is not None:
            self.writer.close()