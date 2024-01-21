#Code based on https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html 
import torch
import numpy as np
from utils import *
from network import Net
from collections import deque
import random

class Agent:
    def __init__(self, env, batch_size, learning_rate, observation_size, action_size, exploration_rate, discount_factor, model="DDQN", epsilon_min = 0.1, epsilon_decay = 0.99999975):
        #General
        self.memory = deque(maxlen=40000)
        self.batch_size = batch_size
        self.step = 0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.env = env

        #DQN params
        self.double_dqn = (model=="DDQN")
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        self.update_target_from_online_every = 1e4
        self.update_online_every = 3
        self.start_learning_after = 5e4

        #NN
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.net = Net(observation_size, action_size)
        self.net = self.net.to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)

    def act(self, state):
        if np.random.randn() <= self.epsilon:
            action_idx = np.random.randint(0, get_action_space_size(self.env))#exploration
        else:
            state = torch.tensor(state.__array__(), device=self.device).unsqueeze(0)
            state_action_values = self.net(state, model="online")#NN outputs Q(s,a) values for all actions from state s
            action_idx = torch.argmax(state_action_values[0]).item()#exploitation
        
        #GLIE:
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

        self.step += 1
        return action_idx
    
    def append_experience_to_memory(self, state, next_state, action, reward, done):
        state = torch.tensor(state.__array__(), device=self.device)
        next_state = torch.tensor(next_state.__array__(), device=self.device)
        action = torch.tensor([action], device=self.device)
        reward = torch.tensor([reward], device=self.device)
        done = torch.tensor([done], device=self.device)

        self.memory.append((state, next_state, action, reward, done, ))

    def sample_experience_from_memory(self):
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def estimate(self, state, action):
        return self.net(state, "online")[np.arange(0, self.batch_size), action]
    
    @torch.no_grad()
    def target(self, next_state, reward):
        if self.double_dqn:
            greedy_policy_action = torch.argmax(self.net(next_state, "online"), axis=1)
        else:
            greedy_policy_action = torch.argmax(self.net(next_state, "target"), axis=1)
        
        return reward + self.gamma*self.net(next_state, "target")[np.arange(0, self.batch_size), greedy_policy_action] 

    def update_weights_online(self, estimate, target):#Gradient descent
        loss = self.loss_fn(estimate, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_weights_target(self):#Update target network weights periodically using online network weights
        self.net.target.load_state_dict(self.net.online.state_dict())
    
    def learn(self):
        if self.step < self.start_learning_after or self.step % self.update_online_every!=0:
            return None
        if self.step % self.update_target_from_online_every==0:
            self.update_weights_target()
        state, next_state, action, reward, done = self.sample_experience_from_memory()
        estimate = self.estimate(state, action)
        target = self.target(next_state, reward)
        loss = self.update_weights_online(estimate, target)
        return loss