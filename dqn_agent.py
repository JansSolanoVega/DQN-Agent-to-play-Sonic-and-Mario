#Code based on https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html 
import torch
import numpy as np
from utils import *
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from network import SonicNet

class SonicAgent:
    def __init__(self, exploration_rate, discount_factor, loss_fn, optimizer, double_dqn=True):
        #General
        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(100000, device=torch.device("cpu")))
        self.batch_size = 64
        self.step = 0

        #DQN params
        self.double_dqn = double_dqn
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.999
        
        self.update_target_from_online_every = 1e4
        self.update_online_every = 4
        self.start_learning_after = 1e4

        #NN
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.net = SonicNet()

    def act(self, state):
        if np.random.randn() < self.epsilon:
            action_idx = np.random.randint(0, len(POSSIBLE_ACTIONS))#exploration
        else:
            state_action_values = self.sonicNet(state)#NN outputs Q(s,a) values for all actions from state s
            action_idx = np.argmax(state_action_values)#exploitation
        
        #GLIE:
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

        self.step += 1
        return action_idx
    
    def append_experience_to_memory(self, state, next_state, action, reward, done):
        state = torch.tensor(state)
        next_state = torch.tensor(next_state)
        action = torch.tensor([action])
        reward = torch.tensor([reward])
        done = torch.tensor([done])

        self.memory.add(TensorDict({"state": state, "next_state": next_state, "action": action, "reward": reward, "done": done}))

    def sample_experience_from_memory(self):
        batch = self.memory.sample(self.batch_size).to(self.device)
        state, next_state, action, reward, done = (batch.get(key) for key in ("state", "next_state", "action", "reward", "done"))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def estimate(self, state, action):
        return self.net(state, "online")[action]
    
    @torch.no_grad()
    def target(self, next_state, reward):
        if self.double_dqn:
            greedy_policy_action = torch.argmax(self.net(next_state, "online"))
        else:
            greedy_policy_action = torch.argmax(self.net(next_state, "target"))
        
        return reward + self.gamma*self.net(next_state, "target")[greedy_policy_action] 

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