import torch
import torch.nn as nn
from collections import deque, namedtuple
import random
torch.set_default_dtype(torch.float64)


def rewardFunction(state, Cov, action, Q_lqr, R_lqr, max_stageCost):
    stageCost = state.T @ Q_lqr @ state + action.T @ R_lqr @ action + (Q_lqr @ Cov).trace() 
    reward = -stageCost.clip(-max_stageCost, max_stageCost) / max_stageCost 
    return torch.atleast_2d(reward).detach() 

class FeedForward_NN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.Sequential = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.Sequential(x) * 20
    

Transitions = namedtuple('Transitions', ('state', 'action', 'reward', 'next_state', 'time_step_k'))

class ReplayBuffer(deque):
    def __init__(self, Buffer_size, device):
        super().__init__([], maxlen=Buffer_size)
        self.device = device
    def push(self, *args):
        self.append(Transitions(*args))
    def GET(self):
        transitions_batch = self
        unzipped_batch = Transitions(*zip(*transitions_batch))
        states = torch.cat(unzipped_batch.state, dim = 0).detach().to(self.device)
        actions = torch.cat(unzipped_batch.action, dim = 0).detach().to(self.device)
        rewards = torch.cat(unzipped_batch.reward, dim = 0).detach().to(self.device)
        next_states = torch.cat(unzipped_batch.next_state, dim = 0).detach().to(self.device)
        time_step_k = torch.cat(unzipped_batch.time_step_k, dim = 0).detach().to(self.device)

        return states, actions, rewards, next_states, time_step_k