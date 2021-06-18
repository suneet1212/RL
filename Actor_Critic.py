# Environment is Lunar Lander
from gym import spaces
from gym.spaces import space
from Deep_Q_Network import NUM_EPISODES
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.multivariate_normal as mnormal
from collections import deque
device = 'cuda' if torch.cuda.is_available() else 'cpu'

###############################################
LR = 0.01
NUM_EPISODES = 10000
GAMMA = 0.99
###############################################

class Actor_Critic(nn.Module):
    def __init__(self):
        super(Actor_Critic,self).__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(24,50),
            nn.ReLU(),
            nn.Linear(50,50),
            nn.ReLU(),
            nn.Linear(50,4),
            nn.Softmax()
        )
    def forward(self, x):
        x = self.linear_stack(x)
        return x

class Agent:
    def __init__(self) -> None:
        self.model = Actor_Critic().to(device)
        self.env = gym.make('LunarLander-v2')
        self.state = self.env.reset()
        self.episode = []
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.G = []
        self.grads = []
        # self.loss = None
        self.total_rewards = 0

    # def 


if __name__ == "__main__":
    agent = Agent()
    deq = deque(maxlen = 100)
    scores = []
    for i in range(NUM_EPISODES):
        total_reward = agent.train_loop()
        print(f"Number of Episodes : {i+1}   Total reward = {total_reward}" )
        deq.append(total_reward)
        scores.append(total_reward)
        if np.mean(deq)>200:
            print(f'\nEnvironment solved in {i-100} episodes!\tAverage Score: {np.mean(deq):.2f}')
            torch.save(agent.model.state_dict(), 'actor_critic_lunar_lander.pth')
            break
    
    x = input()
    if x == 'y':
        torch.save(agent.model.state_dict(), 'actor_critic_lunar_lander.pth')
    