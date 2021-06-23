# Environment is Lunar Lander
from gym import spaces
from gym.spaces import space
from torch.nn.modules.activation import ReLU, Softmax
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.categorical as c
from collections import deque
device = 'cuda' if torch.cuda.is_available() else 'cpu'

###############################################
LR = 0.1
NUM_EPISODES = 10000
GAMMA = 0.99
###############################################

# class Actor(nn.Module):
#     def __init__(self):
#         super(Actor,self).__init__()
#         self.linear_stack = nn.Sequential(
#             nn.Linear(8,10),
#             nn.ReLU(),
#             nn.Linear(10,10),
#             nn.ReLU(),
#             nn.Linear(10,4),
#             nn.Softmax()
#         )
#     def forward(self, x):
#         x = self.linear_stack(x)
#         return x

# class Critic(nn.Module):
#     def __init__(self):
#         super(Critic, self).__init__()
#         self.linear_stack = nn.Sequential(
#             nn.Linear(8,10),
#             nn.ReLU(),
#             nn.Linear(10,10),
#             nn.ReLU(),
#             nn.Linear(10,4)
#         )

#     def forward(self, x):
#         x = self.linear_stack(x)
#         return x

class Actor_critic(nn.Module):
    def __init__(self):
        super(Actor_critic,self).__init__()
        self.actor_linear_stack = nn.Sequential(
            nn.Linear(8,50),
            nn.ReLU(),
            nn.Linear(50,50),
            nn.ReLU(),
            nn.Linear(50,50),
            nn.ReLU(),
            nn.Linear(50,50),
            nn.ReLU(),
            nn.Linear(50,50),
            nn.ReLU(),
            nn.Linear(50,4),
            nn.Softmax()
        )
        self.critic_linear_stack = nn.Sequential(
            nn.Linear(8,60),
            nn.ReLU(),
            nn.Linear(60,10),
            nn.ReLU(),
            nn.Linear(10,4)
        )
    def forward(self,x):
        actor = self.actor_linear_stack(x)
        critic = self.critic_linear_stack(x)
        return actor, critic

class Agent:
    def __init__(self) -> None:
        # self.actor = Actor().to(device)
        # self.critic = Critic().to(device)
        self.model = Actor_critic().to(device)
        self.env = gym.make('LunarLander-v2')
        self.state = self.env.reset()
        # self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=LR)
        # self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=LR)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.total_rewards = 0
    
    def train_loop(self):
        probs, q = (self.model((torch.from_numpy(self.state).unsqueeze(0).float().to(device))))
        probs = torch.squeeze(probs)
        m = c.Categorical(probs)
        action = m.sample()
        next_state, reward, done, _ = self.env.step(action.item())
        self.env.render()
        probs2, q_prime = self.model(torch.from_numpy(next_state).unsqueeze(0).float().to(device))
        probs2 = torch.squeeze(probs2)
        # q = self.critic(torch.from_numpy(self.state).unsqueeze(0).float().to(device))
        # probs2 = (self.actor((torch.from_numpy(next_state).unsqueeze(0)).float().to(device)))
        m2 = c.Categorical(probs2)
        action2 = m2.sample().item()
        if done == True:
            q_prime[0][action2] = 0
        td_error = reward + GAMMA*q_prime[0][action2] - q[0][action]
        loss_a = -m.log_prob(action) * td_error
        loss_obj = nn.MSELoss()
        loss_c = -loss_obj(torch.tensor([reward], device=device).float() + GAMMA*q_prime[0][action2].float(), q[0][action].float())
        loss = loss_a + loss_c
        # self.optimizer_actor.zero_grad()
        # self.optimizer_critic.zero_grad()
        self.optimizer.zero_grad()
        # loss_a.backward()
        # loss_c.backward()
        loss.backward()
        # self.optimizer_actor.step()
        # self.optimizer_critic.step()
        self.optimizer.step()

        # update next state
        if done == True:
            self.state = self.env.reset()
        else:
            self.state = next_state
        return done, reward  

if __name__ == "__main__":
    agent = Agent()
    # deq = deque(maxlen = 100)
    # scores = []
    for i in range(NUM_EPISODES):
        done = False
        total_reward = 0
        while not done:
            done, reward = agent.train_loop()
            total_reward += reward
        print(f"Number of Episodes : {i+1}   Total reward = {total_reward}")    