import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from options import args_parser
# from FL.FL import Federate_learing as FL
from tqdm import tqdm
import random
import json
from Env import Env
import matplotlib.pyplot as plt


# from FL.datasets.get_data import *


class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, max_action=1):
        super(ActorNetwork, self).__init__()

        self.l1 = nn.Linear(state_dim, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, action_dim)
        self.max_action = max_action
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        # self.layer_norm1 = nn.LayerNorm(normalized_shape=state_dim // 2, eps=0, elementwise_affine=False)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=action_dim, eps=0, elementwise_affine=False)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        x = self.layer_norm2(x)
        x = torch.tanh(x)
        x = x / 20
        # """将输出限定到[0,2]"""
        # x = x + 1
        return x

    def choose_action(self, s):
        s = torch.FloatTensor(s)
        s = s.unsqueeze(0)
        s = self.forward(s)
        s = s.squeeze()
        s = s.detach().cpu()
        return s  # single action


class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticNetwork, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x, u):
        temp_ = torch.cat((x, u), 1)
        temp = self.l1(temp_)
        x = F.relu(temp)
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class Memory(object):
    def __init__(self, capacity, state_dim, action_dim, num_agents):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.dims = 2 * state_dim + action_dim + self.num_agents
        self.data = np.zeros((capacity, self.dims))
        self.pointer = 0

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, r, s_))
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= n, 'Memory has not been fulfilled'
        indices = np.random.choice(min(self.capacity, self.pointer), size=n)
        b_M = self.data[indices, :]
        b_s = b_M[:, :self.state_dim]
        b_a = b_M[:, self.state_dim: self.state_dim + self.action_dim]
        b_r = b_M[:, -self.state_dim - self.num_agents: -self.state_dim]
        b_s_ = b_M[:, -self.state_dim:]

        states = torch.FloatTensor(b_s)  # 转换成tensor类型
        actions = torch.FloatTensor(b_a)
        rewards = torch.FloatTensor(b_r)
        states_ = torch.FloatTensor(b_s_)
        return states, actions, rewards, states_


class Agent:
    def __init__(self, state_dim, action_dim, id, args, max_action):
        self.id = id
        self.target_actor = ActorNetwork(state_dim, action_dim, max_action=max_action)
        self.actor = ActorNetwork(state_dim, action_dim, max_action=max_action)
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.args = args

    # 用于FL
    def observation(self, state):
        if type(state) == list:
            state = copy.copy(state)
        else:
            state = state.clone()
        return state


class Server:
    def __init__(self, state_dim, action_dim, num_of_agents):
        self.critic_list = [CriticNetwork(state_dim, action_dim) for i in range(num_of_agents)]
        self.target_critic_list = [CriticNetwork(state_dim, action_dim) for i in range(num_of_agents)]
        self.num_of_agents = num_of_agents
        self.state_dim = state_dim
        self.action_dim = action_dim

#
# args = args_parser()


PHN = ['iy', 'ih', 'eh', 'ey', 'ae', 'aa', 'aw', 'ay', 'ah', 'ao', 'oy', 'ow', 'uh', 'uw',
           'ux', 'er', 'ax', 'ix', 'arx', 'ax-h']  # 20个元音音素
SOURCE_PATH_PHN = r'../example/si836.phn'
SOURCE_PATH_WAV = r'../example/si836.wav'

env = Env(PHN, SOURCE_PATH_WAV, SOURCE_PATH_PHN)
var = 0.3


model = ActorNetwork(env.get_s_dim(), env.get_a_dim(), max_action=1)
model.load_state_dict(torch.load(r'../actor_model_deepspeech'))
# # model = torch.load(r'../actor_model')
# # model = nn.Module.load_state_dict(state_dict, strict=True)
# model.eval()
ddpg = model
#
# # test
# # ddpg.load_ckpt()
record = []
threshold = []
for i_ in range(10):
        s = env.reset()
        for i in range(20):
            a = ddpg.choose_action(s).numpy()
            a = np.clip(np.random.normal(a, 0.3), 0, 2)
            s, r, done, info = env.step(a)
            if done is True:
                print('\n Reward:{} | Step:{} | Epoch:{} '.format(r, i, i_))
                record.append([r, i, i_])
                threshold.append(a)
                break

threshold = np.array(threshold)

mean_threshold = np.zeros(shape=len(threshold[0]))
for col in range(len(threshold[0])):
    for row in range(len(threshold)):
        cnt = 0
        temp_threshold = 0
        if threshold[row][col] != 0:
            temp_threshold += threshold[row][col]
            cnt += 1
    if cnt == 0:
        mean_threshold[col] = 0.0
    else:
        mean_threshold[col] = temp_threshold / cnt

print(mean_threshold)

# for j in range(len(record)):
#     print('\n Reward:{} | Step:{} | Epoch:{} '.format(record[j][0], record[j][1], record[j][2]))


# reward = np.load(r'../result/Reward.npy')
# ep_reward = []
# for j in range(len(reward)):
#     ep_reward.append(np.sum(reward[j]))
#
# plt.figure(figsize=(10, 5))
# plt.title('EP-Reward')
# plt.xlabel('epoch')
# plt.ylabel('reward value')
# plt.plot(np.array(range(len(ep_reward))), ep_reward)
# plt.show()
# for i in range(20):
#     for j in range(len(reward[0])):
#         if reward[i][j] >= 60:
#             print('\n Reward:{} | Step:{} '.format(reward[i][j], j))
#             break