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


if __name__ == '__main__':
    np.random.seed(1)
    args = args_parser()
    PHN = ['iy', 'ih', 'eh', 'ey', 'ae', 'aa', 'aw', 'ay', 'ah', 'ao', 'oy', 'ow', 'uh', 'uw',
           'ux', 'er', 'ax', 'ix', 'arx', 'ax-h']  # 20个元音音素
    # PHN = ['jh', 'ch', 's', 'sh', 'z', 'zh', 'f', 'th', 'v', 'dh', 'b', 'd', 'g', 'p', 't', 'k',
    #        'dx', 'q']  # 摩擦音/破擦音/爆破音
    SOURCE_PATH_PHN = r'../example/si836.phn'
    SOURCE_PATH_WAV = r'../example/si836.wav'

    env = Env(PHN, SOURCE_PATH_WAV, SOURCE_PATH_PHN)
    var = 0.3
    agents_list = [Agent(env.get_s_dim(), env.get_a_dim(), i, args, env.action_space_high()) for i in range(1)]
    M = Memory(capacity=args.memory_capacity, state_dim=env.get_s_dim(), action_dim=env.get_a_dim(), num_agents=1)
    server = Server(env.get_s_dim(), env.get_a_dim(), 1)

    r_max_record = np.zeros(env.get_s_dim(), dtype=float)
    epoch_record = 0
    r_max = 0.0
    reward = -10000.0
    total_reward = []
    total_AVG_threshold = []
    total_TD_error = []

    for t in tqdm(range(10)):
        state = env.reset()
        ep_reward = []  # 记录当前EP的reward
        TD_ERROR = []  # 记录当前EP的TD_ERROR
        asr_time = 0  # 记录当前EP的ASR消耗的时间
        avg_S = np.zeros(env.get_s_dim(), dtype=float)

        for ep in range(args.max_ep_step):
            actions = []
            for i, agent in enumerate(agents_list):
                a = agent.actor.choose_action(agent.observation(state))
                if reward < r_max and M.pointer > args.memory_capacity:
                    for dim in range(0, len(state)):
                        if state[dim] >= 2:
                            a[dim] = -np.abs(np.clip(np.random.normal(a[dim], var), -0.05, 0.05))
                        else:
                            a[dim] = np.clip(np.random.normal(a[dim], var), -0.05, 0.05)
                else:
                    for dim in range(0, len(state)):
                        if state[dim] <= 0:
                            a[dim] = np.abs(np.clip(np.random.normal(a[dim], var), -0.05, 0.05))
                        else:
                            a[dim] = np.clip(np.random.normal(a[dim], var), -0.05, 0.05)

                # a = np.clip(np.random.normal(a, var), -env.action_space_high(),
                #             env.action_space_high())  # add randomness to action selection for exploration
                actions += a.tolist()
            state_, reward, done, asr_time = env.step(state, actions)

            if reward > r_max:
                r_max_record = state
                r_max = reward
                epoch_record = t
                print('\n temp_record_done_r', r_max)
                print('\n temp_record_done_s', r_max_record)
                print('\n temp_record_epoch:', epoch_record)
            ep_reward.append(reward)
            avg_S += state_

            M.store_transition(state, actions, reward, state_)
            state = copy.deepcopy(state_)

            if not (M.pointer <= args.memory_capacity or ep % 2):
                for i, agent in enumerate(agents_list):
                    states, actions, rewards, states_ = M.sample(args.rl_batch_size)
                    server.critic_list[i].optimizer.zero_grad()
                    actions_ = torch.FloatTensor([])
                    for i_, agent_ in enumerate(agents_list):
                        a_ = agent_.target_actor.forward(agent_.observation(states_))
                        actions_ = torch.cat((actions_, a_), 1)
                    y_ = rewards[:, i:i + 1] + args.gamma * server.target_critic_list[i].forward(states_, actions_)
                    y = server.critic_list[i].forward(states, actions)
                    td_error = F.mse_loss(y_.detach(), y)
                    TD_ERROR.append(td_error.detach().numpy())

                    td_error.backward()
                    torch.nn.utils.clip_grad_norm_(server.critic_list[i].parameters(), 0.5)
                    server.critic_list[i].optimizer.step()

                    agent.actor.optimizer.zero_grad()
                    _actions = torch.FloatTensor([])
                    temp = 0
                    for i_, agent_ in enumerate(agents_list):
                        # a = agent_.actor.forward(agent_.observation(states))
                        if i_ == i:
                            temp = agent_.actor.forward(agent_.observation(states))
                            a = temp
                        else:
                            a = actions[:, i_ * agent.action_dim: (i_ + 1) * agent.action_dim]
                        _actions = torch.cat((_actions, a), 1)
                    loss = server.critic_list[i].forward(states, _actions)
                    actor_loss = -torch.mean(loss)
                    actor_loss += (temp ** 2).mean() * 1e-3
                    actor_loss = actor_loss

                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 0.5)
                    agent.actor.optimizer.step()

                """更新target网络"""
                # print('Paras Update')
                for i, agent in enumerate(agents_list):
                    for target_param, param in zip(agent.target_actor.parameters(), agent.actor.parameters()):
                        target_param.data.copy_(target_param.data * (1.0 - args.TAU) + param.data * args.TAU)
                    for target_param, param in zip(server.target_critic_list[i].parameters(),
                                                   server.critic_list[i].parameters()):
                        target_param.data.copy_(target_param.data * (1.0 - args.TAU) + param.data * args.TAU)

                # 保存
                torch.save(agents_list[0].target_actor.state_dict(), '../actor_model')

            if done is True and M.pointer >= args.memory_capacity:
                break
        # if (t + 1) % 5 == 0:

        plt.title('EP-Reward,epoch_{}'.format(t))
        plt.xlabel('steps')
        plt.ylabel('reward value')
        plt.plot(np.array(range(len(ep_reward))), ep_reward)
        plt.savefig(r'..\figure\reward_{}'.format(t))
        plt.show()

        plt.figure()
        plt.title('EP-TD_ERROR,epoch_{}'.format(t))
        plt.xlabel('steps')
        plt.ylabel('td_error value')
        plt.plot(np.array(range(len(TD_ERROR))), TD_ERROR)
        plt.savefig(r'..\figure\TD-Error_{}'.format(t))
        plt.show()

        total_TD_error.append(TD_ERROR)
        total_reward.append(ep_reward)
        total_AVG_threshold.append(avg_S / (ep + 1))

        print('Reward:{} | AVG_Threshold:{}'.format(sum(ep_reward), (avg_S / (ep + 1))))

    plt.figure()
    plt.title('Reward')
    plt.xlabel('epoch')
    plt.ylabel('reward value')
    plt.plot(np.array(range(len(total_reward))), total_reward)
    plt.savefig(r'..\figure\total-reward')
    plt.show()

    np.save(r'../result/TD_ERROR.npy', np.array(total_TD_error))
    np.save(r'../result/Reward.npy', np.array(total_reward))
    np.save(r'../result/AVG.npy', np.array(total_AVG_threshold))
