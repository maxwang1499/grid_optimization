import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
from copy import deepcopy

class QNetwork(nn.Module):
    """Implements a Q-network to approximate the Q-values"""
    def __init__(self, input_size, hidden_size, num_cats):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.outs = nn.ModuleList([nn.Linear(hidden_size, out_size) for out_size in num_cats])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        output = []
        for net in self.outs:
            output.append(net(x).argmax(axis=-1))
        return torch.stack(output, dim=-1)

class DQNAgent:
    """Implements a DQN agent to choose an action from a Q-network"""
    def __init__(self, env, state_size, hidden_size, num_cats, q_network=None):
        self.env = deepcopy(env)
        self.state_size = state_size
        self.num_cats = deepcopy(num_cats)
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        if q_network is None :
            self.q_network = QNetwork(state_size, hidden_size, num_cats)
        else :
            self.q_network = q_network
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad() :
                state = torch.from_numpy(state).float().unsqueeze(0)
                q_values = self.q_network(state)
                return np.argmax(q_values.detach().numpy())

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.from_numpy(np.array(states)).float()
        actions = torch.from_numpy(np.array(actions)).long()
        rewards = torch.from_numpy(np.array(rewards)).float()
        next_states = torch.from_numpy(np.array(next_states)).float()
        dones = torch.from_numpy(np.array(dones, dtype=np.uint8)).float()
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.q_network(next_states).max(1)[0]
        expected_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        loss = nn.functional.smooth_l1_loss(q_values, expected_q_values.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, episodes):
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                self.replay()

class DoubleDQNAgent(DQNAgent):
    """Implements a DoubleDQN agent to choose an action from a Q-network"""
    def __init__(self, env, state_size, hidden_size, num_cats, q_network=None):
        super().__init__(env, state_size, hidden_size, num_cats, q_network)

        if q_network is None :
            self.q_network = QNetwork(state_size, hidden_size, num_cats)
        else :
            self.q_network = q_network
        self.target_q_network = deepcopy(q_network)
        self.target_update_interval = 100
        self.steps = 0

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.from_numpy(np.array(states)).float()
        actions = torch.from_numpy(np.array(actions)).long()
        rewards = torch.from_numpy(np.array(rewards)).float()
        next_states = torch.from_numpy(np.array(next_states)).float()
        dones = torch.from_numpy(np.array(dones, dtype=np.uint8)).float()

        # Compute the Q-values using the online Q-network
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute the argmax action using the online Q-network
        argmax_actions = self.q_network(next_states).argmax(1)

        # Compute the Q-values using the target Q-network
        target_q_values = self.target_q_network(next_states).gather(1, argmax_actions.unsqueeze(1)).squeeze(1)

        # Compute the expected Q-values
        expected_q_values = rewards + (1 - dones) * self.gamma * target_q_values

        # Update the online Q-network
        loss = nn.functional.smooth_l1_loss(q_values, expected_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Update the target Q-network
        if self.steps % self.target_update_interval == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

        self.steps += 1
