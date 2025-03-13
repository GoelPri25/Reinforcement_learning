# Re-import necessary libraries after execution state reset
import torch 
from torch import nn 
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import random  
import pandas as pd
import itertools
import os
import time
from scipy.signal import savgol_filter


# Replay Memory Class
class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, next_state, reward, done):
        """Store experience in replay memory"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, next_state, reward, done)
        self.position = (self.position + 1) % self.capacity  

    def sample(self, batch_size):
        """Randomly sample a batch"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Convert state to tensor
def _state_tensor(state, device):
    return torch.FloatTensor(state).unsqueeze(0).to(device)

# Select Action (ε-Greedy)
def select_action(state, policy_net, epsilon=0.5, policy_type='epsilon_greedy', device='cpu'):
    if policy_type == 'greedy':
        state_tensor = _state_tensor(state, device)
        q_values = policy_net(state_tensor)
        return torch.argmax(q_values).item()
    
    if np.random.rand() < epsilon:
        state_tensor = _state_tensor(state, device)
        with torch.no_grad():
            q_values = policy_net(state_tensor).cpu().numpy().flatten()
        
        probs = np.ones(len(q_values)) * (epsilon / len(q_values))
        best_action = np.argmax(q_values)
        probs[best_action] += (1 - epsilon)

        action = np.random.choice(len(q_values), p=probs)
        
    else:
        with torch.no_grad():
            state_tensor = _state_tensor(state, device)
            q_values = policy_net(state_tensor)
            action = torch.argmax(q_values).item()

    return action

def evaluate(policy_net,device, n_eval_episodes=30):
    
    eval_env = gym.make("CartPole-v1")
    returns = []  # list to store the reward per episode
  
    for i in range(n_eval_episodes):
        done = False
        s,_ = eval_env.reset()
        R_ep = 0
        while not done:
            a = select_action(s,policy_net,_, 'greedy', device)
            s_prime, r, terminated, truncated, _  = eval_env.step(a)
            done = terminated or truncated
            R_ep += r
            if done:
                break
            else:
                s = s_prime
        
        returns.append(R_ep)
        
    mean_return = np.mean(returns)
    #print(f"Evaluation return: {mean_return}") 
    return mean_return

# Train DQN with batch updates
def train_dqn(env, policy_net, optimizer, batch_size, gamma, target_net, use_replay, use_target_net, target_steps, update_interval, max_timesteps, device):

    replay_memory = ReplayMemory(100000) if use_replay else None  
    loss_function = nn.MSELoss()
    total_timesteps = 0  
    eval_timesteps = []
    eval_returns = []
    temp_buffer = []  
    eval_interval = 50
    

    while total_timesteps <= max_timesteps:
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done and total_timesteps <= max_timesteps:
            action = select_action(state, policy_net, device=device)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated  

            temp_buffer.append((state, action, next_state, reward, done))

            if len(temp_buffer) == update_interval:
                states, actions, next_states, rewards, dones = zip(*temp_buffer)

                # states = torch.FloatTensor(states).to(device)
                # actions = torch.LongTensor(actions).unsqueeze(1).to(device)
                # next_states = torch.FloatTensor(next_states).to(device)
                # rewards = torch.FloatTensor(rewards).to(device)
                # dones = torch.FloatTensor(dones).to(device)
                states = torch.FloatTensor(np.array(states)).to(device)
                actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(device)
                next_states = torch.FloatTensor(np.array(next_states)).to(device)
                rewards = torch.FloatTensor(np.array(rewards)).to(device)
                dones = torch.FloatTensor(np.array(dones)).to(device)

                with torch.no_grad():
                    next_q_values = policy_net(next_states).max(1)[0]
                    target_q_values = rewards + gamma * next_q_values * (1 - dones)

                q_values = policy_net(states).gather(1, actions).squeeze(1)

                loss = loss_function(q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                temp_buffer = []  

            state = next_state
            total_timesteps += 1
            

            if use_target_net and total_timesteps % target_steps == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if total_timesteps % eval_interval == 0:
                avg_return = evaluate(policy_net,  device = device)
                eval_returns.append(avg_return)
                eval_timesteps.append(total_timesteps)
                print(f"[Eval] Timestep: {total_timesteps}, Avg Return: {avg_return}")


    env.close()
    return policy_net, eval_returns, eval_timesteps

# Define hyperparameter values to test
learning_rates = [0.01, 0.001, 0.0005]
network_sizes = [(128, 128), (256, 256), (64, 64)]
update_intervals = [5, 10, 15]
epsilons = [0.5, 0.1, 0.05]

# learning_rates = [0.0005]
# network_sizes = [(128, 128)]
# update_intervals = [5]
# epsilons = [0.5]

hyperparameter_combinations = list(itertools.product(learning_rates, network_sizes, update_intervals, epsilons))
num_repetitions = 5  

results_dir = "dqn_results"
os.makedirs(results_dir, exist_ok=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Function to smooth results AFTER averaging
def smooth(y, window, poly=2):
    return savgol_filter(y, window, poly)

# Function to average over multiple runs of a given setting
def average_over_repetitions(n_repetitions,  batch_size, gamma, use_replay, use_target_net, 
                             target_steps, update_interval, max_timesteps, device, smoothing_window=5):
    
    returns_over_repetitions = []
    
    start_time = time.time()

    for rep in range(n_repetitions):
        print(f"Running  for one setting...")

        env = gym.make("CartPole-v1")
        n_observations_state = env.observation_space.shape[0]
        n_actions = env.action_space.n
        policy_net = DQN(n_observations_state, n_actions).to(device)
        target_net = DQN(n_observations_state, n_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())

        optimizer = optim.Adam(policy_net.parameters(), lr=lr)

        _, eval_returns, eval_timesteps = train_dqn(env,
            policy_net, optimizer, batch_size, gamma, target_net, use_replay, use_target_net, 
            target_steps, update_interval, max_timesteps, device
        )
        returns_over_repetitions.append(eval_returns)

    print(f'One setting took {(time.time() - start_time)/60:.2f} minutes')

    # Compute the mean learning curve over repetitions
    
    
    
    learning_curve = np.mean(np.array(returns_over_repetitions), axis=0)
    # Apply smoothing **AFTER computing the mean**
    if smoothing_window is not None:
        learning_curve = smooth(learning_curve, smoothing_window)

    return learning_curve, eval_timesteps


for lr, (hidden1, hidden2), update_int, eps in hyperparameter_combinations:
    print(f"Running experiment for LR={lr}, Net={hidden1}-{hidden2}, UpdateInt={update_int}, Eps={eps}")
    # env = gym.make("CartPole-v1")
  
    # n_observations_state = env.observation_space.shape[0]
    # n_actions = env.action_space.n

    # Define DQN model dynamically
    class DQN(nn.Module):
        def __init__(self, n_observations, n_actions):
            super(DQN, self).__init__()
            self.fc1 = nn.Linear(n_observations, hidden1)
            self.fc2 = nn.Linear(hidden1, hidden2)
            self.fc3 = nn.Linear(hidden2, n_actions)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)

    
    # Run multiple repetitions and compute mean performance
    learning_curve, eval_timesteps = average_over_repetitions(
        n_repetitions=num_repetitions, batch_size=64, 
        gamma=0.99,  use_replay=False, use_target_net=False, target_steps=1000, 
        update_interval=update_int, max_timesteps=100000, device=device, smoothing_window=None
    )

    # Store results
    df = pd.DataFrame({'Timestep': eval_timesteps, 'Mean Evaluation Return': learning_curve})

    filename = f"results_lr{lr}_net{hidden1}-{hidden2}_update{update_int}_eps{eps}.csv"
    filepath = os.path.join(results_dir, filename)
    df.to_csv(filepath, index=False)

    print(f"Saved averaged results for LR={lr}, Net={hidden1}-{hidden2}, UpdateInt={update_int}, Eps={eps}")

print("All experiments completed!")

