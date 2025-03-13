import torch 
from torch import nn 
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import random  # Needed for replay sampling
import pandas as pd

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
        self.position = (self.position + 1) % self.capacity  # Circular buffer

    def sample(self, batch_size):
        """Randomly sample a batch"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Define DQN Model
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(n_observations, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Convert state to tensor
def _state_tensor(state):
    return torch.FloatTensor(state).unsqueeze(0).to(device)

# Select Action (ε-Greedy Without Decay)
def select_action1(state,policy_net, epsilon=0.1):
    if np.random.rand() < epsilon:
        return np.random.choice(n_actions)  # Explore
    else:
        with torch.no_grad():
            state_tensor = _state_tensor(state)
            q_values = policy_net(state_tensor)
            return torch.argmax(q_values).item()  # Exploit best action


    
def select_action(state, policy_net, epsilon=0.1, policy_type='epsilon_greedy'):
    """
    Selects an action using an improved ε-greedy policy with probability distribution.
    
    Args:
    - state: Current state of the environment.
    - policy_net: The trained policy network (DQN).
    - epsilon: Exploration rate (probability of choosing a random action).

    Returns:
    - Selected action (int).
    """
    if policy_type == 'greedy':
        state_tensor = _state_tensor(state)
        q_values = policy_net(state_tensor)
        return torch.argmax(q_values).item()
    
    if np.random.rand() < epsilon:
        # Probabilistic action selection instead of pure random
        state_tensor = _state_tensor(state)
        with torch.no_grad():
            q_values = policy_net(state_tensor).cpu().numpy().flatten()  # Convert to NumPy array
        
        # Compute probability distribution: 
        # ε probability is spread evenly, (1 - ε) is concentrated on the best action
        probs = np.ones(len(q_values)) * (epsilon / len(q_values))
        best_action = np.argmax(q_values)  # Action with highest Q-value
        probs[best_action] += (1 - epsilon)  # Higher probability for the best action

        # Sample from the probability distribution
        action = np.random.choice(len(q_values), p=probs)
        
    else:
        # Greedy action selection (Exploitation)
        with torch.no_grad():
            state_tensor = _state_tensor(state)
            q_values = policy_net(state_tensor)
            action = torch.argmax(q_values).item()

    return action
     
def evaluate(policy_net,eval_env,n_eval_episodes=30):
    returns = []  # list to store the reward per episode
  
    for i in range(n_eval_episodes):
        done = False
        s,_ = eval_env.reset()
        R_ep = 0
        while not done:
            a = select_action(s,policy_net,_, 'greedy')
            s_prime, r, terminated, truncated, _  = eval_env.step(a)
            done = terminated or truncated
            R_ep += r
            if done:
                break
            else:
                s = s_prime
        
        returns.append(R_ep)
        
    mean_return = np.mean(returns)
    print(f"Evaluation return: {mean_return}") 
    return mean_return

# Train DQN with or without Replay Memory
def train_dqn(policy_net, optimizer, batch_size, gamma, target_net, use_replay=False, use_target_net=False, target_steps=1000):
    replay_memory = ReplayMemory(1000000) if use_replay else None  # Only use if enabled
    loss_function = nn.MSELoss()
    total_timesteps = 0  # Track total timesteps across all episodes
    eval_timesteps = []
    eval_returns = []
    while total_timesteps < max_timesteps:
        
        state, _ = env.reset()
        done = False
        total_reward = 0
      
        while not done:
            action = select_action(state, policy_net)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated  # End episode if either flag is True

            if use_replay:
                replay_memory.push(state, action, next_state, reward, done)  # Store in buffer

            # Convert states to tensors
            state_tensor = _state_tensor(state)
            next_state_tensor = _state_tensor(next_state)

            # Compute target Q-value using Bellman equation
            with torch.no_grad():
                if use_target_net:
                    q_value_next_state = target_net(next_state_tensor).max(1)[0]  # max_a Q(s', a) from target net
                else:
                    q_value_next_state = policy_net(next_state_tensor).max(1)[0]  # max_a Q(s', a) from policy net
                target_q_value = reward + gamma * q_value_next_state * (1 - done)

            # Get predicted Q-value for the taken action
            q_values = policy_net(state_tensor)
            q_value = q_values.gather(1, torch.LongTensor([[action]], device=device)).squeeze(1)

            # Compute loss & update model
            loss = loss_function(q_value, target_q_value)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
            state = next_state
            total_timesteps += 1
            total_reward += reward
            # Train with Replay Buffer (if enabled)
            if use_replay and len(replay_memory) > batch_size:
                batch = replay_memory.sample(batch_size)
                states, actions, next_states, rewards, dones = zip(*batch)

                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).unsqueeze(1).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                dones = torch.FloatTensor(dones).to(device)

                # Compute max Q-values for next states
                with torch.no_grad():
                    if use_target_net:
                        next_q_values = target_net(next_states).max(1)[0]
                    else:
                        next_q_values = policy_net(next_states).max(1)[0]
                    target_q_values = rewards + gamma * next_q_values * (1 - dones)

                # Compute predicted Q-values for chosen actions
                q_values = policy_net(states).gather(1, actions).squeeze(1)

                # Compute loss and update
                loss = loss_function(q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Update target network every target_steps steps
            if use_target_net and total_timesteps % target_steps == 0:
                target_net.load_state_dict(policy_net.state_dict())
                print("Target Net Updated")

        # Store episode reward   
        print(f'Total Reward In training: {total_reward},  Total Timesteps: {total_timesteps}')

        
        if total_timesteps % 100 == 0:
            print("Now lets test on real environment")
            eval_returns.append(evaluate(policy_net,env))
            eval_timesteps.append(total_timesteps)
            
            
    env.close()
    
    
    return policy_net, eval_returns, eval_timesteps



# Initialize Environment


# Training settings
max_timesteps = 100000
batch_size = 50
gamma = 0.99
use_replay = False  # Set to True for Experience Replay
use_target_net = False  # Set to True for Fixed Q-Targets
target_steps = 1000  # Update target network every n steps

# Train model
for rep in range(5):
    env = gym.make("CartPole-v1")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Using Device:", device)


    
    # Initialize parameters

    n_observations_state = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy_net = DQN(n_observations_state, n_actions).to(device)
    
    target_net = DQN(n_observations_state, n_actions).to(device)
    
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=0.0001)
    policy_net, eval_returns, eval_timesteps= train_dqn(policy_net, optimizer, batch_size, gamma, target_net, use_replay, use_target_net, target_steps)

    print(eval_returns)
    print(eval_timesteps)
    print("Training Complete")
    df = pd.DataFrame({'Timestep': eval_timesteps, 'Evaluation Return': eval_returns})
    df.to_csv(f"results_{rep}.csv", index=False)

 
# Plot Results
plt.figure(figsize=(8, 5))
plt.plot(eval_timesteps, eval_returns, label="Average Rewards across timesteps")
plt.xlabel('Timesteps')
plt.ylabel('Total Reward')
plt.title('Reward attained across Timesteps')
plt.legend()
plt.show()

# add stopping rule
# add mean coverage