import torch
import numpy as np
from model import PNetwork, VNetwork

class ActorCriticAgent:
    def __init__(self, state_dim, action_dim, lr_vc, lr_pa, hidden_layers_vc, hidden_layers_pa, n_steps = 5, no_concurrent_envs=1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_steps = n_steps
        self.gamma = 0.99

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        print(f"Device for processing: {self.device}")

        self.actor_network = PNetwork(state_dim, action_dim, hidden_layers_pa).to(self.device)
        self.critic_network = VNetwork(state_dim, hidden_layers_vc).to(self.device)
        
        self.optimizer_actor = torch.optim.Adam(self.actor_network.parameters(), lr=lr_pa)
        self.optimizer_critic = torch.optim.Adam(self.critic_network.parameters(), lr=lr_vc)
        self.criterion = torch.nn.MSELoss()

    def select_action(self, state, type="explore"):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        model_output = self.actor_network(state)
        if type == "greedy":
            return torch.argmax(model_output, dim=-1).item()
        
        dist = torch.distributions.Categorical(logits=model_output)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update(self, all_buffer, advantage=False):
        """Updates after n-step returns. Buffer stores last n transitions"""
        total_critic_loss = 0
        total_actor_loss = 0
        counter = 0
        for each_buffer in all_buffer:
            for i in range(len(each_buffer['states'])):
                end = min(i+self.n_steps, len(each_buffer['states']))
                states = torch.FloatTensor(np.array(each_buffer['states'][i:end])).to(self.device)
                rewards = torch.FloatTensor(each_buffer['rewards'][i:end]).to(self.device)
                dones = torch.FloatTensor(each_buffer['dones'][i:end]).to(self.device)
                next_states = torch.FloatTensor(np.array(each_buffer['next_states'][i:end])).to(self.device)

                log_prob = each_buffer['log_probs'][i]

                # Compute n-step returns
                Q_t = 0
                for k in range(len(rewards)):
                    Q_t += (self.gamma ** k) * rewards[k]

                # if not terminal, bootstrap from Value network
                if not dones[-1]:
                    with torch.no_grad():
                        Q_t += (self.gamma ** len(dones)) * self.critic_network(next_states[-1])

                value = self.critic_network(states[0])

                # Critic Update MSE with Q_t and V(s) of the corresponding start states
                critic_loss = self.criterion(Q_t, value)
                # critic_loss = (n_step_return.detach() - value)** 2
                # critic_loss = torch.mean(critic_loss)
                total_critic_loss += critic_loss

                # Actor loss
                if advantage:
                    actor_loss =  -torch.sum(log_prob * (Q_t - value).detach())
                else:
                    actor_loss =  -torch.sum(log_prob * Q_t)

                total_actor_loss += actor_loss
                counter += 1
        # Normalize losses
        # total_critic_loss /= counter
        # total_actor_loss /= counter

        self.optimizer_critic.zero_grad()
        self.optimizer_actor.zero_grad()
        total_critic_loss.backward(retain_graph=True)
        total_actor_loss.backward()
        self.optimizer_critic.step()

        self.optimizer_actor.step()

    def evaluate(self, evaluate_env):
        returns = []
        for _ in range(10):
            evaluate_done = False
            evaluate_failed = False
            total_reward = 0
            state, _ = evaluate_env.reset()
            while not (evaluate_done or evaluate_failed):
                action = self.select_action(state, type="greedy")
                state, reward, evaluate_failed, evaluate_done, _ = evaluate_env.step(action)
                total_reward += reward
            returns.append(total_reward)

        return round(np.mean(returns), 2)
