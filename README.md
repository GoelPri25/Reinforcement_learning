## About

This project implements and compares three policy-based Reinforcement Learning algorithms—**REINFORCE**, **Actor-Critic**, and **Advantage Actor-Critic (A2C)**—on the classic `CartPole-v1` environment. These methods aim to address the limitations of value-based approaches like DQN, especially in handling continuous action spaces and reducing bias.

---

## Overview

The goal is to study the effectiveness of policy-gradient algorithms through:

- A detailed theoretical foundation  
- Practical implementation using PyTorch  
- Extensive ablation studies to tune hyperparameters  
- Performance evaluation against a DQN baseline with Experience Replay and Target Networks

Each algorithm is tested for learning stability, convergence speed, and overall performance. The results show that **A2C outperforms** both REINFORCE and standard Actor-Critic in terms of variance reduction and training stability.
