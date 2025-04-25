import time
import os

import numpy as np
import pandas as pd
import gymnasium as gym

from tqdm import tqdm

from batch_actor_critic_agent import ActorCriticAgent
from utils import LearningCurvePlot, sma_smooth, moving_variance, arg_parser, get_folder


file_dir = os.path.dirname(os.path.abspath(__file__))
base_csv_folder = os.path.join(file_dir, "csv")
base_image_folder = os.path.join(file_dir, "images")


def train(lr_vc=1e-4, lr_pa= 1e-4, hidden_layer_size_vc="512_512", hidden_layer_size_pa="512_512", n_steps = 5, M=1, timesteps=int(1e6+1), trial="test", advantage=False, write_to_file=True):
    csv_folder = get_folder(trial, base_csv_folder)
    image_folder = get_folder(trial, base_image_folder)
    name = "ActorCritic"
    if advantage:
        name = "Advantage" + name
    file_name = f"{name}_timesteps_{timesteps}_lr_vc_{lr_vc:.4g}_lr_pa_{lr_pa:.4g}_nn_vc_{hidden_layer_size_vc}_nn_pa_{hidden_layer_size_pa}_n_steps_{n_steps}_M_{M}"
    print(file_name)
    hidden_layer_size_vc_ = tuple(map(int, hidden_layer_size_vc.split("_")))  # Convert back to tuple
    hidden_layer_size_pa_ = tuple(map(int, hidden_layer_size_pa.split("_")))  # Convert back to tuple
    
    env = gym.make("CartPole-v1")
    evaluate_env = gym.make("CartPole-v1")
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    eval_interval = timesteps // 4000

    eval_df = pd.DataFrame(columns=["Timestep"] + list(range(NUM_ENVS_SEPARATELY)) + ["Total Reward"])
    eval_df['Timestep'] = np.arange(0, timesteps, eval_interval)

    for i in range(NUM_ENVS_SEPARATELY):
        now = time.time()
        timestep = 0
        m = 0
        agent = ActorCriticAgent(state_dim, action_dim, lr_vc, lr_pa,  hidden_layers_vc=hidden_layer_size_vc_, hidden_layers_pa= hidden_layer_size_pa_, n_steps=n_steps)
        # pbar = tqdm(total=timesteps)
        episode_batches = []
        while timestep < timesteps:
            state, _ = env.reset()
            done = False
            episode_buffer = {
                'states': [],  'log_probs': [],
                'rewards': [], 'next_states': [], 'dones': []
            }

            while not done:
                action, log_prob = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_buffer['states'].append(state)
                episode_buffer['log_probs'].append(log_prob)
                episode_buffer['rewards'].append(reward)
                episode_buffer['next_states'].append(next_state)
                episode_buffer['dones'].append(done)

                state = next_state
                if timestep % eval_interval == 0:
                    eval_rewards = agent.evaluate(evaluate_env)
                    eval_df.loc[eval_df['Timestep'] == timestep, i] = eval_rewards
                    print(f"Timestep {timestep+1}, Average Score: {eval_rewards:.2f}")
                timestep += 1
            episode_batches.append(episode_buffer)
            m += 1
            if m % M == 0:
                agent.update(episode_batches, advantage=advantage)
                episode_batches = []
        # pbar.close()
        # pbar.update(1)

        print('Running one setting takes {} minutes'.format((time.time()-now)/60))

    if write_to_file:
        eval_df["Total Reward"] = eval_df[list(range(NUM_ENVS_SEPARATELY))].mean(axis=1)
        eval_df.to_csv(os.path.join(csv_folder, f"{file_name}.csv"), index=False)
        
        # Plot for each hyperparameter separately
        timesteps = eval_df["Timestep"]
        data = eval_df["Total Reward"]
        img_plt = LearningCurvePlot(title=r'Batch Actor Critic: LR_Critic: {}, LR_Actor: {}, N-steps: {}, M:{}, Hidden Layers size for Actor{} & Critic: {}'.format(lr_vc, lr_pa, n_steps,M, hidden_layer_size_pa, hidden_layer_size_vc))

        img_plt.set_ylim(0, OPTIMAL_EPISODE_RETURN + 50)
        img_plt.add_hline(OPTIMAL_EPISODE_RETURN, label="Reward Optimum")

        smoothed_rewards = np.array(sma_smooth(data, window_size=60), dtype=np.float64)
        smoothed_variance = np.array(moving_variance(data, window_size=60), dtype=np.float64)
        img_plt.add_curve(timesteps, smoothed_rewards, label='SMA - Window: 60')
        img_plt.fill_between(timesteps, smoothed_rewards - np.sqrt(smoothed_variance),
                            smoothed_rewards + np.sqrt(smoothed_variance))

        img_plt.save(os.path.join(image_folder, file_name+".png"))


NUM_ENVS_SEPARATELY = 5
OPTIMAL_EPISODE_RETURN = 500

LEARNING_RATES_PA = [1e-4]
LEARNING_RATES_VC = [1.5e-4]
PA_NN_SIZE = ["32_32"]
VC_NN_SIZE = ["64_64"]
N_STEPS = [50]
M=[1]
HYPER_PARAMETER_TRAINING_STEPS = int(1e6+1)
TRIAL = "actor-critic"

LEARNING_RATES_VC, LEARNING_RATES_PA, N_STEPS, M, PA_NN_SIZE, VC_NN_SIZE, HYPER_PARAMETER_TRAINING_STEPS, TRIAL = arg_parser(LEARNING_RATES_VC, LEARNING_RATES_PA, N_STEPS, M, PA_NN_SIZE, VC_NN_SIZE, HYPER_PARAMETER_TRAINING_STEPS, TRIAL)

# Hyperparameter Training
for each_lr_vc in LEARNING_RATES_VC:
    for each_lr_pa in LEARNING_RATES_PA:
        for each_pa_nn in PA_NN_SIZE:
            for each_vc_nn in VC_NN_SIZE:
                for m in M:
                    for n_steps in N_STEPS:
                        # Actor Critic
                        train(lr_vc=each_lr_vc, lr_pa=each_lr_pa, hidden_layer_size_pa=each_pa_nn, hidden_layer_size_vc=each_vc_nn, n_steps=n_steps, M=m, timesteps=HYPER_PARAMETER_TRAINING_STEPS, trial=TRIAL)
                        ## Advantage Actor Critic
                        train(lr_vc=each_lr_vc, lr_pa=each_lr_pa, hidden_layer_size_pa=each_pa_nn, hidden_layer_size_vc=each_vc_nn, n_steps=n_steps, M=m, timesteps=HYPER_PARAMETER_TRAINING_STEPS, trial=TRIAL, advantage=True)
