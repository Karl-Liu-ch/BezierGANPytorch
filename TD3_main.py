import torch
from agent.TD3 import *
import math
import os
from Environment import OptimEnv
import argparse
parser = argparse.ArgumentParser(description="Matlab Simscape")
parser.add_argument('--version', type=str, default='Dronesimscape.slx')
parser.add_argument("--expl_noise", default=0.1, type=float)
parser.add_argument("--gpu_id", type=str, default='0', help='path log files')
opt = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

model_path=opt.version
from pathlib import Path
import warnings
import numpy as np
warnings.filterwarnings("ignore")
env = env=OptimEnv()
K_epochs = 80               # update policy for K epochs
eps_clip = 0.2              # clip parameter for PPO
gamma = 0.97                # discount factor

lr_actor = 0.0003           # learning rate for actor
lr_critic = 0.001           # learning rate for critic
td3_agent = TD3(state_dim=512, action_dim=13, max_action = 1.0, policy_freq=2)
env_name = 'Airfoil'
directory = "agent/TD3_preTrained/" + env_name + '/'
try:
    os.mkdir(directory)
except:
    pass
checkpoint_path = directory + "TD3_{}.pth".format(env_name)
try:
    td3_agent.load(checkpoint_path)
except:
    pass

env_name = "Airfoil"
has_continuous_action_space = True

max_ep_len = 100              # max timesteps in one episode
max_training_timesteps = int(1e8)   # break training loop if timeteps > max_training_timesteps

print_freq = max_ep_len * 4     # print avg reward in the interval (in num timesteps)
# log_freq = max_ep_len * 2       # log avg reward in the interval (in num timesteps)
save_model_freq = int(2e2)      # save model frequency (in num timesteps)
print_running_reward = 0
print_running_episodes = 0

log_running_reward = 0
log_running_episodes = 0
# action_std = None

update_timestep = max_ep_len * 4      # update policy every n timesteps
K_epochs = 40

log_running_reward = 0
log_running_episodes = 0

def sample_action():
    bounds = (0.0, 1.0)
    latent_dim = 3
    noise_dim = 10
    y_latent = np.random.uniform(low=bounds[0], high=bounds[1], size=(1, latent_dim))
    noise = np.random.normal(scale=0.5, size=(1, noise_dim))
    y_latent = torch.from_numpy(y_latent).to(device)
    y_latent = y_latent.float()
    noise = torch.from_numpy(noise).to(device)
    noise = noise.float()
    noise = torch.concat([noise, y_latent], dim=-1)
    noise = noise.squeeze(dim=0)
    return noise.detach().cpu().numpy()

time_step = 0
i_episode = 0
action_dim = 13
replay_buffer = ReplayBuffer(state_dim=512, action_dim=action_dim)
episode_timesteps  = 0
episode_num = 0
# training loop
state = env.reset()
for time_step in range(int(max_training_timesteps)):
    episode_timesteps += 1
    current_ep_reward = 0
    
    if time_step < 2e2:
        action = sample_action()
    else:
        action = td3_agent.select_action(state)
    
    new_state, reward, done, _ = env.step(action)
    if done and episode_timesteps < max_ep_len:
        done_bool = 1
    else:
        done_bool = 0
    replay_buffer.add(state, action, new_state, reward, done_bool)
    state = new_state
    
    # saving reward and is_terminals
    
    current_ep_reward += reward

    # update PPO agent
    if time_step > 2e2:
        # print('training')
        td3_agent.train(replay_buffer)
        
    # save model weights
    if (time_step + 1) % save_model_freq == 0:
        print("--------------------------------------------------------------------------------------------")
        print("saving model at : " + checkpoint_path)
        td3_agent.save(checkpoint_path)
        print("model saved")
        
    # break; if the episode is over
    if done or episode_timesteps > max_ep_len: 
        # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
        print(f"Total T: {time_step+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {current_ep_reward:.3f}")
        # Reset environment
        state, done = env.reset(), False
        current_ep_reward = 0
        episode_timesteps = 0
        episode_num += 1 

    print_running_reward += current_ep_reward
    print_running_episodes += 1

    log_running_reward += current_ep_reward
    log_running_episodes += 1

    i_episode += 1

env.close()