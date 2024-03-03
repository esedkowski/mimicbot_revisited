from functools import partial
from multiprocessing import Process, Pipe
# import random
from typing import Tuple, Iterable

# import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
# from torch.autograd import Variable

# import botbowl
from botbowl.ai.env import BotBowlEnv, RewardWrapper, EnvConf, ScriptedActionWrapper, BotBowlWrapper, PPCGWrapper

# from a2c_agent import A2CAgent, CNNPolicy

from a2c_env import A2C_Reward, a2c_scripted_actions
from botbowl.ai.layers import *
import numpy as np

# import csv

import os
import torch
import pandas as pd
# from skimage import io, transform
import numpy as np
# import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, utils

from dataset import MimicDataset, mimic_dataset, mimic_dataloader

from testowy_agent import CNNPolicy


# Environment
env_size = 1  # Options are 1,3,5,7,11
env_name = f"botbowl-{env_size}"
env_conf = EnvConf(size=env_size, pathfinding=False)

#make_agent_from_model = partial(A2CAgent, env_conf=env_conf, scripted_func=a2c_scripted_actions)


def make_env():
    env = BotBowlEnv(env_conf)
    if ppcg:
        env = PPCGWrapper(env)
    # env = ScriptedActionWrapper(env, scripted_func=a2c_scripted_actions)
    env = RewardWrapper(env, home_reward_func=A2C_Reward())
    return env


# Training configuration
num_steps = 1000000
num_processes = 1
steps_per_update = 20
learning_rate = 0.001
gamma = 0.99
entropy_coef = 0.01
value_loss_coef = 0.5
max_grad_norm = 0.05
log_interval = 50
save_interval = 10
ppcg = False

# Architecture
num_hidden_nodes = 128
num_cnn_kernels = [32, 64]

# Make directories
def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


ensure_dir("logs/")
ensure_dir("models/")
ensure_dir("plots/")
exp_id = str(uuid.uuid1())
log_dir = f"logs/{env_name}/"
model_dir = f"models/{env_name}/"
plot_dir = f"plots/{env_name}/"
ensure_dir(log_dir)
ensure_dir(model_dir)
ensure_dir(plot_dir)



class VecEnv:
    def __init__(self, envs):
        """
        envs: list of botbowl environments to run in subprocesses
        """
        self.closed = False
        nenvs = len(envs)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])

        # self.ps = [Process(target=worker, args=(work_remote, remote, env, envs.index(env)))
        #            for (work_remote, remote, env) in zip(self.work_remotes, self.remotes, envs)]

        # for p in self.ps:
        #     p.daemon = True  # If the main process crashes, we should not cause things to hang
        #     p.start()
        # for remote in self.work_remotes:
        #     remote.close()

    def step(self, actions: Iterable[int], difficulty=1.0) -> Tuple[np.ndarray, ...]:
        """
        Takes one step in each environment, returns the results as stacked numpy arrays
        """
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', [action, difficulty]))
        results = [remote.recv() for remote in self.remotes]
        return tuple(map(np.stack, zip(*results)))

    def reset(self, difficulty=1.0):
        for remote in self.remotes:
            remote.send(('reset', difficulty))
        results = [remote.recv() for remote in self.remotes]
        return tuple(map(np.stack, zip(*results)))

    def swap(self, agent):
        for remote in self.remotes:
            remote.send(('swap', agent))

    # def close(self):
    #     if self.closed:
    #         return

    #     for remote in self.remotes:
    #         remote.send(('close', None))
    #     for p in self.ps:
    #         p.join()
    #     self.closed = True

    @property
    def num_envs(self):
        return len(self.remotes)
      
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        # values, actions = model.act(X['spatial_obs'], X['non_spatial_obs'], X['mask'], False, None, None, None)
        for i in range (X['spatial_obs'].size()[0]):
            actions = model(X['spatial_obs'][i], X['non_spatial_obs'][i])
            loss = loss_fn(actions, y['action_probs'][i])
        #values, actions = model.act(X['spatial_obs'], X['non_spatial_obs'], X['mask'], False, None, None, None)
        #loss = loss_fn(values, y['values']) + loss_fn(actions, y['action_probs'])

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            actions = model(X['spatial_obs'][0], X['non_spatial_obs'][0])
            test_loss += loss_fn(actions, y['action_probs'][0]).item()
            # correct += (values.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    # correct /= size
    # print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")



def main():
    envs = VecEnv([make_env() for _ in range(num_processes)])

    env = make_env()
    print(env.env.away_agent, env.env.home_agent)
    spat_obs, non_spat_obs, action_mask = env.reset()
    spatial_obs_space = spat_obs.shape
    non_spatial_obs_space = non_spat_obs.shape[0]
    action_space = len(action_mask)
    del env, non_spat_obs, action_mask  # remove from scope to avoid confusion further down

    # MODEL
    ac_agent = CNNPolicy(spatial_obs_space,
                         non_spatial_obs_space,
                         hidden_nodes=num_hidden_nodes,
                         kernels=num_cnn_kernels,
                         actions=action_space)
    

    train_dataloader = mimic_dataloader
    test_dataloader = mimic_dataloader

    learning_rate = 1e-3
    batch_size = 64
    epochs = 5

    # Initialize the loss function
    #loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.MSELoss()

    # OPTIMIZER
    optimizer = optim.RMSprop(ac_agent.parameters(), learning_rate)



    epochs = 2
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, ac_agent, loss_fn, optimizer)
        test_loop(test_dataloader, ac_agent, loss_fn)
    print("Done!")
    torch.save(ac_agent.state_dict(), "saved.pt")

#def main():
#    pass


if __name__ == "__main__":
    main()