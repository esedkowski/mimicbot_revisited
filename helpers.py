#from botbow import botbowl

from botbowl.core.model import Action
from botbowl.ai.env import BotBowlEnv
from botbowl.core import ActionType

import torch
import numpy as np
#import torch.nn as nn
#import torch.optim as optim
#from torch.autograd import Variable

import csv

def decode_action(action: Action, env):
    #print(action)
    action_idx = False
    if action.position is None:
        
        #print("NO POSITION")

        #if action.action_type == ActionType.END_SETUP:
            #return action_idx
        try:
            action_idx = env.env.env.env.env_conf.simple_action_types.index(action.action_type)
        except ValueError:
            pass
            #print("pass")
    else:

        #print(action.position)
        try:
            flip = env.env.env.env.away_team_active()
            spatial_x = action.position.x
            if flip:
                spatial_x = env.env.env.env.width - spatial_x - 1

            #action.position.y * env.env.env.width + spatial_x

            spatial_idx = env.env.env.env.env_conf.positional_action_types.index(action.action_type) * env.env.env.env.board_squares + action.position.y * env.env.env.env.width + spatial_x
            action_idx = spatial_idx + len(env.env.env.env.env_conf.simple_action_types)
            
            
            # spatial_pos_idx = spatial_idx % env.env.env.env.board_squares
            # spatial_x = spatial_pos_idx % env.env.env.env.width
            # spatial_y = spatial_pos_idx // env.env.env.env.width
            # spatial_action_type = env.env.env.env.env_conf.positional_action_types[spatial_idx // env.env.env.env.board_squares]
            # print("spatial_action_type:", spatial_action_type, "spatial_y:", spatial_y, "spatial_x", spatial_x)
        except ValueError:
            pass
            #print("pass")


    #print(action_idx)
    #print(env.env.env.env._compute_action(action_idx))
    return action_idx

def save_data(spatial_inputs, non_spatial_input, action_mask, file_num, action):
            #print(type(spatial_inputs))
            spatial_inputs = spatial_inputs[None, :]
            non_spatial_input = non_spatial_input[None, :]
            spatial_obs=torch.from_numpy(spatial_inputs).float()
            non_spatial_obs=torch.from_numpy(non_spatial_input).float()
            mask=torch.from_numpy(action_mask).float()
            #print(non_spatial_obs.size())
            #print(spatial_obs.size())

            i_file = f"\\data\\in_{file_num}.pt"
            i_dict = {'spatial_obs' : spatial_obs, 'non_spatial_obs' : non_spatial_obs, 'mask' : mask}
            #print(type(i_dict['spatial_obs']), i_dict['spatial_obs'].size())
            #input()
            torch.save(i_dict, i_file)

            

            o_file = f"C:\\data\\out_{file_num}.pt"
            #print("OUTPUT:", action.detach()[0])
            o_dict = {'action_probs' : action.detach()}

            torch.save(o_dict, o_file)

            with open('C:data\\data.csv', 'a', newline='') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                spamwriter.writerow([i_file, o_file])

a = np.array([[1,2,4]])
print(a.size)

b = np.array(a,)
#b = np.stack(results)
