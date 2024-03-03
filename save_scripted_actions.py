from functools import partial
import torch

import botbowl
from botbowl.ai.env import BotBowlEnv, RewardWrapper, EnvConf, ScriptedActionWrapper, PPCGWrapper
from a2c_agent import A2CAgent
from a2c_env import A2C_Reward, a2c_scripted_actions
from botbowl.ai.layers import *
import helpers
import a2c_scripted_bot_example

#examples.a2c.a2c_scripted_bot_example.

# Environment
env_size = 1 # Options are 1,3,5,7,11
env_name = f"botbowl-{env_size}"
env_conf = EnvConf(size=env_size, pathfinding=False)


def make_env(script=False):
    #print(type(BotBowlEnv()))
    if script:
        env = BotBowlEnv(env_conf)
    else:
        env = BotBowlEnv(env_conf)
        env = PPCGWrapper(env)
    env = ScriptedActionWrapper(env, scripted_func=a2c_scripted_actions)
    env = RewardWrapper(env, home_reward_func=A2C_Reward())
    return env

def main():

    env = make_env()
    spat_obs, non_spat_obs, action_mask = env.reset()
    spatial_obs_space = spat_obs.shape
    non_spatial_obs_space = non_spat_obs.shape[0]
    action_space = len(action_mask)
    del env, non_spat_obs, action_mask  # remove from scope to avoid confusion further down

    #[1, 534]
    #[1, 8116]
    file_num = 0
    for i in range(300):
        home_agent = botbowl.make_bot('scripted_a2c')
        home_agent.name = "Scripted Bot"
        env = make_env(script=False)
        spat_obs, non_spat_obs, action_mask = env.reset()
        home_agent.new_game(env.game, env.game.state.home_team)
        start_action = Action(ActionType.START_GAME)
        env.game.step(start_action)
        
        lista = dict()
        
        #testa = None

        while True:
            #print(env.env.env.env.env_conf.positional_action_types)
            sc_action = home_agent.act(env.game)
            #print(sc_action)
            #print(env.env.env.env.env_conf.positional_action_types)
            decoded_action = helpers.decode_action(sc_action, env)
            env.game.step(sc_action)

            if env.game.state.game_over:
                env.game._end_game()
                break
            spatial_obs, non_spatial_obs, action_mask = env.get_state()
            if decoded_action:
                # print("TEST #11:", decoded_action)
                action_to_save = torch.zeros([1, 534])
                action_to_save[0][decoded_action] = 2
                # print("TEST #12:", action_to_save.index(100))
                helpers.save_data(spatial_obs, non_spatial_obs, action_mask, file_num, action_to_save)
                file_num += 1
        if env.game.get_winning_team() is not None:
            print(env.game.get_winning_team().name)


if __name__ == "__main__":
    main()