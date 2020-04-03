import torch
from rl_modules.models import actor
from arguments import get_args
import gym
import numpy as np
import csv

from envs.my_manipulate_touch_sensors import SharedBlockTouchSensorsEnv, SharedBlockTouchSensorsEnvSparse
from gym.wrappers import TimeLimit

# process the inputs
def process_inputs(o, g, o_mean, o_std, g_mean, g_std, args):
    o_clip = np.clip(o, -args.clip_obs, args.clip_obs)
    g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -args.clip_range, args.clip_range)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -args.clip_range, args.clip_range)
    inputs = np.concatenate([o_norm, g_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32) # pylint: disable=not-callable
    return inputs

if __name__ == '__main__':
    args = get_args()
    # load the model param
    model_path = args.save_dir + args.env_name + '/model.pt'
    o_mean, o_std, g_mean, g_std, model = torch.load(model_path, map_location=lambda storage, loc: storage)
    # create the environment
    env = SharedBlockTouchSensorsEnvSparse()
    env = TimeLimit(env,100)
    # get the env param
    observation = env.reset()
    # get the environment params
    env_params = {'obs': observation['observation'].shape[0], 
                  'goal': observation['desired_goal'].shape[0], 
                  'action': env.action_space.shape[0], 
                  'action_max': env.action_space.high[0],
                  }
    # create the actor network
    actor_network = actor(env_params)
    actor_network.load_state_dict(model)
    actor_network.eval()
    for i in range(args.demo_length):
        ag_log = []
        dg_log = []
        pi_log = []
        observation = env.reset()
        # start to do the demo
        obs = observation['observation']
        g = observation['desired_goal']
        ag_log.append(observation['achieved_goal'])
        dg_log.append(observation['desired_goal'])
        for t in range(env._max_episode_steps):
            env.render()
            inputs = process_inputs(obs, g, o_mean, o_std, g_mean, g_std, args)
            inputs = inputs[None,:]
            with torch.no_grad():
                pi = actor_network(inputs)
            action = pi.detach().numpy().squeeze()
            # put actions into the environment
            observation_new, reward, _, info = env.step(action)
            obs = observation_new['observation']
            # print("reward:{}".format(reward))
            ag_log.append(observation_new['achieved_goal'])
            dg_log.append(observation_new['desired_goal'])
            pi_log.append(pi)
        print('the episode is: {}, is success: {}'.format(i, info['is_success']))
        pi_log.append(pi)
        p1 = np.vstack(ag_log)
        p2 = np.vstack(dg_log)
        p3 = np.vstack(pi_log)
        p = np.hstack([p1,p2,p3])
        np.savetxt("saved_results/{}.csv".format(i), p, delimiter=",")
