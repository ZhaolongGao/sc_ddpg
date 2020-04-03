import torch
import torch.nn as nn
import torch.nn.functional as F

"""
the input x in both networks should be [o, g], where o is the observation and g is the goal.

"""


def sample_z(mu, log_var):
    mb_size = mu.shape[0]
    Z_dim = mu.shape[-1]
    eps = torch.randn(mb_size, Z_dim)
    return mu + torch.exp(log_var / 2) * eps


# define the actor network
class actor(nn.Module):
    def __init__(self, env_params):
        super(actor, self).__init__()
        self.num_aux = 20
        self.num_obs = env_params['obs']
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.mu = nn.Linear(env_params['obs'], self.num_aux)
        self.var = nn.Linear(env_params['obs'], self.num_aux)
        self.action_out = nn.Linear(256 + self.num_aux, env_params['action'])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mus = self.mu(x[:, :self.num_obs])
        vars = self.var(x[:, :self.num_obs])
        aux = sample_z(mus, vars)
        x = torch.cat([x, aux], axis=1)
        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions


class critic(nn.Module):
    def __init__(self, env_params):
        super(critic, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.q_out = nn.Linear(256, 1)

    def forward(self, x, actions):
        x = torch.cat([x, actions / self.max_action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)

        return q_value
