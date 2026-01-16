import torch
import torch.nn as nn
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.util import init, check

class MADDPGActor(nn.Module):
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(MADDPGActor, self).__init__()
        self.hidden_size = args.hidden_size
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self.tpdv = dict(dtype=torch.float32, device=device)
        
        obs_shape = obs_space.shape
        self.base = MLPBase(args, obs_shape)
        
        if action_space.__class__.__name__ == "Discrete":
             action_dim = action_space.n
             self.act_type = "discrete"
        elif action_space.__class__.__name__ == "Box":
             action_dim = action_space.shape[0]
             self.act_type = "continuous"
        elif action_space.__class__.__name__ == "MultiDiscrete":
             action_dim = action_space.shape
             self.act_type = "multidiscrete"
        else:
             raise NotImplementedError
        
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), self._gain)

        if self.act_type == "discrete":
             self.act = init_(nn.Linear(self.hidden_size, action_dim))
        elif self.act_type == "continuous":
             self.act = init_(nn.Linear(self.hidden_size, action_dim))
        
        self.to(device)

    def forward(self, obs):
        obs = check(obs).to(**self.tpdv)
        actor_features = self.base(obs)
        actions = self.act(actor_features)
        return actions

class MADDPGCritic(nn.Module):
    def __init__(self, args, cent_obs_space, action_space, device=torch.device("cpu")):
        super(MADDPGCritic, self).__init__()
        self.hidden_size = args.hidden_size
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self.tpdv = dict(dtype=torch.float32, device=device)
        
        cent_obs_shape = cent_obs_space.shape
        
        if action_space.__class__.__name__ == "Discrete":
             action_dim = action_space.n
        elif action_space.__class__.__name__ == "Box":
             action_dim = action_space.shape[0]
        else:
             action_dim = 0 # Todo: deal with MultiDiscrete
        
        self.base = MLPBase(args, (cent_obs_shape[0] + action_dim,))
        
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), self._gain)
            
        self.v_out = init_(nn.Linear(self.hidden_size, 1))
        
        self.to(device)

    def forward(self, cent_obs, actions):
        cent_obs = check(cent_obs).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv)
        cat_input = torch.cat([cent_obs, actions], dim=1)
        critic_features = self.base(cat_input)
        values = self.v_out(critic_features)
        return values
