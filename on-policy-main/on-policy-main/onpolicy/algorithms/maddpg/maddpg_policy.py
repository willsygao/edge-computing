import torch
import torch.nn.functional as F
from onpolicy.algorithms.maddpg.algorithm.actor_critic import MADDPGActor, MADDPGCritic
from onpolicy.utils.util import update_linear_schedule, check

class MADDPGPolicy:
    def __init__(self, args, obs_space, cent_obs_space, action_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.weight_decay = args.weight_decay
        self.opti_eps = args.opti_eps

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.action_space = action_space
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.actor = MADDPGActor(args, obs_space, action_space, device)
        self.critic = MADDPGCritic(args, cent_obs_space, action_space, device)
        
        self.target_actor = MADDPGActor(args, obs_space, action_space, device)
        self.target_critic = MADDPGCritic(args, cent_obs_space, action_space, device)
        
        # Hard update targets initially
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        
    def lr_decay(self, episode, episodes):
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, share_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None, deterministic=False):
        obs = check(obs).to(**self.tpdv)
        share_obs = check(share_obs).to(**self.tpdv)
        rnn_states_actor = check(rnn_states_actor).to(**self.tpdv)
        rnn_states_critic = check(rnn_states_critic).to(**self.tpdv)
        
        logits = self.actor(obs)
        if available_actions is not None:
             available_actions = check(available_actions).to(**self.tpdv)
             logits[available_actions == 0] = -1e10

        action_log_probs = torch.zeros(obs.shape[0], 1).to(**self.tpdv) # Dummy

        if self.action_space.__class__.__name__ == "Discrete":
             if deterministic:
                  actions = logits.argmax(dim=-1, keepdim=True)
                  one_hot_action = F.one_hot(actions.squeeze(-1), num_classes=self.action_space.n).float()
             else:
                  # Use Gumbel for noise/exploration
                  one_hot_action = F.gumbel_softmax(logits, hard=True)
                  actions = one_hot_action.argmax(dim=-1, keepdim=True)
        else:
             # Assume Box
             actions = torch.tanh(logits)
             if not deterministic:
                  # Simple epsilon noise
                  noise = torch.randn_like(actions) * 0.1
                  actions = (actions + noise).clamp(-1, 1)
             one_hot_action = actions # Continuous

        # Get values (Q(s, a))
        values = self.critic(share_obs, one_hot_action)
        
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, share_obs, rnn_states_critic, masks):
        # Used by compute_returns in base_runner, but mostly for GAE.
        # Can return 0s or estimate V if possible.
        return torch.zeros(share_obs.shape[0], 1).to(self.device)

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        # Used by eval()
        obs = check(obs).to(**self.tpdv)
        rnn_states_actor = check(rnn_states_actor).to(**self.tpdv)
        logits = self.actor(obs)
        if available_actions is not None:
             available_actions = check(available_actions).to(**self.tpdv)
             logits[available_actions == 0] = -1e10

        if self.action_space.__class__.__name__ == "Discrete":
             if deterministic:
                  actions = logits.argmax(dim=-1, keepdim=True)
             else:
                  one_hot_action = F.gumbel_softmax(logits, hard=True)
                  actions = one_hot_action.argmax(dim=-1, keepdim=True)
        else:
             actions = torch.tanh(logits)

        return actions, rnn_states_actor
        
    def soft_update(self, tau):
        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
