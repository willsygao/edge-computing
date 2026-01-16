import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from onpolicy.utils.util import check

class MADDPG:
    def __init__(self, args, policy, device=torch.device("cpu")):
        self.args = args
        self.policy = policy
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.tau = 0.005 # Soft update parameter
        self.batch_size = 64 # Batch size 
        self.value_normalizer = None
        self._use_valuenorm = args.use_valuenorm
        if args.num_mini_batch > 0:
             # Just an approximation if user supplies mini_batch
             pass

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()
        self.policy.target_actor.train()
        self.policy.target_critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
        self.policy.target_actor.eval()
        self.policy.target_critic.eval()
        
    def train(self, buffer):
        self.prep_training()
        
        # Flatten data from buffer
        # obs: (T+1, N, dim) -> (T*N, dim)
        # We exclude the last observation for current state s, use it for s'
        
        # s: slice 0 to -1
        share_obs = buffer.share_obs[:-1].reshape(-1, *buffer.share_obs.shape[2:])
        obs = buffer.obs[:-1].reshape(-1, *buffer.obs.shape[2:])
        actions = buffer.actions.reshape(-1, *buffer.actions.shape[2:])
        rewards = buffer.rewards.reshape(-1, 1) # matches T steps
        masks = buffer.masks[1:].reshape(-1, 1) # Mask at t+1 is for transition t->t+1
        
        # s': slice 1 to end
        share_obs_next = buffer.share_obs[1:].reshape(-1, *buffer.share_obs.shape[2:])
        obs_next = buffer.obs[1:].reshape(-1, *buffer.obs.shape[2:])
        
        if buffer.available_actions is not None:
             available_actions = buffer.available_actions[:-1].reshape(-1, *buffer.available_actions.shape[2:])
             available_actions_next = buffer.available_actions[1:].reshape(-1, *buffer.available_actions.shape[2:])
        else:
             available_actions = None
             available_actions_next = None

        total_samples = share_obs.shape[0]
        train_info = {}
        train_info['critic_loss'] = 0
        train_info['actor_loss'] = 0
        updates = 0
        
        # Determine number of epochs
        num_epochs = self.args.ppo_epoch
        
        for _ in range(num_epochs):
             sampler = torch.randperm(total_samples)
             for i in range(0, total_samples, self.batch_size):
                  indices = sampler[i:i+self.batch_size]
                  if len(indices) < self.batch_size: continue
                  
                  # Move to device
                  sb = torch.from_numpy(share_obs[indices]).to(**self.tpdv)
                  sb_next = torch.from_numpy(share_obs_next[indices]).to(**self.tpdv)
                  ob = torch.from_numpy(obs[indices]).to(**self.tpdv)
                  ob_next = torch.from_numpy(obs_next[indices]).to(**self.tpdv)
                  ab = torch.from_numpy(actions[indices]).to(**self.tpdv)
                  rb = torch.from_numpy(rewards[indices]).to(**self.tpdv)
                  mb = torch.from_numpy(masks[indices]).to(**self.tpdv)
                  
                  if available_actions is not None:
                       aab_next = torch.from_numpy(available_actions_next[indices]).to(**self.tpdv)
                       aab = torch.from_numpy(available_actions[indices]).to(**self.tpdv)
                  else:
                       aab_next = None
                       aab = None

                  # --- Critic Update ---
                  with torch.no_grad():
                       # Target Actor
                       next_actions_logits = self.policy.target_actor(ob_next)
                       if aab_next is not None:
                            next_actions_logits[aab_next == 0] = -1e10
                            
                       if self.policy.action_space.__class__.__name__ == "Discrete":
                            # Use softmax for probability distribution as "action" input to Critic (continuous relaxation)
                            # Or one-hot via argmax? Softmax is better for smooth target.
                            next_actions = F.softmax(next_actions_logits, dim=-1)
                       else:
                            next_actions = torch.tanh(next_actions_logits)
                       
                       # Target Critic
                       next_Q = self.policy.target_critic(sb_next, next_actions)
                       target_Q = rb + self.args.gamma * mb * next_Q

                  # Current Critic
                  if self.policy.action_space.__class__.__name__ == "Discrete":
                       # Convert discrete action indices to one-hot
                       ab_one_hot = F.one_hot(ab.long().squeeze(-1), num_classes=self.policy.action_space.n).float()
                       current_Q = self.policy.critic(sb, ab_one_hot)
                  else:
                       current_Q = self.policy.critic(sb, ab)
                  
                  critic_loss = F.mse_loss(current_Q, target_Q)
                  
                  self.policy.critic_optimizer.zero_grad()
                  critic_loss.backward()
                  self.policy.critic_optimizer.step()
                  
                  # --- Actor Update ---
                  curr_actions_logits = self.policy.actor(ob)
                  if aab is not None:
                       curr_actions_logits[aab == 0] = -1e10
                  
                  if self.policy.action_space.__class__.__name__ == "Discrete":
                       # Gumbel Softmax (Hard=False -> differentiable soft approximation)
                       # Hard=True -> discrete output but gradients via straight-through
                       # DDPG gradients usually flow better with Softmax or Gumbel-Softmax(hard=False) into Critic
                       curr_actions = F.gumbel_softmax(curr_actions_logits, hard=False)
                  else:
                       curr_actions = torch.tanh(curr_actions_logits)
                  
                  actor_loss = -self.policy.critic(sb, curr_actions).mean()
                  
                  self.policy.actor_optimizer.zero_grad()
                  actor_loss.backward()
                  self.policy.actor_optimizer.step()
                  
                  # --- Soft Update ---
                  self.policy.soft_update(self.tau)
                  
                  train_info['critic_loss'] += critic_loss.item()
                  train_info['actor_loss'] += actor_loss.item()
                  updates += 1
        
        if updates > 0:
             train_info['critic_loss'] /= updates
             train_info['actor_loss'] /= updates
             
        return train_info
