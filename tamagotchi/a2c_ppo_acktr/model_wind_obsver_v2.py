# Version 2. Modular wind predictor that feeds prediction and uncertainty into the decision making network.
import numpy as np
import torch
import torch.nn as nn
from tamagotchi.a2c_ppo_acktr.gnODE import GNODE
from tamagotchi.a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian, BetaCustom
from tamagotchi.a2c_ppo_acktr.utils import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    
class WindObserverModule(nn.Module):
    # Wind obsver v1 modification: new RNN-based wind observer
    # Wind obsver v2 modification: rename to module since its RNN with heads 
    def __init__(self, obs_dim, hidden_size=32):
        super().__init__()
        self.rnn = nn.RNN(obs_dim, hidden_size)
        self.head_mu = nn.Linear(hidden_size, 2)
        self.head_logvar = nn.Linear(hidden_size, 2)
        self.recurrent_hidden_state_size = hidden_size

    @property
    def hidden_state_size(self):
        return self.rnn.hidden_size

    def forward(self, obs_seq, h_obs, masks):
        # not a _forward_rnn since this is a full module. 
        """
        obs_seq: 
            - [B, obs_dim]               (single step)
            - or [T*N, obs_dim]          (flattened sequence)
        h_obs:
            - if single step: [B, H]
            - if sequence:    [N, H]     (one hidden per env)
        masks:
            - [B, 1]                      in single-step
            - [T*N, 1]                    in sequence mode
        returns:
            wind_mu:     [B, 2] or [T*N, 2]
            wind_logvar: [B, 2] or [T*N, 2]
            h_obs:       [B, H] or [N, H] (final hidden)
        """
        # obs_seq: [T*B, obs_dim] or [B, obs_dim]
        x = obs_seq
        B = h_obs.size(0)

        if x.size(0) == B:  # single step
            x, h_obs = self.rnn(x.unsqueeze(0), (h_obs * masks).unsqueeze(0))
            x = x.squeeze(0)
            h_obs = h_obs.squeeze(0)
            mu = self.head_mu(x)
            logvar = self.head_logvar(x)
            return mu, logvar, h_obs

        else:
            # 这里 B 实际上是 N（env 个数）
            N = B
            T = int(x.size(0) / N)
            # obs_seq: [T*N, obs_dim] -> [T, N, obs_dim]
            x = x.view(T, N, x.size(1))

            # masks: [T*N, 1] -> [T, N]
            masks = masks.view(T, N)

            # 找哪些 time step 有 done（mask == 0），用于分段跑 RNN
            has_zeros = ((masks[1:] == 0.0)
                         .any(dim=-1)
                         .nonzero()
                         .squeeze()
                         .cpu())

            # +1 to correct masks[1:]
            if has_zeros.dim() == 0:
                # 单个 index 的情况
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # 把 t=0 和 t=T 塞进去形成区间边界
            has_zeros = [0] + has_zeros + [T]

            # h_obs: [N, H] -> [1, N, H]
            h = h_obs.unsqueeze(0)
            outputs = []

            for i in range(len(has_zeros) - 1):
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                # masks[start_idx]: [N]
                # h * masks[...] : 把某些 env 的 hidden 清零（episode reset）
                rnn_scores, h = self.rnn(
                    x[start_idx:end_idx],                           # [L, N, obs_dim]
                    h * masks[start_idx].view(1, -1, 1)            # [1, N, H]
                )
                outputs.append(rnn_scores)                         # [L, N, H]

            # 拼回完整序列：x_seq: [T, N, H]
            x_seq = torch.cat(outputs, dim=0)
            # 展平成 [T*N, H]
            x_flat = x_seq.view(T * N, -1)
            # 最终 hidden: [1, N, H] -> [N, H]
            h_obs = h.squeeze(0)

            mu = self.head_mu(x_flat)         # [T*N, 2]
            logvar = self.head_logvar(x_flat) # [T*N, 2]
            return mu, logvar, h_obs
        
        
class Policy(nn.Module):
    def __init__(self, obs_shape, action_space,
                 observer_obs_dim,
                 base_obs_dim,
                 base_kwargs=None, args=None): 
        super().__init__()
        # Wind obsver v1 modification: observer_obs_dim
        if base_kwargs is None:
            base_kwargs = {}

        # wind observer
        self.observer = WindObserverModule(observer_obs_dim, hidden_size=32)

        # base network (policy + value + representation)
        WIND_FEATURE_DIM = 4  # wind_mu(2) + wind_logvar(2)
        self.base = MLPBase(base_obs_dim + WIND_FEATURE_DIM, **base_kwargs) 

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
            if args is not None: 
                if args.if_train_actor_std:
                    print("Using DiagGaussian with trainable std!")
                    self.dist = DiagGaussian(self.base.output_size, num_outputs, trainable_std_dev=args.if_train_actor_std)
                if args.betadist:
                    print("Using BetaCustom distribution!")
                    self.dist = BetaCustom(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size
    
    @property
    def observer_hidden_state_size(self):
        """Size of h_obs."""
        return self.observer.hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    @staticmethod
    def split_observer_base_obs(full_obs: torch.Tensor):
        # Wind obsver v2 modification: split observer and base observations
        """
        Minimal split:
        - observer sees all inputs except index 2
        - base sees all original inputs (full_obs)

        full_obs: [B, D]
        returns:
            obs_wind_module:  [B, D-1]  (drop dim=2)
            obs_base: [B, D]    (unchanged)
        """
        # Check shape
        assert full_obs.dim() == 2, f"Expected full_obs shape [B, D], got {full_obs.shape}"

        B, D = full_obs.shape
        assert D > 2, f"Need at least 3 dims to drop index 2, got D={D}"

        # Take out odor from wind observer inputs
        # Equivalent to concat([0,1], [3,4,...,D-1])
        keep_left  = full_obs[:, :2]          # [B, 2]
        keep_right = full_obs[:, 3:]          # [B, D-3] (if D=3, this is empty)
        obs_wind_module = torch.cat([keep_left, keep_right], dim=-1)  # [B, D-1]

        obs_base = full_obs  # base sees full input

        return obs_wind_module, obs_base
        
    def act(self, full_obs, rnn_hxs, h_wind_module, masks, deterministic=False, vr_wind_mod=False):
        # Wind obsver v2 modification: split obs for observer and base; 
        # full_obs: [B, full_dim]
        if full_obs.shape[-1] == 11: # during vrvr, obs passed in with predtermined wind predictor outputs; split if wind in inputs, no matter if vr_wind_mod
            vr_wind_mu = full_obs[:, -4:-2]
            vr_wind_logvar = full_obs[:, -2:]
            full_obs = full_obs[:, :-4]  # remove wind pred inputs for observer 

        obs_wind_module, obs_base = self.split_observer_base_obs(full_obs) # TODO: static version for now. Set obs later.
        
        # 1) Wind inputs into observer
        wind_mu, wind_logvar, h_wind_module = self.observer(obs_wind_module, h_wind_module, masks)

        # 2) General inputs into base, concatenated with wind belief
        if full_obs.shape[-1] == 11 and vr_wind_mod:
            wind_mu = wind_mu.clone()
            wind_logvar = wind_logvar.clone()

            mu_valid = ~torch.isnan(vr_wind_mu)
            logvar_valid = ~torch.isnan(vr_wind_logvar)

            wind_mu[mu_valid] = vr_wind_mu[mu_valid]
            wind_logvar[logvar_valid] = vr_wind_logvar[logvar_valid]

        base_in = torch.cat([obs_base, wind_mu.detach(), wind_logvar.detach()], dim=-1) # 如果你想让 observer 只走 aux loss，不走 RL 梯度就 detach；否则去掉 detach
        value, actor_features, rnn_hxs, activities = self.base(base_in, rnn_hxs, masks)
        
        # 3) Store wind predictions in activities for logging
        activities['wind_mu'] = wind_mu
        activities['wind_logvar'] = wind_logvar
        if vr_wind_mod:
            activities['vr_wind_mu'] = vr_wind_mu
            activities['vr_wind_logvar'] = vr_wind_logvar

        dist = self.dist(actor_features)
        action = dist.mode() if deterministic else dist.sample()
        action_log_probs = dist.log_probs(action)

        return value, action, action_log_probs, (rnn_hxs, h_wind_module), activities

    def get_value(self, inputs, rnn_hxs, h_wind_module, masks):
        '''return the value estimation given inputs and states
        Ran in training loop for critic loss computation of the last step. We don't store wind predictions there so recompute.
        '''
        # full_obs: [B, D]
        obs_wind_module, obs_base = self.split_observer_base_obs(inputs)

        wind_mu, wind_logvar, h_wind_module = self.observer(obs_wind_module, h_wind_module, masks)

        base_in = torch.cat([obs_base, wind_mu.detach(), wind_logvar.detach()], dim=-1)

        value, _, rnn_hxs, _ = self.base(base_in, rnn_hxs, masks)
        return value

    def evaluate_actions(self, base_inputs, rnn_hxs, masks, action):
        '''Run the base module to evaluate actions (for actor loss) in PPO. Ran wind prediction upstream and the inputs here contain predictions already. 
        Return action log prob and entropy of actor distribution for PPO update
        
        base_inputs: includes base observations and wind predictions 
        '''
        value, actor_features, rnn_hxs, activities = self.base(base_inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs, activities
    
    def reset_actor(self):
        """
        Reset the actor part of the base network.
        This is useful for resetting the actor weights after training.
        """
        self.base.reset_actor()
        print("Actor weights reset.", flush=True)
        
    def reset_critic(self):
        """
        Reset the critic part of the base network.
        This is useful for resetting the critic weights after training.
        """
        self.base.reset_critic()
        print("Critic weights reset.", flush=True)
        
    def print_weights(self, head=""):
        """
        Print the actor weights of the actor_critic network.
        This is useful for debugging and understanding the model.
        """
        if head:
            print(f"{head} Weights:")
            for k, v in self.actor_critic.state_dict().items():
                if head in k:
                    print(f"{k}: {tuple(v)}")
        else:
            # Iterate through the state_dict and print only actor weights
            for k, v in self.actor_critic.state_dict().items():
                print(f"{k}: {tuple(v)}")


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size, rnn_type):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent
        if hidden_size == 16:
            layer_size = 250
        elif hidden_size == 2:
            layer_size = 2000

        if recurrent:
            print("hidden_size", hidden_size)
            if rnn_type == 'VRNN':
                print("Using VanillaRNN")
                self.rnn = nn.RNN(recurrent_input_size, hidden_size)
            elif rnn_type == 'GRU':
                print("Using GRU")
                self.rnn = nn.GRU(recurrent_input_size, hidden_size)
            elif rnn_type == 'GNODE':
                print("Using GNODE")
                self.rnn = GNODE(recurrent_input_size, hidden_size, layer_size)
            
            if rnn_type == 'GNODE':
                for name, param in self.rnn.named_parameters():
                        if 'bias' in name:
                            nn.init.constant_(param, 0) 
                        elif 'weight' in name:
                            if len(param.shape) == 2:  # Linear layer
                                fan_in = param.shape[1]
                                nn.init.trunc_normal_(param, mean=0.0, std=np.sqrt(1.414/fan_in))
            else:
                for name, param in self.rnn.named_parameters():
                    if 'bias' in name:
                        nn.init.constant_(param, 0) 
                    elif 'weight' in name:
                        # nn.init.orthogonal_(param)
                        # nn.init.normal_(param, mean=0.0, std=1./hidden_size)
                        nn.init.normal_(param, mean=0.0, std=1./np.sqrt(hidden_size))


    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_rnn(self, x, hxs, masks, input_masks = None):
        if x.size(0) == hxs.size(0):
            if input_masks is not None:
                input_masks = input_masks.to(x.device, x.dtype)
                x = x * input_masks # used for silencing inputs
            x, hxs = self.rnn(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # print("NOTE ----------------- Using masks for VecEnvs")
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.rnn(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs

class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64, rnn_type='GRU'):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size, rnn_type)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor1 = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh())
        self.actor = nn.Sequential(
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic1 = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh())
        self.critic = nn.Sequential(
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))
        
        # Wind obsver v2 : no wind prediction heads - handled in observer module
        # self.wind_mu_head = nn.Linear(hidden_size, 2)       # [cosθ, sinθ]
        # self.wind_logvar_head = nn.Linear(hidden_size, 2)   # per-dim log-variance


        self.train() # tells your model that you are training the model not eval().
    
    def forward(self, inputs, rnn_hxs, masks, input_masks = None):
        x = inputs
        if self.is_recurrent:
            x, rnn_hxs = self._forward_rnn(x, rnn_hxs, masks, input_masks=input_masks)

        hx1_critic = self.critic1(x)
        hx1_actor = self.actor1(x)
        hidden_critic = self.critic(hx1_critic)
        hidden_actor = self.actor(hx1_actor)

        value = self.critic_linear(hidden_critic)
        
        # Wind obsver v2 : no wind prediction in base 
        # wind_mu = self.wind_mu_head(hidden_actor)
        # wind_logvar = self.wind_logvar_head(hidden_actor)

        
        activities = {
            'rnn_hxs': rnn_hxs,
            'hx1_actor': hx1_actor,
            'hx1_critic': hx1_critic,
            'hidden_actor': hidden_actor,
            'hidden_critic': hidden_critic,
            'value': value,            
        }

        return value, hidden_actor, rnn_hxs, activities
    
    def reset_actor(self):
        def init_(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        
        for layer in self.actor1:
            if isinstance(layer, nn.Linear):
                init_(layer)
        
        for layer in self.actor:
            if isinstance(layer, nn.Linear):
                init_(layer)

    def reset_critic(self):
        def init_(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        
        for layer in self.critic1:
            if isinstance(layer, nn.Linear):
                init_(layer)
        
        for layer in self.critic:
            if isinstance(layer, nn.Linear):
                init_(layer)

class Simple_MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64, rnn_type='GRU'):
        super(Simple_MLPBase, self).__init__(recurrent, num_inputs, hidden_size, rnn_type)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Identity())

        self.critic1 = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.ReLU())
        self.critic = nn.Sequential(
            init_(nn.Linear(hidden_size, hidden_size)), nn.ReLU())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train() # tells your model that you are training the model not eval().

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_rnn(x, rnn_hxs, masks)

        hx1_critic = self.critic1(x)
        hx1_actor = self.actor(x)
        hidden_actor = hx1_actor
        hidden_critic = self.critic(hx1_critic)

        value = self.critic_linear(hidden_critic)
        
        activities = {
            'rnn_hxs': rnn_hxs,
            'hx1_actor': hx1_actor,
            'hx1_critic': hx1_critic,
            'hidden_actor': hidden_actor,
            'hidden_critic': hidden_critic,
            'value': value,
        }

        return value, hidden_actor, rnn_hxs, activities