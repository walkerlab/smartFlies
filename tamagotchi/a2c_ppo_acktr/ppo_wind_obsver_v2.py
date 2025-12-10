import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tamagotchi.a2c_ppo_acktr.utils import wind_nll_stats


class PPO():
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 track_ppo_fraction=True,
                 weight_decay=0, # 1e-4 per default across all as of 11/17/25
                 wind_loss_coef=0.0,
                 wind_lr=None):  # Wind obsver v2 modification: wind optimizer lr - option to set different from main lr

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps, weight_decay=weight_decay)
        self.track_ppo_fraction = track_ppo_fraction    
        # Wind obsver v1 modification: track wind loss coef
        self.wind_loss_coef = wind_loss_coef
        
        # Wind obsver v2 modification: RL optimizer：只更新 base + dist（policy / value）
        if wind_lr is None:
            wind_lr = lr  # 不同也可以，这里先设成一样

        base_params = list(actor_critic.base.parameters())
        base_params += list(actor_critic.dist.parameters())

        self.optimizer = optim.Adam(
            base_params, lr=lr, eps=eps, weight_decay=weight_decay
        )

        # Wind obsver v2 modification: wind optimizer：只更新 observer
        obs_params = list(actor_critic.observer.parameters())
        self.wind_optimizer = optim.Adam(
            obs_params, lr=wind_lr, eps=eps, weight_decay=0.0
        )
        
    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        clip_fraction_epoch = 0
        wind_loss_epoch = 0
        all_wind_nll = []
        all_wind_sqerr = []
        all_wind_logvar = []

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                # Wind obsver v2 modification: grab wind targets and module-specific inputs from data generator
                (obs_batch, # all inputs - to be split 
                   recurrent_hidden_states_batch,
                   observer_hidden_states_batch,
                   actions_batch,
                   value_preds_batch,
                   return_batch,
                   masks_batch,
                   old_action_log_probs_batch,
                   adv_targ,
                   wind_targets_batch,) = sample  # Wind observer hidden state
                # Wind obsver v2 modification: separate updates for observer and base module + policy 
                # Ssplit obs_batch into base module inputs and wind observer inputs               
                obs_wind_module_batch, obs_base_batch = self.actor_critic.split_observer_base_obs(obs_batch)
                # ================================
                # 1) wind aux update (observer only)
                # ================================
                # Always run and train this since the latter requries its outputs
                wind_mu, wind_logvar, _ = self.actor_critic.observer(
                    obs_wind_module_batch, observer_hidden_states_batch, masks_batch)

                wind_loss, wind_nll_per = wind_nll_stats(
                    wind_mu, wind_logvar, wind_targets_batch)

                wind_sqerr_per = ((wind_mu - wind_targets_batch) ** 2).sum(-1)
                wind_logvar_per = wind_logvar.mean(-1)

                all_wind_nll.append(wind_nll_per.detach().cpu())
                all_wind_sqerr.append(wind_sqerr_per.detach().cpu())
                all_wind_logvar.append(wind_logvar_per.detach().cpu())

                scaled_wind_loss = self.wind_loss_coef * wind_loss
                wind_loss_epoch += scaled_wind_loss.item()

                self.wind_optimizer.zero_grad()
                scaled_wind_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.observer.parameters(),
                                            self.max_grad_norm)
                self.wind_optimizer.step()


                # ========================
                # 2) RL update (base only)
                # ========================
                # Get values, action log probs, dist entropy from base module
                # base_in = torch.cat([obs_base, wind_mu.detach(), wind_logvar.detach()], dim=-1)
                obs_base_batch = torch.cat([obs_base_batch, wind_mu.detach(), wind_logvar.detach()], dim=-1) # TODO: correct to detach here?
                values, action_log_probs, dist_entropy, rnn_hxs, _ = \
                    self.actor_critic.evaluate_actions(
                        obs_base_batch,             # Base module inputs + wind obsver outputs
                        recurrent_hidden_states_batch,
                        masks_batch,
                        actions_batch)

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                clip_fraction = ((ratio > (1.0 + self.clip_param)) | (ratio < (1.0 - self.clip_param))).float().mean()
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                # Wind obsver v2 modification: total RL loss (no wind loss here)
                rl_total_loss = (value_loss * self.value_loss_coef
                                 + action_loss
                                 - dist_entropy * self.entropy_coef)

                self.optimizer.zero_grad()
                rl_total_loss.backward()
                nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]["params"],
                         self.max_grad_norm) # TODO: check when debug - see if param_groups[0] is correctly shaped
                self.optimizer.step()

                # ======= 日志累积 RL 指标 =======
                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                clip_fraction_epoch += clip_fraction.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        clip_fraction_epoch /= num_updates
        wind_loss_epoch /= num_updates    # Wind obsver v1 modification: average wind loss
        
        # wind obsver v1 modification: return wind loss
        if len(all_wind_nll) > 0:
            all_wind_nll = torch.cat(all_wind_nll, dim=0)          # [N_total]
            all_wind_sqerr = torch.cat(all_wind_sqerr, dim=0)      # [N_total]
            all_wind_logvar = torch.cat(all_wind_logvar, dim=0)    # [N_total]
        else:
            all_wind_nll = torch.tensor([])
            all_wind_sqerr = torch.tensor([])
            all_wind_logvar = torch.tensor([])
                
        extras = {
            "wind_loss_epoch": wind_loss_epoch,
            "wind_nll_all": all_wind_nll,           # [N_total]
            "wind_sqerr_all": all_wind_sqerr,       # [N_total]
            "wind_logvar_all": all_wind_logvar,     # [N_total]
        }

        if self.track_ppo_fraction:
            return (value_loss_epoch,
                    action_loss_epoch,
                    dist_entropy_epoch,
                    clip_fraction_epoch,
                    advantages.flatten(),
                    extras) # wind obsver v1 modification: return wind loss
        else:
            return (value_loss_epoch,
                    action_loss_epoch,
                    dist_entropy_epoch,
                    advantages.flatten(),
                    extras) # wind obsver v1 modification: return wind loss
