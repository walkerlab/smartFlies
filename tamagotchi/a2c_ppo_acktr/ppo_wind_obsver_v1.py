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
                 wind_loss_coef=0.0):

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

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        clip_fraction_epoch = 0
        wind_loss_epoch = 0   # Wind obsver v1 modification: track wind loss epoch
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
                # Wind obsver v1 modification: grab wind targets from data generator
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ, wind_targets_batch = sample
                # Wind obsver v1 modification: grab activities which contain predicted wind mu/logvar
                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, rnn_hxs, activities = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,
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

                # Wind obsver v1 modification: grab wind targets from activities
                wind_mu     = activities['wind_mu']       # [B,2]
                wind_logvar = activities['wind_logvar']   # [B,2]
                # Compute wind NLL loss
                wind_loss, wind_nll_per = wind_nll_stats(wind_mu, wind_logvar, wind_targets_batch)  # mean + [B]
                wind_sqerr_per = ((wind_mu - wind_targets_batch) ** 2).sum(-1)   # [B]
                wind_logvar_per = wind_logvar.mean(-1)  # 每个样本把2维取平均 → [B]

                all_wind_nll.append(wind_nll_per.detach().cpu())
                all_wind_sqerr.append(wind_sqerr_per.detach().cpu())
                all_wind_logvar.append(wind_logvar_per.detach().cpu())

                total_loss = (value_loss * self.value_loss_coef
                              + action_loss
                              - dist_entropy * self.entropy_coef
                              + self.wind_loss_coef * wind_loss)
                self.optimizer.zero_grad()
                total_loss.backward()
                
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                clip_fraction_epoch += clip_fraction.item()
                wind_loss_epoch += wind_loss.item()   # Wind obsver v1 modification: accumulate wind loss

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
