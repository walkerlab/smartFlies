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
                 weight_decay=0,
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
        # Weight of the wind-observer NLL auxiliary loss; 0 disables it entirely
        # (no wind term enters backward(), matching a plain actor-critic update).
        self.wind_loss_coef = wind_loss_coef

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        clip_fraction_epoch = 0
        wind_loss_epoch = 0
        all_wind_nll = []
        all_wind_sqerr = []
        all_wind_logvar = []
        checked_ou_log_probs = False
        ou_configured = getattr(self.actor_critic, 'ou_configured', False)
        if hasattr(self.actor_critic, 'ou_sigma_current'):
            print(f"  [OU] sigma={self.actor_critic.ou_sigma_current:.4f}", flush=True)

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ, wind_targets_batch, ou_states_batch = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _, activities = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,
                    actions_batch, ou_state=ou_states_batch)
                # Check during development that OU log_probs are consistent with the old_action_log_probs_batch, which are computed in a single thread and stored in the rollouts. If there is a discrepancy, it may indicate an issue with how ou_state is being handled across threads.
                if ou_configured and not checked_ou_log_probs:
                    max_log_prob_diff = (action_log_probs - old_action_log_probs_batch).abs().max().item()
                    print(f"  [OU] log_prob_check max_abs_diff={max_log_prob_diff:.6g}", flush=True)
                    if max_log_prob_diff > 1e-3:
                        print(f"  [OU] WARNING: log_prob discrepancy {max_log_prob_diff:.6g} exceeds 1e-3 — "
                              f"check ou_state threading", flush=True)
                    checked_ou_log_probs = True

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

                total_loss = (value_loss * self.value_loss_coef
                              + action_loss
                              - dist_entropy * self.entropy_coef)

                if self.wind_loss_coef > 0:
                    wind_mu = activities['wind_mu']         # [B, 2]
                    wind_logvar = activities['wind_logvar']  # [B, 2]
                    wind_loss, wind_nll_per = wind_nll_stats(wind_mu, wind_logvar, wind_targets_batch)  # mean + [B]
                    wind_sqerr_per = ((wind_mu - wind_targets_batch) ** 2).sum(-1)  # [B]
                    wind_logvar_per = wind_logvar.mean(-1)  # per-sample mean over the 2 dims -> [B]

                    all_wind_nll.append(wind_nll_per.detach().cpu())
                    all_wind_sqerr.append(wind_sqerr_per.detach().cpu())
                    all_wind_logvar.append(wind_logvar_per.detach().cpu())

                    total_loss = total_loss + self.wind_loss_coef * wind_loss
                    wind_loss_epoch += wind_loss.item()

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                clip_fraction_epoch += clip_fraction.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        clip_fraction_epoch /= num_updates

        # Wind-observer diagnostics; None when the aux loss is disabled.
        if self.wind_loss_coef > 0:
            extras = {
                "wind_loss_epoch": wind_loss_epoch / num_updates,
                "wind_nll_all": torch.cat(all_wind_nll, dim=0) if all_wind_nll else torch.tensor([]),
                "wind_sqerr_all": torch.cat(all_wind_sqerr, dim=0) if all_wind_sqerr else torch.tensor([]),
                "wind_logvar_all": torch.cat(all_wind_logvar, dim=0) if all_wind_logvar else torch.tensor([]),
            }
        else:
            extras = None

        if self.track_ppo_fraction:
            return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, clip_fraction_epoch, advantages.flatten(), extras
        else:
            return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, advantages.flatten(), extras
