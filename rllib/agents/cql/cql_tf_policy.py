"""
TF policy class used for CQL.
"""
import numpy as np
import gym
import logging
from typing import Dict

import ray
import ray.experimental.tf_utils
from ray.rllib.agents.sac.sac_tf_policy import postprocess_trajectory, build_sac_model, \
    ActorCriticOptimizerMixin, ComputeTDErrorMixin, TargetNetworkMixin, \
    get_distribution_inputs_and_class, get_dist_class, stats, gradients, apply_gradients
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_tf, try_import_tfp
from ray.rllib.utils.typing import TensorType, TrainerConfigDict

tf = try_import_tf()
tfp = try_import_tfp()

logger = logging.getLogger(__name__)


# Returns policy tiled actions and log probabilities for CQL Loss
def policy_actions_repeat(model, action_dist, obs, num_repeat=1):
    obs_temp = tf.tile(obs, [num_repeat, 1])
    policy_dist = action_dist(model.get_policy_output(obs_temp), model)
    actions = policy_dist.sample()
    log_p = tf.expand_dims(policy_dist.logp(actions), -1)
    return actions, tf.squeeze(log_p, axis=len(log_p.shape) - 1)


def q_values_repeat(model, obs, actions, twin=False):
    action_shape = actions.shape[0]
    obs_shape = obs.shape[0]
    num_repeat = int(action_shape / obs_shape)
    obs_temp = tf.tile(obs, [num_repeat, 1])
    if twin:
        preds = model.get_q_values(obs_temp, actions)
    else:
        preds = model.get_twin_q_values(obs_temp, actions)
    preds = tf.reshape(preds, [obs.shape[0], num_repeat, 1])
    return preds


def cql_loss(policy: Policy, model: ModelV2, _, train_batch: SampleBatch):
    print(policy.cur_iter)
    policy.cur_iter += 1
    # For best performance, turn deterministic off
    deterministic = policy.config["_deterministic_loss"]
    twin_q = policy.config["twin_q"]
    discount = policy.config["gamma"]

    # CQL Parameters
    bc_iters = policy.config["bc_iters"]
    cql_temp = policy.config["temperature"]
    num_actions = policy.config["num_actions"]
    min_q_weight = policy.config["min_q_weight"]
    use_lagrange = policy.config["lagrangian"]
    target_action_gap = policy.config["lagrangian_thresh"]

    obs = train_batch[SampleBatch.CUR_OBS]
    actions = train_batch[SampleBatch.ACTIONS]
    rewards = train_batch[SampleBatch.REWARDS]
    next_obs = train_batch[SampleBatch.NEXT_OBS]
    terminals = train_batch[SampleBatch.DONES]

    model_out_t, _ = model({
        "obs": obs,
        "is_training": True,
    }, [], None)

    model_out_tp1, _ = model({
        "obs": next_obs,
        "is_training": True,
    }, [], None)

    target_model_out_tp1, _ = policy.target_model({
        "obs": next_obs,
        "is_training": True,
    }, [], None)

    action_dist_class = get_dist_class(policy.config, policy.action_space)
    action_dist_t = action_dist_class(
        model.get_policy_output(model_out_t), policy.model)
    policy_t = action_dist_t.sample() if not deterministic else \
        action_dist_t.deterministic_sample()
    log_pis_t = tf.expand_dims(action_dist_t.logp(policy_t), -1)

    # Unlike original SAC, Alpha and Actor Loss are computed first.
    # Alpha Loss
    alpha_loss = -(model.log_alpha *
                   (log_pis_t + model.target_entropy).detach()).mean()

    # Policy Loss (Either Behavior Clone Loss or SAC Loss)
    if policy.cur_iter >= bc_iters:
        min_q = model.get_q_values(model_out_t, policy_t)
        if twin_q:
            twin_q = model.get_twin_q_values(model_out_t, policy_t)
            min_q = tf.reduce_min((min_q, twin_q), axis=0)
        actor_loss = (model.alpha.detach() * log_pis_t - min_q).mean()
    else:
        bc_logp = action_dist_t.logp(actions)
        actor_loss = (model.alpha * log_pis_t - bc_logp).mean()

    # Critic Loss (Standard SAC Critic L2 Loss + CQL Entropy Loss)
    # SAC Loss
    action_dist_tp1 = action_dist_class(
        model.get_policy_output(model_out_tp1), policy.model)
    policy_tp1 = action_dist_tp1.sample() if not deterministic else \
        action_dist_tp1.deterministic_sample()

    # Q-values for the batched actions.
    q_t = model.get_q_values(model_out_t, train_batch[SampleBatch.ACTIONS])
    q_t = tf.squeeze(q_t, axis=len(q_t.shape) - 1)
    if twin_q:
        twin_q_t = model.get_twin_q_values(model_out_t,
                                           train_batch[SampleBatch.ACTIONS])
        twin_q_t = tf.squeeze(twin_q_t, axis=len(twin_q_t.shape) - 1)

    # Target q network evaluation.
    q_tp1 = policy.target_model.get_q_values(target_model_out_tp1, policy_tp1)
    if twin_q:
        twin_q_tp1 = policy.target_model.get_twin_q_values(
            target_model_out_tp1, policy_tp1)
        # Take min over both twin-NNs.
        q_tp1 = tf.reduce_min(
            (q_tp1, twin_q_tp1), axis=0)
    q_tp1 = tf.squeeze(q_tp1, axis=len(q_tp1.shape) - 1)
    q_tp1 = (1.0 - terminals.float()) * q_tp1

    # compute RHS of bellman equation
    q_t_target = (
        rewards + (discount**policy.config["n_step"]) * q_tp1).detach()

    # Compute the TD-error (potentially clipped), for priority replay buffer
    base_td_error = tf.abs(q_t - q_t_target)
    if twin_q:
        twin_td_error = tf.abs(twin_q_t - q_t_target)
        td_error = 0.5 * (base_td_error + twin_td_error)
    else:
        td_error = base_td_error
    critic_loss = [tf.losses.mean_squared_error(labels=q_t_target, predictions=q_t)]
    if twin_q:
        critic_loss.append(tf.losses.mean_squared_error(labels=q_t_target,
                                                        predictions=twin_q_t))

    # CQL Loss (We are using Entropy version of CQL (the best version))
    rand_actions = policy._unif_dist.sample([actions.shape[0] * num_actions,
                                             actions.shape[-1]])
    curr_actions, curr_logp = policy_actions_repeat(model, action_dist_class,
                                                    obs, num_actions)
    next_actions, next_logp = policy_actions_repeat(model, action_dist_class,
                                                    next_obs, num_actions)
    curr_logp = curr_logp.view(actions.shape[0], num_actions, 1)
    next_logp = next_logp.view(actions.shape[0], num_actions, 1)

    q1_rand = q_values_repeat(model, model_out_t, rand_actions)
    q1_curr_actions = q_values_repeat(model, model_out_t, curr_actions)
    q1_next_actions = q_values_repeat(model, model_out_t, next_actions)

    if twin_q:
        q2_rand = q_values_repeat(model, model_out_t, rand_actions, twin=True)
        q2_curr_actions = q_values_repeat(
            model, model_out_t, curr_actions, twin=True)
        q2_next_actions = q_values_repeat(
            model, model_out_t, next_actions, twin=True)

    random_density = np.log(0.5**curr_actions.shape[-1])
    cat_q1 = tf.concat([
        q1_rand - random_density, q1_next_actions - next_logp.detach(),
        q1_curr_actions - curr_logp.detach()
    ], 1)
    if twin_q:
        cat_q2 = tf.concat([
            q2_rand - random_density, q2_next_actions - next_logp.detach(),
            q2_curr_actions - curr_logp.detach()
        ], 1)

    min_qf1_loss = tf.reduce_logsumexp(
        cat_q1 / cql_temp, axis=1).mean() * min_q_weight * cql_temp
    min_qf1_loss = min_qf1_loss - q_t.mean() * min_q_weight
    if twin_q:
        min_qf2_loss = tf.reduce_logsumexp(
            cat_q2 / cql_temp, axis=1).mean() * min_q_weight * cql_temp
        min_qf2_loss = min_qf2_loss - twin_q_t.mean() * min_q_weight

    if use_lagrange:
        alpha_prime = tf.clip_by_value(
            model.log_alpha_prime.exp(), clip_value_min=0.0, clip_value_max=1000000.0)[0]
        min_qf1_loss = alpha_prime * (min_qf1_loss - target_action_gap)
        if twin_q:
            min_qf2_loss = alpha_prime * (min_qf2_loss - target_action_gap)
            alpha_prime_loss = 0.5 * (-min_qf1_loss - min_qf2_loss)
        else:
            alpha_prime_loss = -min_qf1_loss

    cql_loss = [min_qf2_loss]
    if twin_q:
        cql_loss.append(min_qf2_loss)

    critic_loss[0] += min_qf1_loss
    if twin_q:
        critic_loss[1] += min_qf2_loss

    # Save for stats function.
    policy.q_t = q_t
    policy.policy_t = policy_t
    policy.log_pis_t = log_pis_t
    policy.td_error = td_error
    policy.actor_loss = actor_loss
    policy.critic_loss = critic_loss
    policy.alpha_loss = alpha_loss
    policy.log_alpha_value = model.log_alpha
    policy.alpha_value = model.alpha
    policy.target_entropy = model.target_entropy
    # CQL Stats
    policy.cql_loss = cql_loss
    if use_lagrange:
        policy.log_alpha_prime_value = model.log_alpha_prime[0]
        policy.alpha_prime_value = alpha_prime
        policy.alpha_prime_loss = alpha_prime_loss

    # Return all loss terms corresponding to our optimizers.
    if use_lagrange:
        return tuple([policy.actor_loss] + policy.critic_loss +
                     [policy.alpha_loss] + [policy.alpha_prime_loss])
    return tuple([policy.actor_loss] + policy.critic_loss +
                 [policy.alpha_loss])


def cql_stats(policy: Policy,
              train_batch: SampleBatch) -> Dict[str, TensorType]:
    sac_dict = stats(policy, train_batch)
    sac_dict["cql_loss"] = tf.reduce_mean(tf.stack(policy.cql_loss))
    if policy.config["lagrangian"]:
        sac_dict["log_alpha_prime_value"] = policy.log_alpha_prime_value
        sac_dict["alpha_prime_value"] = policy.alpha_prime_value
        sac_dict["alpha_prime_loss"] = policy.alpha_prime_loss
    return sac_dict


#TODO: Adapt this into a class CQLActorCriticOptimizerMixin
# def cql_optimizer_fn(policy: Policy, config: TrainerConfigDict) -> \
#         Tuple[LocalOptimizer]:
#     policy.cur_iter = 0
#     opt_list = optimizer_fn(policy, config)
#     if config["lagrangian"]:
#         log_alpha_prime = nn.Parameter(
#             torch.zeros(1, requires_grad=True).float())
#         policy.model.register_parameter("log_alpha_prime", log_alpha_prime)
#         policy.alpha_prime_optim = torch.optim.Adam(
#             params=[policy.model.log_alpha_prime],
#             lr=config["optimization"]["critic_learning_rate"],
#             eps=1e-7,  # to match tf.keras.optimizers.Adam's epsilon default
#         )
#         return tuple([policy.actor_optim] + policy.critic_optims +
#                      [policy.alpha_optim] + [policy.alpha_prime_optim])
#     return opt_list

class CQLActorCriticOptimizerMixin(ActorCriticOptimizerMixin):
    def __init__(self, config):
        super().__init__(config)
        if config["lagrangian"]:
            self._alpha_prime_optimizer = tf.train.AdamOptimizer(
                learning_rate=config["optimization"]["critic_learning_rate"])


def cql_setup_early_mixins(policy, obs_space, action_space, config):
    CQLActorCriticOptimizerMixin.__init__(policy, config)


def cql_setup_mid_mixins(policy, obs_space, action_space, config):
    action_low = policy.model.action_space.low[0]
    action_high = policy.model.action_space.high[0]
    policy._unif_dist = tfp.Uniform(action_low, action_high)
    ComputeTDErrorMixin.__init__(policy, cql_loss)


def cql_setup_late_mixins(policy, obs_space: gym.spaces.Space,
                          action_space: gym.spaces.Space,
                          config: TrainerConfigDict) -> None:
    TargetNetworkMixin.__init__(policy, config)
    if config["lagrangian"]:
        policy.model.log_alpha_prime = policy.model.log_alpha_prime.to(
            policy.device)


# Build a child class of `TorchPolicy`, given the custom functions defined
# above.
CQLTFPolicy = build_tf_policy(
    name="CQLTFPolicy",
    get_default_config=lambda: ray.rllib.agents.cql.cql.CQL_DEFAULT_CONFIG,
    make_model=build_sac_model,
    postprocess_fn=postprocess_trajectory,
    action_distribution_fn=get_distribution_inputs_and_class,
    loss_fn=cql_loss,
    stats_fn=cql_stats,
    gradients_fn=gradients,
    apply_gradients_fn=apply_gradients,
    extra_learn_fetches_fn=lambda policy: {"td_error": policy.td_error},
    mixins=[
        TargetNetworkMixin, CQLActorCriticOptimizerMixin, ComputeTDErrorMixin
    ],
    before_init=cql_setup_early_mixins,
    before_loss_init=cql_setup_mid_mixins,
    after_init=cql_setup_late_mixins,
    obs_include_prev_action_reward=False,
)
