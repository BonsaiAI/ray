import logging

import ray
import tensorflow as tf
from ray.rllib.agents.sac.proxy.models.model_v2 import ModelV2
from ray.rllib.evaluation import TFPolicyGraph
from ray.rllib.evaluation.policy_graph import PolicyGraph
from ray.rllib.utils.annotations import DeveloperAPI, override
from ray.rllib.utils.debug import log_once, summarize

logger = logging.getLogger(__name__)

__all__ = ["TFPolicy"]

ACTION_PROB = "action_prob"
ACTION_LOGP = "action_logp"


class TFPolicy(TFPolicyGraph):

    @DeveloperAPI
    def __init__(self,
                 observation_space,
                 action_space,
                 sess,
                 obs_input,
                 action_sampler,
                 loss,
                 loss_inputs,
                 model=None,
                 action_prob=None,
                 state_inputs=None,
                 state_outputs=None,
                 prev_action_input=None,
                 prev_reward_input=None,
                 seq_lens=None,
                 max_seq_len=20,
                 batch_divisibility_req=1,
                 update_ops=None):
        """Initialize the policy graph.

        Arguments:
            observation_space (gym.Space): Observation space of the env.
            action_space (gym.Space): Action space of the env.
            sess (Session): TensorFlow session to use.
            obs_input (Tensor): input placeholder for observations, of shape
                [BATCH_SIZE, obs...].
            action_sampler (Tensor): Tensor for sampling an action, of shape
                [BATCH_SIZE, action...]
            loss (Tensor): scalar policy loss output tensor.
            loss_inputs (list): a (name, placeholder) tuple for each loss
                input argument. Each placeholder name must correspond to a
                SampleBatch column key returned by postprocess_trajectory(),
                and has shape [BATCH_SIZE, data...]. These keys will be read
                from postprocessed sample batches and fed into the specified
                placeholders during loss computation.
            model (rllib.models.Model): used to integrate custom losses and
                stats from user-defined RLlib models.
            action_prob (Tensor): probability of the sampled action.
            state_inputs (list): list of RNN state input Tensors.
            state_outputs (list): list of RNN state output Tensors.
            prev_action_input (Tensor): placeholder for previous actions
            prev_reward_input (Tensor): placeholder for previous rewards
            seq_lens (Tensor): placeholder for RNN sequence lengths, of shape
                [NUM_SEQUENCES]. Note that NUM_SEQUENCES << BATCH_SIZE. See
                models/lstm.py for more information.
            max_seq_len (int): max sequence length for LSTM training.
            batch_divisibility_req (int): pad all agent experiences batches to
                multiples of this value. This only has an effect if not using
                a LSTM model.
            update_ops (list): override the batchnorm update ops to run when
                applying gradients. Otherwise we run all update ops found in
                the current variable scope.
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.model = model
        self._sess = sess
        self._obs_input = obs_input
        self._prev_action_input = prev_action_input
        self._prev_reward_input = prev_reward_input
        self._sampler = action_sampler
        self._is_training = self._get_is_training_placeholder()
        self._action_prob = action_prob
        self._state_inputs = state_inputs or []
        self._state_outputs = state_outputs or []
        self._seq_lens = seq_lens
        self._max_seq_len = max_seq_len
        self._batch_divisibility_req = batch_divisibility_req
        self._update_ops = update_ops
        self._stats_fetches = {}
        self._loss_input_dict = None

        if loss is not None:
            self._initialize_loss(loss, loss_inputs)
        else:
            self._loss = None

        if len(self._state_inputs) != len(self._state_outputs):
            raise ValueError(
                "Number of state input and output tensors must match, got: "
                "{} vs {}".format(self._state_inputs, self._state_outputs))
        if len(self.get_initial_state()) != len(self._state_inputs):
            raise ValueError(
                "Length of initial state must match number of state inputs, "
                "got: {} vs {}".format(self.get_initial_state(),
                                       self._state_inputs))
        if self._state_inputs and self._seq_lens is None:
            raise ValueError(
                "seq_lens tensor must be given if state inputs are defined")

    @override(PolicyGraph)
    def get_weights(self):
        return self._variables.get_weights()

    @override(PolicyGraph)
    def set_weights(self, weights):
        return self._variables.set_weights(weights)

    @DeveloperAPI
    def extra_compute_action_fetches(self):
        """Extra values to fetch and return from compute_actions().

        By default we only return action probability info (if present).
        """
        if self._action_logp is not None:
            return {
                ACTION_PROB: self._action_prob,
                ACTION_LOGP: self._action_logp,
            }
        else:
            return {}

    @DeveloperAPI
    def optimizer(self):
        """TF optimizer to use for policy optimization."""
        if hasattr(self, "config"):
            return tf.train.AdamOptimizer(self.config["lr"])
        else:
            return tf.train.AdamOptimizer()

    @property
    def obs_input(self):
        return self._obs_input

    def get_session(self):
        return self._sess

    def variables(self):
        return self.model.variables()

    def _initialize_loss(self, loss, loss_inputs):
        self._loss_inputs = loss_inputs
        self._loss_input_dict = dict(self._loss_inputs)
        for i, ph in enumerate(self._state_inputs):
            self._loss_input_dict["state_in_{}".format(i)] = ph

        if self.model:
            self._loss = self.model.custom_loss(loss, self._loss_input_dict)
            self._stats_fetches.update({
                "model": self.model.metrics() if isinstance(
                    self.model, ModelV2) else self.model.custom_stats()
            })
        else:
            self._loss = loss

        self._optimizer = self.optimizer()
        self._grads_and_vars = [
            (g, v) for (g, v) in self.gradients(self._optimizer, self._loss)
            if g is not None
        ]
        self._grads = [g for (g, v) in self._grads_and_vars]
        if hasattr(self, "model") and isinstance(self.model, ModelV2):
            self._variables = ray.experimental.tf_utils.TensorFlowVariables(
                [], self._sess, self.variables())
        else:
            # TODO(ekl) deprecate support for v1 models
            self._variables = ray.experimental.tf_utils.TensorFlowVariables(
                self._loss, self._sess)

        # gather update ops for any batch norm layers
        if not self._update_ops:
            self._update_ops = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS, scope=tf.get_variable_scope().name)
        if self._update_ops:
            logger.info("Update ops to run on apply gradient: {}".format(
                self._update_ops))
        with tf.control_dependencies(self._update_ops):
            self._apply_op = self.build_apply_op(self._optimizer,
                                                 self._grads_and_vars)

        if log_once("loss_used"):
            logger.debug(
                "These tensors were used in the loss_fn:\n\n{}\n".format(
                    summarize(self._loss_input_dict)))

        self._sess.run(tf.global_variables_initializer())
