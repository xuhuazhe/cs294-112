import tensorflow as tf
from dqn_utils import *

def select_each_row(q, act_t_ph, num_actions, keep_dims=False):
    with tf.variable_scope("select_each_row"):
        return tf.reduce_sum(q * tf.one_hot(act_t_ph, num_actions), 1, keep_dims=keep_dims)

def Q2V(target_q, alpha):
    with tf.variable_scope("Q2V"):
        q_max = tf.reduce_max(target_q, 1, keep_dims=True)
        V = alpha * tf.log(tf.reduce_sum(tf.exp(1 / alpha * (target_q - q_max)), 1)) + tf.squeeze(q_max)
    return V

def QV2pi(q, v, alpha):
    with tf.variable_scope("QV2pi"):
        v = tf.expand_dims(v, 1)
        return tf.exp((q - v) / (1.0 * alpha))

def compute_hinge(act_t, q_t, q_act_t, num_actions):
    loss_l = 0.8 - 0.8 * tf.one_hot(act_t, num_actions)
    large_margin = tf.reduce_max(loss_l + q_t, 1)
    return tf.reduce_mean(large_margin - q_act_t)


def generate_losses(q_func,
                    num_actions,
                    gamma,
                    batch_size,

                    obs_t_float,
                    obs_tp1_float,
                    act_t_ph,
                    rew_t_ph,
                    done_mask_ph,
                    ):

    # The (rapidly updated Q network)
    q_rapid_t = q_func(obs_t_float, num_actions, scope="q_func", reuse=False)
    q_rapid_tp1 = q_func(obs_tp1_float, num_actions, scope="q_func", reuse=True)
    q_func_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='q_func')

    # The target Q network
    q_target_tp1 = q_func(obs_tp1_float, num_actions, scope="target_q_func", reuse=False)
    q_target_t = q_func(obs_t_float, num_actions, scope="target_q_func", reuse=True)
    target_q_func_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='target_q_func')

    # visualize two set of vars
    for setname, set in [("rapid_net", q_func_vars), ("target_net", target_q_func_vars)]:
        for v in set:
            tf.contrib.deprecated.histogram_summary(setname + "/" + v.op.name + "_weights", v)
    # visualize all activations
    end_points = tf.get_collection("activation_collection")
    activation_summaries(end_points)

    # relative weight between entropy and reward has already been included, which is the temperature tau
    alpha = FLAGS.soft_Q_alpha

    # compute a bunch of helper variables
    q_rapid_act = select_each_row(q_rapid_t, act_t_ph, num_actions)

    V_rapid_t = Q2V(q_rapid_t, alpha)
    V_rapid_tp1 = Q2V(q_rapid_tp1, alpha)
    V_target_t = Q2V(q_target_t, alpha)
    V_target_tp1 = Q2V(q_target_tp1, alpha)
    V_target_tp1_hardmax = tf.reduce_max(q_target_tp1, 1)

    q_soft_ahead = rew_t_ph + (1 - done_mask_ph) * gamma * V_target_tp1
    q_hard_ahead = rew_t_ph + (1 - done_mask_ph) * gamma * V_target_tp1_hardmax


    # below is all possible errors
    total_error = 0
    if FLAGS.supervise_cross_entropy_loss_weight > 0:
        cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=q_rapid_t, labels=tf.one_hot(act_t_ph, num_actions),
                                                                     name='cross_entropy')
        cross_entropy_loss = tf.reduce_mean(cross_entropy_loss)
        tf.contrib.deprecated.scalar_summary("loss/supervise_cross_ent", cross_entropy_loss)
        total_error += FLAGS.supervise_cross_entropy_loss_weight * cross_entropy_loss

    if FLAGS.hard_Q_loss_weight > 0:
        temporal_diff_loss = tf.nn.l2_loss(q_rapid_act - q_hard_ahead) * 2 / batch_size
        tf.contrib.deprecated.scalar_summary("loss/hard_Q", temporal_diff_loss)
        total_error += FLAGS.hard_Q_loss_weight * temporal_diff_loss

    if FLAGS.l2_regularization_loss_weight > 0:
        regularization_loss = tf.add_n([tf.reduce_sum(tf.square(reg_weight)) for reg_weight in q_func_vars])
        tf.contrib.deprecated.scalar_summary("loss/l2_regularization", regularization_loss)
        total_error += FLAGS.l2_regularization_loss_weight * regularization_loss

    if FLAGS.soft_Q_loss_weight > 0:
        max_ent_loss = tf.nn.l2_loss(q_rapid_act - q_soft_ahead) * 2 / batch_size
        tf.contrib.deprecated.scalar_summary("loss/soft_Q", max_ent_loss)
        total_error += FLAGS.soft_Q_loss_weight * max_ent_loss

    if FLAGS.supervise_hinge_DQfD_loss_weight > 0:
        if FLAGS.demo_mode == 'dqfd':
            indices = tf.constant(range(0, int(batch_size * FLAGS.demo_portion)))
            act_t_hinge = tf.gather(act_t_ph, indices)
            obs_t_hinge = tf.gather(obs_t_float, indices)
            q_hinge = q_func(obs_t_hinge, num_actions, scope="q_func", reuse=True)
            q_act_hinge = select_each_row(q_hinge, act_t_hinge, num_actions)
            hinge_loss = compute_hinge(act_t_hinge, q_hinge, q_act_hinge, num_actions)
        else:
            hinge_loss = compute_hinge(act_t_ph, q_rapid_t, q_rapid_act, num_actions)
            tf.contrib.deprecated.scalar_summary("loss/supervise_hinge_DQfD", hinge_loss)
        total_error += FLAGS.supervise_hinge_DQfD_loss_weight * hinge_loss

    # grad = grad(Q-V)*(Q-Q_hat)
    if FLAGS.exp_policy_grad_weighting > 0:
        with tf.variable_scope("exp_policy_grad_weighting"):
            node_grad = q_rapid_act - V_rapid_t
            node_no_grad = tf.stop_gradient(q_rapid_act - q_soft_ahead, name="q_yStar")
            # node_no_grad = tf.stop_gradient(q_rapid_act - V_rapid_t - q_soft_ahead, name="q_yStar")

            weighted_grad = tf.reduce_mean(node_grad * node_no_grad, name="grad_final")
            total_error += FLAGS.exp_policy_grad_weighting * weighted_grad

            tf.contrib.deprecated.histogram_summary("sign_visualize/policy_gradient_weighting", node_no_grad)
            tf.contrib.deprecated.scalar_summary("loss/policy_gradient_soft_1_step", weighted_grad)

    if FLAGS.exp_value_critic_weighting > 0:
        # fit a value critic using the Q values
        # V_rapid_t == rapid net, on current frame
        # V_target_tp1 == target net, on next frame
        # first compute the target value
        print("using value fitting baseline")
        pi_target_t = QV2pi(q_target_t, V_target_t, alpha)
        KL = tf.reduce_sum(pi_target_t * q_target_t, 1) - V_target_t

        y = rew_t_ph - KL + (1 - done_mask_ph) * gamma * V_target_tp1
        y = tf.stop_gradient(y)
        loss = tf.reduce_mean(tf.square(y - V_rapid_t))
        tf.contrib.deprecated.scalar_summary("loss/exp_value_critic_mean_square_error", loss)
        total_error += FLAGS.exp_value_critic_weighting * loss

    if FLAGS.PCL_1_step_weighting > 0:
        # PCL doesn't require any target network
        # It also doesn't require any on policy sample
        #   the original paper use the Boltzman distribution for exploration
        #   we keep the same epsilon-exploration strategy here

        # the standard error term
        # V_now = Q2V(rapid_now, alpha)
        # pi_now = QV2pi(rapid_now, V_now, alpha)
        # pi_now = select_each_row(pi_now, act_t_ph, num_actions)
        # pcl_error = -V_now + rew_t_ph + gamma*(1-done_mask_ph)*V_next - alpha*tf.log(pi_now)
        # the unified PCL, simplified term
        pcl_error = q_rapid_act - (rew_t_ph + (1 - done_mask_ph) * gamma * V_rapid_tp1)
        pcl_error = tf.reduce_mean(tf.square(pcl_error))
        tf.contrib.deprecated.scalar_summary("loss/PCL_1_step_error", pcl_error)
        total_error += FLAGS.PCL_1_step_weighting * pcl_error

        # up to now, PCL with 1 step roll out, and unified parameterization, is equal to:
        # soft Q learning without target network
        # this happen to be exactly the same as "exp_soft_Q_bellman", and it doesn't work

    return total_error, q_func_vars, target_q_func_vars, q_rapid_t
