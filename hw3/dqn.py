from __future__ import print_function
import sys
import gym.spaces
import itertools
import numpy as np
import random
import tensorflow                as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.losses as losses
from collections import namedtuple
from dqn_utils import *
import os
import math

FLAGS = tf.app.flags.FLAGS
OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])


def learn(env,
          q_func,
          optimizer_spec,
          session,
          exploration=LinearSchedule(1000000, 0.1),
          stopping_criterion=None,
          replay_buffer_size=1000000,
          batch_size=32,
          gamma=0.99,
          learning_starts=50000,
          learning_freq=4,
          frame_history_len=4,
          target_update_freq=10000,
          grad_norm_clipping=10):
    """Run Deep Q-learning algorithm.

    You can specify your own convnet using q_func.

    All schedules are w.r.t. total number of steps taken in the environment.

    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    q_func: function
        Model to use for computing the q function. It should accept the
        following named arguments:
            img_in: tf.Tensor
                tensorflow tensor representing the input image
            num_actions: int
                number of actions
            scope: str
                scope in which all the model related variables
                should be created
            reuse: bool
                whether previously created variables should be reused.
    optimizer_spec: OptimizerSpec
        Specifying the constructor and kwargs, as well as learning rate schedule
        for the optimizer
    session: tf.Session
        tensorflow session to use.
    exploration: rl_algs.deepq.utils.schedules.Schedule
        schedule for probability of chosing random action.
    stopping_criterion: (env, t) -> bool
        should return true when it's ok for the RL algorithm to stop.
        takes in env and the number of steps executed so far.
    replay_buffer_size: int
        How many memories to store in the replay buffer.
    batch_size: int
        How many transitions to sample each time experience is replayed.
    gamma: float
        Discount Factor
    learning_starts: int
        After how many environment steps to start replaying experiences
    learning_freq: int
        How many steps of environment to take between every experience replay
    frame_history_len: int
        How many past frames to include as input to the model.
    target_update_freq: int
        How many experience replay rounds (not steps!) to perform between
        each update to the target Q network
    grad_norm_clipping: float or None
        If not None gradients' norms are clipped to this value.
    """
    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space) == gym.spaces.Discrete

    ###############
    # BUILD MODEL #
    ###############

    if len(env.observation_space.shape) == 1:
        # This means we are running on low-dimensional observations (e.g. RAM)
        input_shape = env.observation_space.shape
    else:
        img_h, img_w, img_c = env.observation_space.shape
        input_shape = (img_h, img_w, frame_history_len * img_c)
    num_actions = env.action_space.n

    # set up placeholders
    # placeholder for current observation (or state)
    obs_t_ph = tf.placeholder(tf.uint8, [None] + list(input_shape))
    # placeholder for current action
    act_t_ph = tf.placeholder(tf.int32, [None])
    # placeholder for current reward
    rew_t_ph = tf.placeholder(tf.float32, [None])
    # placeholder for next observation (or state)
    obs_tp1_ph = tf.placeholder(tf.uint8, [None] + list(input_shape))
    # placeholder for end of episode mask
    # this value is 1 if the next state corresponds to the end of an episode,
    # in which case there is no Q-value at the next state; at the end of an
    # episode, only the current state reward contributes to the target, not the
    # next state Q-value (i.e. target is just rew_t_ph, not rew_t_ph + gamma * q_tp1)
    done_mask_ph = tf.placeholder(tf.float32, [None])

    need_hinge_ph = tf.placeholder(tf.float32, shape=())

    # visualize inputs
    to_vis_format = lambda batch: tf.expand_dims(tf.transpose(batch[0, :, :, :], perm=[2, 0, 1]), 3)
    tf.image_summary("observation_now", to_vis_format(obs_t_ph), max_images=4)
    tf.image_summary("observation_next", to_vis_format(obs_tp1_ph), max_images=4)
    tf.scalar_summary("env_input/action", act_t_ph[0])
    tf.scalar_summary("env_input/reward", rew_t_ph[0])

    # casting to float on GPU ensures lower data transfer times.
    obs_t_float = tf.cast(obs_t_ph, tf.float32) / 255.0
    obs_tp1_float = tf.cast(obs_tp1_ph, tf.float32) / 255.0

    # Here, you should fill in your own code to compute the Bellman error. This requires
    # evaluating the current and next Q-values and constructing the corresponding error.
    # TensorFlow will differentiate this error for you, you just need to pass it to the
    # optimizer. See assignment text for details.
    # Your code should produce one scalar-valued tensor: total_error
    # This will be passed to the optimizer in the provided code below.
    # Your code should also produce two collections of variables:
    # q_func_vars
    # target_q_func_vars
    # These should hold all of the variables of the Q-function network and target network,
    # respectively. A convenient way to get these is to make use of TF's "scope" feature.
    # For example, you can create your Q-function network with the scope "q_func" like this:
    # <something> = q_func(obs_t_float, num_actions, scope="q_func", reuse=False)
    # And then you can obtain the variables like this:
    # q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func')
    # Older versions of TensorFlow may require using "VARIABLES" instead of "GLOBAL_VARIABLES"
    ######

    # YOUR CODE HERE
    # The (rapidly updated Q network)
    q = q_func(obs_t_float, num_actions, scope="q_func", reuse=False)
    q_func_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='q_func')

    # The target Q network
    target_q = q_func(obs_tp1_float, num_actions, scope="target_q_func", reuse=False)
    target_q_func_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='target_q_func')

    # visualize two set of vars
    for setname, set in [("rapid_net", q_func_vars), ("target_net", target_q_func_vars)]:
        for v in set:
            tf.histogram_summary(setname + "/" + v.op.name + "_weights", v)
    # visualize all activations
    end_points = tf.get_collection("activation_collection")
    activation_summaries(end_points)

    # The DQN update: use Bellman operator over the *target* network outputs
    # one-step look-ahead using target Q network
    # do the update in a batch
    # Get out the Q values of the performed actions, using the rapid Q network
    def select_each_row(q, act_t_ph, num_actions):
        with tf.variable_scope("select_each_row"):
            return tf.reduce_sum(q * tf.one_hot(act_t_ph, num_actions), 1)
    q_act = select_each_row(q, act_t_ph, num_actions)

    if FLAGS.ddqn:
        print("double Q!")
        # The rapid Q values of the next observation
        q_next = q_func(obs_tp1_float, num_actions, scope="q_func", reuse=True)
        q_next_act = tf.argmax(q_next, 1)
        # If we were using the rapid Q network to do the actions, what are the target Q values
        q_next_act_value = tf.reduce_sum(target_q * tf.one_hot(q_next_act, num_actions), 1)
        q_look_ahead = rew_t_ph + (1 - done_mask_ph) * gamma * q_next_act_value
    else:
        q_look_ahead = rew_t_ph + (1 - done_mask_ph) * gamma * tf.reduce_max(target_q, 1)

    total_error = 0
    if FLAGS.supervise_cross_entropy_loss_weight > 0:
        cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(q, tf.one_hot(act_t_ph, num_actions),
                                                                     name='cross_entropy')
        cross_entropy_loss = tf.reduce_mean(cross_entropy_loss)
        tf.scalar_summary("loss/supervise_cross_ent", cross_entropy_loss)
        total_error += FLAGS.supervise_cross_entropy_loss_weight * cross_entropy_loss

    if FLAGS.hard_Q_loss_weight > 0:
        temporal_diff_loss = tf.nn.l2_loss(q_act - q_look_ahead) * 2 / batch_size
        tf.scalar_summary("loss/hard_Q", temporal_diff_loss)
        total_error += FLAGS.hard_Q_loss_weight * temporal_diff_loss

    if FLAGS.l2_regularization_loss_weight > 0:
        regularization_loss = tf.add_n([tf.reduce_sum(tf.square(reg_weight)) for reg_weight in q_func_vars])
        tf.scalar_summary("loss/l2_regularization", regularization_loss)
        total_error += FLAGS.l2_regularization_loss_weight * regularization_loss

    # TODO: relative weight between entropy and reward has to be included
    alpha = FLAGS.soft_Q_alpha
    def Q2V(target_q, alpha):
        with tf.variable_scope("Q2V"):
            q_max = tf.reduce_max(target_q, 1, keep_dims=True)
            V = alpha * tf.log(tf.reduce_sum(tf.exp(1 / alpha * (target_q - q_max)), 1)) \
                + tf.squeeze(q_max)
        return V
    V = Q2V(target_q, alpha)
    q_soft_ahead = rew_t_ph + (1 - done_mask_ph) * gamma * V
    if FLAGS.soft_Q_loss_weight > 0:
        max_ent_loss = tf.nn.l2_loss(q_act - q_soft_ahead) * 2 / batch_size
        tf.scalar_summary("loss/soft_Q", max_ent_loss)
        total_error += FLAGS.soft_Q_loss_weight * max_ent_loss

    def compute_hinge(act_t, q_t, q_act_t, num_actions):
        loss_l = 0.8 - 0.8 * tf.one_hot(act_t, num_actions)
        large_margin = tf.reduce_max(loss_l + q_t, 1)
        return tf.reduce_mean(large_margin - q_act_t)
    if FLAGS.supervise_hinge_DQfD_loss_weight > 0:
        if FLAGS.demo_mode == 'dqfd':
            indices = tf.constant(range(0, int(batch_size*FLAGS.demo_portion)))
            act_t_hinge = tf.gather(act_t_ph, indices)
            obs_t_hinge = tf.gather(obs_t_float, indices)
            q_hinge = q_func(obs_t_hinge, num_actions, scope="q_func", reuse=True)
            q_act_hinge = select_each_row(q_hinge, act_t_hinge, num_actions)
            hinge_loss = compute_hinge(act_t_hinge, q_hinge, q_act_hinge, num_actions)
        else:
            hinge_loss = compute_hinge(act_t_ph, q, q_act, num_actions)
            tf.scalar_summary("loss/supervise_hinge_DQfD", hinge_loss)
        total_error += FLAGS.supervise_hinge_DQfD_loss_weight * hinge_loss * need_hinge_ph

    if FLAGS.supervise_hinge_standard_loss_weight > 0:
        crammer = losses.hinge_loss(q, tf.one_hot(act_t_ph, num_actions))
        crammer = tf.reduce_mean(crammer)
        tf.scalar_summary("loss/supervise_hinge_standard", crammer)
        total_error += FLAGS.supervise_hinge_standard_loss_weight * crammer

    if FLAGS.policy_gradient_soft_1_step > 0:
        Vrapid = Q2V(q, alpha)
        node_grad = q_act - Vrapid
        node_no_grad = tf.stop_gradient(q_act - q_soft_ahead)
        pg1_output = tf.reduce_mean(node_grad * node_no_grad)
        tf.scalar_summary("loss/policy_gradient_soft_1_step", pg1_output)
        # surrogate loss, not used, for vis only
        pg1_surrogate = tf.reduce_mean(tf.square(q_act - Vrapid - q_soft_ahead))

        '''
        # stop gradient implementation of pg1_surrogate
        node_grad = q_act - Vrapid
        node_no_grad = tf.stop_gradient(q_act - Vrapid - q_soft_ahead)
        pg1_surrogate = tf.reduce_mean(node_grad * node_no_grad)
        '''
        tf.scalar_summary("loss/policy_gradient_soft_1_step_surrogate", pg1_surrogate)

        #total_error += FLAGS.policy_gradient_soft_1_step * pg1_output
        total_error += FLAGS.policy_gradient_soft_1_step * pg1_surrogate

    if FLAGS.exp_soft_Q_bellman > 0:
        Vrapid = Q2V(q, alpha)
        exp_q_soft = rew_t_ph + (1 - done_mask_ph) * gamma * Vrapid
        tderror = tf.reduce_mean(tf.square(q_act - exp_q_soft*0.5 - q_soft_ahead*0.5))
        tf.scalar_summary("loss/exp_soft_Q_bellman", tderror)
        total_error += FLAGS.exp_soft_Q_bellman * tderror

    def QV2pi(q, v, alpha):
        with tf.variable_scope("QV2pi"):
            v = tf.expand_dims(v, 1)
            return tf.exp((q-v) / (1.0*alpha))

    if FLAGS.exp_policy_grad_weighting > 0:
        with tf.variable_scope("exp_policy_grad_weighting"):
            Vrapid = Q2V(q, alpha)
            pi_rapid = QV2pi(q, Vrapid, alpha)
            pi_selected = select_each_row(pi_rapid, act_t_ph, num_actions)

            node_grad = q_act - Vrapid
            #node_no_grad = tf.stop_gradient(q_act - q_soft_ahead, name="q_yStar")
            node_no_grad = tf.stop_gradient(q_act - Vrapid - q_soft_ahead, name="q_yStar")

            weighting = tf.stop_gradient(pi_selected, name="weighting")
            weighted_grad = tf.reduce_mean(node_grad * node_no_grad * weighting, name="grad_final")
            tf.scalar_summary("loss/policy_gradient_soft_1_step", weighted_grad)
            total_error += FLAGS.exp_policy_grad_weighting * weighted_grad

            tf.histogram_summary("weighting_of_grad", weighting)

    tf.scalar_summary("loss/total", total_error)

    # construct optimization op (with gradient clipping)
    learning_rate = tf.placeholder(tf.float32, (), name="learning_rate")
    tf.scalar_summary("learning_rate", learning_rate)
    optimizer = optimizer_spec.constructor(learning_rate=learning_rate, **optimizer_spec.kwargs)
    train_fn = minimize_and_clip(optimizer, total_error,
                                 var_list=q_func_vars, clip_val=grad_norm_clipping)

    # update_target_fn will be called periodically to copy Q network to target Q network
    update_target_fn = []
    for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                               sorted(target_q_func_vars, key=lambda v: v.name)):
        update_target_fn.append(var_target.assign(var))
    update_target_fn = tf.group(*update_target_fn)

    summary_op = tf.merge_all_summaries()
    model_save_path = os.path.join('./link_data/', FLAGS.method_name)
    summary_writer = tf.train.SummaryWriter(
        model_save_path,
        graph_def=session.graph.as_graph_def(add_shapes=True))

    # construct the replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

    ###############
    # RUN ENV     #
    ###############
    model_initialized = False
    num_param_updates = 0
    mean_episode_reward = -float('nan')
    best_mean_episode_reward = -float('inf')
    best_reward = -float('inf')
    last_obs = env.reset()
    LOG_EVERY_N_STEPS = 10000

    log_file = os.path.join(model_save_path, "log.txt")
    saver = tf.train.Saver()

    if FLAGS.demo_mode == 'hdf':
        replay_buffer = get_hdf_demo([FLAGS.demo_file_path,FLAGS.demo_file_path_1], replay_buffer)
    elif FLAGS.demo_mode == 'replay':
        replay_buffer = load_replay_pickle(FLAGS.demo_file_path, FLAGS.dataset_size)
    elif FLAGS.demo_mode == 'dqfd':
        replay_buffer_demo = load_replay_pickle(FLAGS.demo_file_path, FLAGS.dataset_size)
    elif FLAGS.demo_mode == 'no_demo':
        pass
    else:
        raise ValueError("invalid FLAGS.demo_mode = %s" % FLAGS.demo_mode)
    #print(replay_buffer.obs.shape, replay_buffer.reward.shape, replay_buffer.action.shape, replay_buffer.done.shape)
    if FLAGS.inenv_finetune:
        print('*'*30)
        print(FLAGS.ckpt_path)
        ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_path)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_path = ckpt.model_checkpoint_path
        else:
            print('No checkpoint file found')
            session.close()
            return
        saver.restore(session, ckpt_path)
        print('model loaded!!!! %s' % ckpt_path)
        print('*'*30)
        model_initialized = True






    for t in itertools.count():
        ### 1. Check stopping criterion
        if stopping_criterion is not None and stopping_criterion(env, t):
            break

        ### 2. Step the env and store the transition
        # At this point, "last_obs" contains the latest observation that was
        # recorded from the simulator. Here, your code needs to store this
        # observation and its outcome (reward, next observation, etc.) into
        # the replay buffer while stepping the simulator forward one step.
        # At the end of this block of code, the simulator should have been
        # advanced one step, and the replay buffer should contain one more
        # transition.
        # Specifically, last_obs must point to the new latest observation.
        # Useful functions you'll need to call:
        # obs, reward, done, info = env.step(action)
        # this steps the environment forward one step
        # obs = env.reset()
        # this resets the environment if you reached an episode boundary.
        # Don't forget to call env.reset() to get a new observation if done
        # is true!!
        # Note that you cannot use "last_obs" directly as input
        # into your network, since it needs to be processed to include context
        # from previous frames. You should check out the replay buffer
        # implementation in dqn_utils.py to see what functionality the replay
        # buffer exposes. The replay buffer has a function called
        # encode_recent_observation that will take the latest observation
        # that you pushed into the buffer and compute the corresponding
        # input that should be given to a Q network by appending some
        # previous frames.
        # Don't forget to include epsilon greedy exploration!
        # And remember that the first time you enter this loop, the model
        # may not yet have been initialized (but of course, the first step
        # might as well be random, since you haven't trained your net...)

        #####

        # YOUR CODE HERE
        # TODO: right now the demonstration and dqn share the same buffer
        if FLAGS.collect_Q_experience:
            assert (np.sign(FLAGS.soft_Q_loss_weight) != np.sign(FLAGS.hard_Q_loss_weight))
            if FLAGS.hard_Q_loss_weight > 0:
                idx = replay_buffer.store_frame(last_obs)
                eps = exploration.value(t)
                #eps = FLAGS.tiny_explore
                is_greedy = np.random.rand(1) >= eps
                if is_greedy and model_initialized:
                    recent_obs = replay_buffer.encode_recent_observation()[np.newaxis, ...]
                    q_values = session.run(q, feed_dict={obs_t_ph: recent_obs})
                    action = np.argmax(np.squeeze(q_values))
                else:
                    action = np.random.choice(num_actions)

                obs, reward, done, info = env.step(action)
                if done:
                    obs = env.reset()
                replay_buffer.store_effect(idx, action, reward, done)
                last_obs = obs
            else:
                idx = replay_buffer.store_frame(last_obs)
                eps = FLAGS.tiny_explore
                #eps = exploration.value(t)
                is_greedy = np.random.rand(1) >= eps
                if is_greedy and model_initialized:
                    recent_obs = replay_buffer.encode_recent_observation()[np.newaxis, ...]
                    q_values = session.run(q, feed_dict={obs_t_ph: recent_obs})
                    q_values = np.exp((q_values - np.max(q_values)) / FLAGS.soft_Q_alpha)
                    dist = q_values / np.sum(q_values)
                    action = np.random.choice(num_actions, p=np.squeeze(dist))
                    #action = np.random.choice(num_actions, p=np.squeeze(q_norm))
                else:
                    action = np.random.choice(num_actions)
                obs, reward, done, info = env.step(action)
                if done:
                    obs = env.reset()
                replay_buffer.store_effect(idx, action, reward, done)
                last_obs = obs

        #####

        # at this point, the environment should have been advanced one step (and
        # reset if done was true), and last_obs should point to the new latest
        # observation

        ### 3. Perform experience replay and train the network.
        # note that this is only done if the replay buffer contains enough samples
        # for us to learn something useful -- until then, the model will not be
        # initialized and random actions should be taken
        summary_value = None
        if (t > learning_starts and
                        t % learning_freq == 0 and
                replay_buffer.can_sample(batch_size)):
            # Here, you should perform training. Training consists of four steps:
            # 3.a: use the replay buffer to sample a batch of transitions (see the
            # replay buffer code for function definition, each batch that you sample
            # should consist of current observations, current actions, rewards,
            # next observations, and done indicator).
            # 3.b: initialize the model if it has not been initialized yet; to do
            # that, call
            #    initialize_interdependent_variables(session, tf.global_variables(), {
            #        obs_t_ph: obs_t_batch,
            #        obs_tp1_ph: obs_tp1_batch,
            #    })
            # where obs_t_batch and obs_tp1_batch are the batches of observations at
            # the current and next time step. The boolean variable model_initialized
            # indicates whether or not the model has been initialized.
            # Remember that you have to update the target network too (see 3.d)!
            # 3.c: train the model. To do this, you'll need to use the train_fn and
            # total_error ops that were created earlier: total_error is what you
            # created to compute the total Bellman error in a batch, and train_fn
            # will actually perform a gradient step and update the network parameters
            # to reduce total_error. When calling session.run on these you'll need to
            # populate the following placeholders:
            # obs_t_ph
            # act_t_ph
            # rew_t_ph
            # obs_tp1_ph
            # done_mask_ph
            # (this is needed for computing total_error)
            # learning_rate -- you can get this from optimizer_spec.lr_schedule.value(t)
            # (this is needed by the optimizer to choose the learning rate)
            # 3.d: periodically update the target network by calling
            # session.run(update_target_fn)
            # you should update every target_update_freq steps, and you may find the
            # variable num_param_updates useful for this (it was initialized to 0)
            #####

            if t % 100000 == 0 and FLAGS.save_model:
                model_save_path = os.path.join('./link_data/', FLAGS.method_name)
                if not os.path.exists(model_save_path):
                    tf.gfile.MakeDirs(model_save_path)
                save_path = saver.save(session,
                                       os.path.join(model_save_path, "model_%s.ckpt" % str(t)))
                print('saved at: ', save_path)


            if FLAGS.demo_mode == 'dqfd':
                demo_size = int(batch_size * FLAGS.demo_portion)
                package_demo, need_hinge_demo = \
                    replay_buffer_demo.sample(demo_size, FLAGS.group_name)
                obs_t_batch_demo, act_t_batch_demo, rew_t_batch_demo, obs_tp1_batch_demo, done_mask_demo = \
                    package_demo
                package, need_hinge = \
                    replay_buffer.sample(batch_size-demo_size, FLAGS.group_name)
                obs_t_batch, act_t_batch, rew_t_batch, obs_tp1_batch, done_mask = \
                    package
                obs_t_batch   = np.concatenate((obs_t_batch_demo, obs_t_batch))
                act_t_batch   = np.concatenate((act_t_batch_demo, act_t_batch))
                rew_t_batch   = np.concatenate((rew_t_batch_demo, rew_t_batch))
                obs_tp1_batch = np.concatenate((obs_tp1_batch_demo, obs_tp1_batch))
                done_mask     = np.concatenate((done_mask_demo  , done_mask))
            else:
                package, need_hinge = \
                    replay_buffer.sample(batch_size, FLAGS.group_name)
                obs_t_batch, act_t_batch, rew_t_batch, obs_tp1_batch, done_mask = \
                    package

                # (b)
            if not model_initialized:
                initialize_interdependent_variables(session, tf.all_variables(), {
                    obs_t_ph: obs_t_batch,
                    obs_tp1_ph: obs_tp1_batch,
                })
                model_initialized = True

            # (c)
            feed_dict = {
                obs_t_ph: obs_t_batch,
                act_t_ph: act_t_batch,
                rew_t_ph: rew_t_batch,
                obs_tp1_ph: obs_tp1_batch,
                done_mask_ph: done_mask,
                need_hinge_ph: need_hinge,
                learning_rate: optimizer_spec.lr_schedule.value(t)
            }

            if t % FLAGS.summary_interval == 0:
                _, summary_value = session.run([train_fn, summary_op], feed_dict)
            else:
                session.run(train_fn, feed_dict)
            num_param_updates += 1

            # (d)
            if num_param_updates % target_update_freq == 0:
                session.run(update_target_fn)

                #####

        ### 4. Log progress

        # evaluating in the environment, when off policy training is used
        # Warning: this evaluation does not use target network!
        # TODO: make sure the environment Monitor return the undiscounted reward
        reward_calc = None
        if FLAGS.eval_freq > 0 and t % FLAGS.eval_freq == 0 and model_initialized:
            print('_' * 50)
            print('Start Evaluating at TimeStep %d' % t)
            eps = FLAGS.tiny_explore

            reward_calc, frame_counter = \
                eval_policy(env, q, obs_t_ph,
                            session,
                            eps, frame_history_len, num_actions, img_c)

            best_reward = np.max([best_reward, reward_calc])
            print("the frame counter is %f" % frame_counter)
            print("test reward %f" % reward_calc)
            print("best test reward %f" % best_reward)
            with open(log_file, 'a') as f:
                print(t, reward_calc, best_reward, file=f)

        # evaluating with the current environment
        if FLAGS.eval_freq <= 0:
            episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
            if len(episode_rewards) > 0:
                mean_episode_reward = np.mean(episode_rewards[-100:])
                reward_calc = episode_rewards[-1]
            if len(episode_rewards) > 100:
                best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)

            if t % LOG_EVERY_N_STEPS == 0 and model_initialized:
                print("mean reward (100 episodes) %f" % mean_episode_reward)
                print("best mean reward %f" % best_mean_episode_reward)
                print("episodes %d" % len(episode_rewards))
                print("exploration %f" % exploration.value(t))
                print("learning_rate %f" % optimizer_spec.lr_schedule.value(t))
                sys.stdout.flush()

                with open(log_file, 'a') as f:
                    print(t, mean_episode_reward, best_mean_episode_reward, file=f)

        # adding visualizations to tensorboard
        # summary condition: train has a summary, eval has a new value(>0), or eval has extended(<0)
        sum_train = (summary_value is not None)
        sum_eval = (reward_calc is not None) and (
            (FLAGS.eval_freq > 0) or
            (FLAGS.eval_freq <= 0 and t % LOG_EVERY_N_STEPS == 0 and model_initialized))
        if sum_train or sum_eval:
            summary = tf.Summary()
            if sum_train:
                summary.ParseFromString(summary_value)
            if sum_eval:
                summary.value.add(tag='exploration', simple_value=exploration.value(t))
                summary.value.add(tag='reward', simple_value=reward_calc)
            summary_writer.add_summary(summary, t)
