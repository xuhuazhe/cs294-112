from __future__ import print_function
import sys
import gym.spaces
import itertools
import numpy as np
import random
import tensorflow                as tf
import tensorflow.contrib.layers as layers
from collections import namedtuple
from dqn_utils import *
import os

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
    assert type(env.action_space)      == gym.spaces.Discrete

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
    obs_t_ph              = tf.placeholder(tf.uint8, [None] + list(input_shape))
    # placeholder for current action
    act_t_ph              = tf.placeholder(tf.int32,   [None])
    # placeholder for current reward
    rew_t_ph              = tf.placeholder(tf.float32, [None])
    # placeholder for next observation (or state)
    obs_tp1_ph            = tf.placeholder(tf.uint8, [None] + list(input_shape))
    # placeholder for end of episode mask
    # this value is 1 if the next state corresponds to the end of an episode,
    # in which case there is no Q-value at the next state; at the end of an
    # episode, only the current state reward contributes to the target, not the
    # next state Q-value (i.e. target is just rew_t_ph, not rew_t_ph + gamma * q_tp1)
    done_mask_ph          = tf.placeholder(tf.float32, [None])

    # casting to float on GPU ensures lower data transfer times.
    obs_t_float   = tf.cast(obs_t_ph,   tf.float32) / 255.0
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
    q_next = q_func(obs_tp1_float, num_actions, scope="q_func", reuse=True)
    # The target Q network
    target_q = q_func(obs_tp1_float, num_actions, scope="target_q_func", reuse=False)
    target_q_func_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='target_q_func')

    # The DQN update: use Bellman operator over the *target* network outputs
    # one-step look-ahead using target Q network
    # do the update in a batch
    q_act = tf.reduce_sum(q*tf.one_hot(act_t_ph, num_actions), 1)

    if FLAGS.ddqn:
        print("double Q!")
        q_next_act = tf.argmax(q_next, 1)
        q_look_ahead = rew_t_ph + (1 - done_mask_ph) * gamma * tf.reduce_sum(target_q*tf.one_hot(q_next_act, num_actions), 1)
    else:
        q_look_ahead = rew_t_ph + (1 - done_mask_ph) * gamma * tf.reduce_max(target_q, 1)
    total_error = tf.nn.l2_loss(q_act-q_look_ahead)*2 / batch_size

    ######

    # construct optimization op (with gradient clipping)
    learning_rate = tf.placeholder(tf.float32, (), name="learning_rate")
    optimizer = optimizer_spec.constructor(learning_rate=learning_rate, **optimizer_spec.kwargs)
    train_fn = minimize_and_clip(optimizer, total_error,
                 var_list=q_func_vars, clip_val=grad_norm_clipping)

    # update_target_fn will be called periodically to copy Q network to target Q network
    update_target_fn = []
    for var, var_target in zip(sorted(q_func_vars,        key=lambda v: v.name),
                               sorted(target_q_func_vars, key=lambda v: v.name)):
        update_target_fn.append(var_target.assign(var))
    update_target_fn = tf.group(*update_target_fn)

    # construct the replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

    ###############
    # RUN ENV     #
    ###############
    model_initialized = False
    num_param_updates = 0
    mean_episode_reward      = -float('nan')
    best_mean_episode_reward = -float('inf')
    best_reward = -float('inf')
    last_obs = env.reset()
    LOG_EVERY_N_STEPS = 10000

    log_file = sys.argv[1]
    saver = tf.train.Saver()

    if FLAGS.demo_mode == 'hdf':
        replay_buffer = Get_HDF_Demo(FLAGS.demo_hdf_dir, replay_buffer, FLAGS.pickle_dir)
    elif FLAGS.demo_mode == 'replay':
        Load_Replay_Pickle(FLAGS.pickle_dir)
    else:
        pass

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
        if FLAGS.collect_Q_experience:
            idx = replay_buffer.store_frame(last_obs)
            eps = exploration.value(t)
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

        #####

        # at this point, the environment should have been advanced one step (and
        # reset if done was true), and last_obs should point to the new latest
        # observation

        ### 3. Perform experience replay and train the network.
        # note that this is only done if the replay buffer contains enough samples
        # for us to learn something useful -- until then, the model will not be
        # initialized and random actions should be taken
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
            
            # YOUR CODE HERE
            # (a)
            if t % 100000 == 0 and FLAGS.save_model:
                if not os.path.exists(os.path.join('./link_data/', FLAGS.method_name)):
                    os.mkdir(os.path.join('./link_data/', FLAGS.method_name))
                save_path=saver.save(session, os.path.join('./link_data/', FLAGS.method_name,"model_%s.ckpt" %(str(t))))
                print('saved at ',save_path)
            obs_t_batch, act_t_batch, rew_t_batch, obs_tp1_batch, done_mask = \
                replay_buffer.sample(batch_size)
            # (b)
            if not model_initialized:
                initialize_interdependent_variables(session, tf.all_variables(), {
                    obs_t_ph: obs_t_batch,
                    obs_tp1_ph: obs_tp1_batch,
                })
                model_initialized = True

            # (c)
            session.run(train_fn, {
                obs_t_ph: obs_t_batch,
                act_t_ph: act_t_batch,
                rew_t_ph: rew_t_batch,
                obs_tp1_ph: obs_tp1_batch,
                done_mask_ph: done_mask,
                learning_rate: optimizer_spec.lr_schedule.value(t)
            })
            num_param_updates += 1

            # (d)
            if num_param_updates % target_update_freq == 0:
                session.run(update_target_fn)

            #####

        ### 4. Log progress
        #print('Now the modular is %d' % int(t%FLAGS.eval_freq) )
        #print(FLAGS.eval_mode)

        if FLAGS.eval_freq > 0 and t%FLAGS.eval_freq == 0 and model_initialized:
            print('_'*50)
            print('Start Evaluating at TimeStep %d' % t)
            eps = FLAGS.tiny_explore
            reward_calc = 0

            obs = env.reset()
            input_obs = obs
            frame_counter = 0
            while True:
                frame_counter += 1
                is_greedy = np.random.rand(1) >= eps
                if is_greedy and frame_counter >= frame_history_len:
                    feed_input_obs = np.reshape(input_obs,[1]+list(input_obs.shape))
                    q_values = session.run(q, feed_dict={obs_t_ph: feed_input_obs})
                    action = np.argmax(np.squeeze(q_values))
                else:
                    action = np.random.choice(num_actions)

                obs, reward, done, info = env.step(action)
                input_obs = np.concatenate((input_obs, obs), 2)
                assert(len(env.observation_space.shape) == 3)
                if input_obs.shape[2] > frame_history_len*img_c:
                    input_obs = input_obs[:,:,-frame_history_len*img_c:]
                reward_calc += reward
                if done:
                    break
            best_reward = np.max([best_reward, reward_calc])
            with open(log_file, 'a') as f:
                print(t, reward_calc, best_reward, file=f)
            print("the frame counter is %f" % frame_counter)
            print("test reward %f" % reward_calc)
            print("best test reward %f" % best_reward)

        if FLAGS.use_env_reward:
            episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
            if len(episode_rewards) > 0:
                mean_episode_reward = np.mean(episode_rewards[-100:])
            if len(episode_rewards) > 100:
                best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)

        if t % LOG_EVERY_N_STEPS == 0 and model_initialized:
            if FLAGS.use_env_reward:
                print("mean reward (100 episodes) %f" % mean_episode_reward)
                print("best mean reward %f" % best_mean_episode_reward)
                print("episodes %d" % len(episode_rewards))
            print("exploration %f" % exploration.value(t))
            print("learning_rate %f" % optimizer_spec.lr_schedule.value(t))
            sys.stdout.flush()

            if FLAGS.use_env_reward:
                with open(log_file, 'a') as f:
                    print(t, mean_episode_reward, best_mean_episode_reward, file=f)
