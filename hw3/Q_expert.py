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
import pickle as p
import os
import h5py

FLAGS = tf.app.flags.FLAGS

def collect(env,
          q_func,
          session,
          exploration,
          stopping_criterion=None,
          replay_buffer_size=1000000,
          frame_history_len=4):

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
    # casting to float on GPU ensures lower data transfer times.
    obs_t_float = tf.cast(obs_t_ph, tf.float32) / 255.0

    q = q_func(obs_t_float, num_actions, scope="q_func", reuse=False)



    ###############
    # RUN ENV     #
    ###############

    mean_episode_reward = -float('nan')
    best_mean_episode_reward = -float('inf')
    best_reward = -float('inf')
    last_obs = env.reset()
    LOG_EVERY_N_STEPS = 10000

    log_file = FLAGS.method_name+'.log'
    saver = tf.train.Saver()

    if FLAGS.m_bad > 0:
        replay_buffer_obs = ReplayBuffer(replay_buffer_size, frame_history_len, num_actions)
        replay_buffer = ReplayBuffer(int(replay_buffer_size*(np.float(FLAGS.m_bad)/(FLAGS.m_bad+FLAGS.m_good))),
                                     frame_history_len, num_actions)
    else:
        replay_buffer = ReplayBuffer(replay_buffer_size , frame_history_len, num_actions)
    ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_path)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_path = ckpt.model_checkpoint_path
    else:
        print('No checkpoint file found')
        session.close()
        return
    saver.restore(session, ckpt_path)
    print('loaded model of Q from', ckpt_path)

    for t in itertools.count():
        #break
        ### 1. Check stopping criterion
        if stopping_criterion is not None and stopping_criterion(env, t):
            break
        if t%10000 == 0:
            print('*'*30)
            print('Iteration:', t)
            print('*'*30)
        if FLAGS.m_bad > 0:
            eps, should_save = eps_scheduler(t, FLAGS.good_step, FLAGS.m_bad, FLAGS.m_good)
            idx_obs = replay_buffer_obs.store_frame(last_obs)
            if should_save:
                idx = replay_buffer.store_frame(last_obs)
        else:
            idx = replay_buffer.store_frame(last_obs)
            eps = exploration.value(t)


        is_greedy = np.random.rand(1) >= eps
        action_dist_this = np.ones((num_actions), dtype=np.float32) / num_actions
        if is_greedy:
            if FLAGS.m_bad > 0:
                recent_obs = replay_buffer_obs.encode_recent_observation()[np.newaxis, ...]
            else:
                recent_obs = replay_buffer.encode_recent_observation()[np.newaxis, ...]
        q_values = session.run(q, feed_dict={obs_t_ph: recent_obs})
        # TODO: find an appropriate soft_Q_alpha for the sampling
        #q_values = np.exp((q_values - np.max(q_values)) / FLAGS.soft_Q_alpha)
        #dist = q_values / np.sum(q_values)
        #action = np.random.choice(num_actions, p=np.squeeze(dist))
        max_action = np.argmax(np.squeeze(q_values))
        greedy_dist_this = np.zeros((num_actions), dtype=np.float32)
        greedy_dist_this[max_action] = 1.0

        action_dist_this = eps*action_dist_this + (1-eps)*greedy_dist_this
        action = np.random.choice(num_actions, p=np.squeeze(action_dist_this))

        obs, reward, done, info = env.step(action)


        if done:
            obs = env.reset()
        # TODO: modify the following 3 lines to add action_dist back in
        if FLAGS.m_bad > 0:
            replay_buffer_obs.store_effect(idx_obs, action, reward, done,
                                           action_dist_this, info)
            if should_save:
                replay_buffer.store_effect(idx, action, reward, done,
                                           action_dist_this, info)
        else:

            replay_buffer.store_effect(idx, action, reward, done,
                                       action_dist_this, info)
        last_obs = obs

        ### 4. Log progress
        episode_length = get_wrapper_by_name(env, "Monitor").get_episode_lengths()
        episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
        if len(episode_rewards) > 0:
            mean_episode_reward = np.mean(episode_rewards[-100:])
        if len(episode_rewards) > 100:
            best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)

        if t % LOG_EVERY_N_STEPS == 0:
            print('printing log!!!___________________________________________')
            print('Q_Values', q_values)
            print('time_step %d' % t)
            print("mean reward (100 episodes) %f" % mean_episode_reward)
            print("best mean reward %f" % best_mean_episode_reward)
            print("episodes %d" % len(episode_rewards))
            if episode_length != []:
                print("episode length ", episode_length)
            sys.stdout.flush()

            with open(log_file, 'a') as f:
                print(t, mean_episode_reward, best_mean_episode_reward, file=f)


    # save the replay buffer
    print('save pickle!')
    FLAGS.Q_expert_path = './link_data/' + str(FLAGS.torcs_divider) +'torcs_new.p'
    with open(FLAGS.Q_expert_path, 'w') as f:
        p.dump(replay_buffer, f, protocol=-1)



