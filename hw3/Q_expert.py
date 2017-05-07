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

FLAGS = tf.app.flags.FLAGS

def collect(env,
          q_func,
          session,
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

    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

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
        ### 1. Check stopping criterion
        if stopping_criterion is not None and stopping_criterion(env, t):
            break

        idx = replay_buffer.store_frame(last_obs)
        eps = 0 #FLAGS.tiny_explore
        is_greedy = np.random.rand(1) >= eps
        if is_greedy:
            recent_obs = replay_buffer.encode_recent_observation()[np.newaxis, ...]
            q_values = session.run(q, feed_dict={obs_t_ph: recent_obs})
            # TODO: find an appropriate soft_Q_alpha for the sampling
            q_values = np.exp((q_values - np.max(q_values)) / FLAGS.soft_Q_alpha)
            dist = q_values / np.sum(q_values)
            action = np.random.choice(num_actions, p=np.squeeze(dist))
            #action = np.argmax(np.squeeze(q_values))
        else:
            action = np.random.choice(num_actions)

        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()

        replay_buffer.store_effect(idx, action, reward, done)
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
            print('time_step %d' % t)
            print("mean reward (100 episodes) %f" % mean_episode_reward)
            print("best mean reward %f" % best_mean_episode_reward)
            print("episodes %d" % len(episode_rewards))
            if episode_length != []:
                print("episode length %d" % episode_length[-1])
            sys.stdout.flush()

            with open(log_file, 'a') as f:
                print(t, mean_episode_reward, best_mean_episode_reward, file=f)

    # save the replay buffer
    print('save pickle!')
    FLAGS.Q_expert_path = './link_data/' + 'softQ_expert.p'
    with open(FLAGS.Q_expert_path, 'w') as f:
        p.dump(replay_buffer, f, protocol=-1)
