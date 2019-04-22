from __future__ import print_function
import sys
import gym.spaces
import itertools
import numpy as np
import tensorflow as tf
from collections import namedtuple
from dqn_utils import *
import os
import losses

FLAGS = tf.app.flags.FLAGS
OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])

def learn(env_train,
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
          grad_norm_clipping=10,
          env_test=None,
          model_store_root_path='./link_data/'):
    env_info = env_train if env_train else env_test

    assert type(env_info.observation_space) == gym.spaces.Box
    assert type(env_info.action_space) == gym.spaces.Discrete

    ###############
    # BUILD MODEL #
    ###############

    if len(env_info.observation_space.shape) == 1:
        # This means we are running on low-dimensional observations (e.g. RAM)
        input_shape = env_info.observation_space.shape
    else:
        img_h, img_w, img_c = env_info.observation_space.shape
        input_shape = (img_h, img_w, frame_history_len * img_c)
    num_actions = env_info.action_space.n

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

    # visualize inputs
    to_vis_format = lambda batch: tf.expand_dims(tf.transpose(batch[0, :, :, :], perm=[2, 0, 1]), 3)
    tf.image_summary("observation_now", to_vis_format(obs_t_ph), max_images=4)
    tf.image_summary("observation_next", to_vis_format(obs_tp1_ph), max_images=4)
    tf.scalar_summary("env_input/action", act_t_ph[0])
    tf.scalar_summary("env_input/reward", rew_t_ph[0])

    # casting to float on GPU ensures lower data transfer times.
    obs_t_float = tf.cast(obs_t_ph, tf.float32) / 255.0
    obs_tp1_float = tf.cast(obs_tp1_ph, tf.float32) / 255.0

    total_error, q_func_vars, target_q_func_vars, q_rapid_t = losses.generate_losses(
                    q_func,
                    num_actions,
                    gamma,
                    batch_size,

                    obs_t_float,
                    obs_tp1_float,
                    act_t_ph,
                    rew_t_ph,
                    done_mask_ph)

    tf.scalar_summary("loss/total", total_error)

    # construct optimization op (with gradient clipping)
    learning_rate = tf.placeholder(tf.float32, (), name="learning_rate")
    tf.scalar_summary("learning_rate", learning_rate)
    optimizer = optimizer_spec.constructor(learning_rate=learning_rate, **optimizer_spec.kwargs)
    train_fn = minimize_and_clip(optimizer, total_error,
                                 var_list=q_func_vars,
                                 clip_val=grad_norm_clipping)

    # update_target_fn will be called periodically to copy Q network to target Q network
    update_target_fn = []
    for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                               sorted(target_q_func_vars, key=lambda v: v.name)):
        update_target_fn.append(var_target.assign(var))
    update_target_fn = tf.group(*update_target_fn)

    summary_op = tf.merge_all_summaries()
    model_save_path = os.path.join(model_store_root_path, FLAGS.method_name)
    summary_writer = tf.train.SummaryWriter(
        model_save_path,
        graph_def=session.graph.as_graph_def(add_shapes=True))

    # construct the replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len, num_actions)

    ###############
    # RUN ENV     #
    ###############
    model_initialized = False
    num_param_updates = 0
    mean_episode_reward = -float('nan')
    best_mean_episode_reward = -float('inf')
    best_reward = -float('inf')
    damage_per_episode = 0
    damage_list = []
    damage_counter = 0
    mean_damage_counter = 0
    if env_train:
        last_obs = env_train.reset()
    LOG_EVERY_N_STEPS = 10000

    log_file = os.path.join(model_save_path, "log.txt")
    saver = tf.train.Saver()

    # HDF files are only used for Enduro demonstration
    if FLAGS.demo_mode == 'hdf':
        replay_buffer = get_hdf_demo(FLAGS.demo_file_path, replay_buffer)
    elif FLAGS.demo_mode == 'replay':
        replay_buffer = load_replay_pickle(FLAGS.demo_file_path, FLAGS.dataset_size)
    elif FLAGS.demo_mode == 'dqfd':
        replay_buffer_demo = load_replay_pickle(FLAGS.demo_file_path, FLAGS.dataset_size)
    elif FLAGS.demo_mode == 'no_demo':
        pass
    else:
        raise ValueError("invalid FLAGS.demo_mode = %s" % FLAGS.demo_mode)


    # load the pretrained models
    # now it does not support resume training from the middle
    if (FLAGS.inenv_finetune or FLAGS.inenv_eval) and (not FLAGS.train_from_scratch):
        print("loading model from ", FLAGS.ckpt_path)
        ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_path)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_path = ckpt.model_checkpoint_path
        else:
            print('No checkpoint file found')
            session.close()
            return

        saver.restore(session, ckpt_path)
        print('model loaded!!!! %s' % ckpt_path + '*'*30)
        model_initialized = True


    try:
        for t in itertools.count():
            ### 1. Check stopping criterion
            if stopping_criterion is not None and stopping_criterion(env_info, t):
                break

            ### 2. Step the env and store the transition
            if FLAGS.collect_Q_experience:
                idx = replay_buffer.store_frame(last_obs)

                action_dist_this = np.ones((num_actions), dtype=np.float32) / num_actions
                # update the sampling distribution if we have a model
                if model_initialized:
                    eps = exploration.value(t)
                    recent_obs = replay_buffer.encode_recent_observation()[np.newaxis, ...]
                    q_values = session.run(q_rapid_t, feed_dict={obs_t_ph: recent_obs})
                    max_action = np.argmax(np.squeeze(q_values))
                    greedy_dist = np.zeros((num_actions), dtype=np.float32)
                    greedy_dist[max_action] = 1.0

                    action_dist_this = eps * action_dist_this + (1 - eps) * greedy_dist

                action = np.random.choice(num_actions, p=np.squeeze(action_dist_this))

                obs, reward, done, info = env_train.step(action)
                if info != {} and "damage" in info:
                    # print(info)
                    damage = int(info['damage'])
                    next_damage = int(info['next_damage'])
                    if next_damage - damage > 1:
                        damage_per_episode += 1
                if done:
                    obs = env_train.reset()
                    damage_list += [damage_per_episode]
                    damage_per_episode = 0

                replay_buffer.store_effect(idx, action, reward, done)
                last_obs = obs
            else:
                # if not collecting any Q experience in the env_train, then we are on the learning from demo phase
                # we would stop after a fix number of steps
                if t > FLAGS.demo_step:
                    break

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
                replay_buffer.can_sample(batch_size) and
                not FLAGS.inenv_eval):

                # save the model periodically
                if t % 100000 == 0 and FLAGS.save_model:
                    model_save_path = os.path.join(model_store_root_path, FLAGS.method_name)
                    if not os.path.exists(model_save_path):
                        tf.gfile.MakeDirs(model_save_path)
                    save_path = saver.save(session,
                                           os.path.join(model_save_path, "model_%s.ckpt" % str(t)))
                    print('saved at: ', save_path)

                # (a) sample from the replay buffer, depending on the demo mode
                if FLAGS.demo_mode == 'dqfd':
                    demo_size = int(batch_size * FLAGS.demo_portion)
                    obs_t_batch_demo, act_t_batch_demo, rew_t_batch_demo, obs_tp1_batch_demo, done_mask_demo = \
                        replay_buffer_demo.sample(demo_size, FLAGS.group_name)
                    obs_t_batch, act_t_batch, rew_t_batch, obs_tp1_batch, done_mask = \
                        replay_buffer.sample(batch_size-demo_size, FLAGS.group_name)

                    obs_t_batch   = np.concatenate((obs_t_batch_demo, obs_t_batch))
                    act_t_batch   = np.concatenate((act_t_batch_demo, act_t_batch))
                    rew_t_batch   = np.concatenate((rew_t_batch_demo, rew_t_batch))
                    obs_tp1_batch = np.concatenate((obs_tp1_batch_demo, obs_tp1_batch))
                    done_mask     = np.concatenate((done_mask_demo  , done_mask))
                else:
                    obs_t_batch, act_t_batch, rew_t_batch, obs_tp1_batch, done_mask = \
                        replay_buffer.sample(batch_size, FLAGS.group_name)

                # (b)
                if not model_initialized:
                    initialize_interdependent_variables(session, tf.all_variables(), {
                        obs_t_ph: obs_t_batch,
                        obs_tp1_ph: obs_tp1_batch,
                    })
                    model_initialized = True

                # (c) train it
                feed_dict = {
                    obs_t_ph: obs_t_batch,
                    act_t_ph: act_t_batch,
                    rew_t_ph: rew_t_batch,
                    obs_tp1_ph: obs_tp1_batch,
                    done_mask_ph: done_mask,
                    learning_rate: optimizer_spec.lr_schedule.value(t),
                }
                if t % FLAGS.summary_interval == 0:
                    _, summary_value = session.run([train_fn, summary_op], feed_dict)
                else:
                    session.run(train_fn, feed_dict)
                num_param_updates += 1

                # (d) update target network periodically
                if num_param_updates % target_update_freq == 0:
                    session.run(update_target_fn)

            ### 4. Log progress

            # evaluating in the environment, when off policy training is used
            # Warning: this evaluation does not use target network!
            # TODO: make sure the environment Monitor return the undiscounted reward
            reward_calc = None
            # In environment evaluation starts here
            if FLAGS.eval_freq > 0 and t % FLAGS.eval_freq == 0 and model_initialized:
                print('_' * 50)
                print('Start Evaluating at TimeStep %d' % t)
                eps = 0.05
                if FLAGS.val_set:
                    # evaluate on a validation dataset to compute the Bellman error.
                    bellman_error = eval_valset(q_rapid_t, obs_t_ph, FLAGS.val_set_file, session, gamma)
                    #  eval on val set only and could deal with all replay buffers
                    inspect_q_values(q_rapid_t, obs_t_ph, session, replay_buffer)

                reward_calc, frame_counter, damage_counter, damage_inds = \
                    eval_policy(env_test, q_rapid_t, obs_t_ph, session, eps, frame_history_len, num_actions, img_c)
                best_reward = np.max([best_reward, reward_calc])

                # output on the command line and into the files
                print("the frame counter is %f" % frame_counter)
                print("test reward %f" % reward_calc)
                print("best test reward %f" % best_reward)
                print("damage counter %d" % damage_counter)
                with open(log_file, 'a') as f:
                    print(t, reward_calc, best_reward, file=f)
                with open(log_file+".damage_indicators", 'a') as f:
                    print(damage_inds, file=f)

            # evaluating with the current environment
            if FLAGS.eval_freq <= 0:
                episode_rewards = get_wrapper_by_name(env_train, "Monitor").get_episode_rewards()
                if len(episode_rewards) > 0:
                    mean_episode_reward = np.mean(episode_rewards[-100:])
                    reward_calc = episode_rewards[-1]
                if len(episode_rewards) > 100:
                    best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)
                if len(damage_list) > 0:
                    mean_damage_counter = np.mean(damage_list[-100:])
                    damage_counter = damage_list[-1]

                if t % LOG_EVERY_N_STEPS == 0 and model_initialized:
                    print("mean reward (100 episodes) %f" % mean_episode_reward)
                    print("best mean reward %f" % best_mean_episode_reward)
                    print("episodes %d" % len(episode_rewards))
                    print("exploration %f" % exploration.value(t))
                    print("learning_rate %f" % optimizer_spec.lr_schedule.value(t))
                    print("mean_damage_counter %d" % mean_damage_counter)
                    print("damage_counter %d" % damage_counter)
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
                    summary.value.add(tag='damage_number', simple_value=damage_counter)
                    if FLAGS.val_set and bellman_error is not None:
                        summary.value.add(tag='bellman_error', simple_value=bellman_error)
                summary_writer.add_summary(summary, t)

            if FLAGS.inenv_eval:
                # this means after training, we re-eval it, so break after we have done evaluation once
                break
    except KeyboardInterrupt:
        print("Control C pressed. Saving model before exit. ")
        model_save_path = os.path.join(model_store_root_path, FLAGS.method_name)
        saver.save(session, os.path.join(model_save_path, "model_%s.ckpt" % str(t)))
        sys.exit()
