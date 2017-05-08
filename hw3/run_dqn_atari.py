import argparse
import gym
from gym import wrappers
import os.path as osp
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.slim as slim
import os

import dqn
from dqn_utils import *
from atari_wrappers import *
import sys
from config import *
import Q_expert

FLAGS = tf.app.flags.FLAGS

# DQN types
tf.app.flags.DEFINE_boolean('ddqn', False,
                            """Enable double Q bellman Update""")

# demonstration related
tf.app.flags.DEFINE_string('demo_mode', 'hdf',
                           """hdf: load data from hdf, replay: from replay buffer pickle"""
                           """no_demo: skip the demo exp pool""")
tf.app.flags.DEFINE_string('demo_file_path', '/data/hxu/cs294-112/hw3/DQfD/enduro-egs.h5',
                           """Demonstration file path""")
tf.app.flags.DEFINE_boolean('collect_Q_experience', False,
                            """Do we want to add Q learning sample to the replay buffer""")
tf.app.flags.DEFINE_boolean('learning_stage', True,
                            """Do we want to learn or just collection""")
tf.app.flags.DEFINE_string('ckpt_path', '/data/hxu/cs294-112/hw3/link_data/',
                           """where did we save the checkpoints""")

# evaluation related
tf.app.flags.DEFINE_integer('eval_freq', -1,
                            """evaluation frequency""")
tf.app.flags.DEFINE_float('tiny_explore', 0.01,
                            """the explore rate for evaluating mode""")
tf.app.flags.DEFINE_integer('summary_interval', 10000,
                            """summary frequency""")

###### weight for losses ##########
tf.app.flags.DEFINE_float('supervise_cross_entropy_loss_weight', -1.0,
                            """weight for supervised learning""")
tf.app.flags.DEFINE_float('hard_Q_loss_weight', -1.0,
                            """weight for Q learning""")
tf.app.flags.DEFINE_float('l2_regularization_loss_weight', -1.0,
                            """weight regularization loss""")
tf.app.flags.DEFINE_float('soft_Q_loss_weight', -1.0,
                            """weight regularization loss""")
tf.app.flags.DEFINE_float('soft_Q_alpha', 1.0,
                            """""")
tf.app.flags.DEFINE_float('supervise_hinge_DQfD_loss_weight', -1.0,
                            """weight for supervised learning""")
tf.app.flags.DEFINE_float('supervise_hinge_standard_loss_weight', -1.0,
                            """weight for supervised learning""")

# resource related
tf.app.flags.DEFINE_string('core_num', '0',
                           """gpu number""")
tf.app.flags.DEFINE_string('method_name', 'vanilla',
                           """the method name we want to use, should be set automatically by config func""")

# optimization related hyperparams
tf.app.flags.DEFINE_integer('batch_size', 32,
                           """optimization batch size""")
tf.app.flags.DEFINE_integer('frame_history_len', 4,
                           """frame_history_len""")

# Other hyper parameters
tf.app.flags.DEFINE_boolean('save_model', True,
                            """save the model of Q""")
tf.app.flags.DEFINE_integer('learning_starts', 50000,
                            """learning_starts point, 50000 for Q learning, 0 for demonstration""")

def atari_model(img_in, num_actions, scope, reuse=False):
    # as described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    with slim.arg_scope([layers.convolution2d, layers.fully_connected], outputs_collections="activation_collection"):
        with tf.variable_scope(scope, reuse=reuse):
            out = img_in
            with tf.variable_scope("convnet"):
                # original architecture
                out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
                out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
                out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
            out = layers.flatten(out)
            with tf.variable_scope("action_value"):
                out = layers.fully_connected(out, num_outputs=512,         activation_fn=tf.nn.relu)
                out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)

            return out

def atari_learn(env,
                session,
                num_timesteps):
    # TODO: principle of Adam and more parameters
    optimizer = dqn.OptimizerSpec(
        constructor=tf.train.AdamOptimizer,
        kwargs=dict(epsilon=1e-4),
        lr_schedule=FLAGS.lr_schedule
    )

    # TODO: t input is not used here
    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

    # TODO: better hyper parameters here
    dqn.learn(
        env,
        q_func=atari_model,
        optimizer_spec=optimizer,
        session=session,
        exploration=FLAGS.exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=1000000,
        batch_size=FLAGS.batch_size,
        gamma=0.99,
        learning_starts=FLAGS.learning_starts,
        learning_freq=4,
        frame_history_len=FLAGS.frame_history_len,
        target_update_freq=10000,
        grad_norm_clipping=10
    )
    env.close()

def atari_collect(env,
                  session,
                  num_timesteps):
    num_steps = 300000

    # TODO: t input is not used here
    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_steps*4

    Q_expert.collect(
        env,
        q_func=atari_model,
        session=session,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=num_steps,
        frame_history_len=FLAGS.frame_history_len)
    env.close()



def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']

def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        raise ValueError("Import tensorflow failed")
    else:
        tf.set_random_seed(i) 
    np.random.seed(i)
    random.seed(i)

def get_session():
    tf.reset_default_graph()
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    tf_config.gpu_options.allow_growth = True
    session = tf.Session(config=tf_config)
    print("AVAILABLE GPUS: ", get_available_gpus())
    return session

def get_env(task, seed):
    env_id = task.env_id

    env = gym.make(env_id)

    set_global_seeds(seed)
    env.seed(seed)

    model_save_path = os.path.join('./link_data/', FLAGS.method_name)
    env = wrappers.Monitor(env, model_save_path, force=True)
    env = wrap_deepmind(env)

    return env


def default_parameters(**kwargs):
    # This is just a rough estimate
    num_timesteps = kwargs["num_timesteps"]
    num_iterations = float(num_timesteps) / 4.0

    lr_multiplier = 1.0

    FLAGS.lr_schedule = PiecewiseSchedule([
                                         (0,                   1e-4 * lr_multiplier),
                                         (num_iterations / 10, 1e-4 * lr_multiplier),
                                         (num_iterations / 2,  5e-5 * lr_multiplier),
                                    ],
                                    outside_value=5e-5 * lr_multiplier)

    FLAGS.exploration_schedule = PiecewiseSchedule(
        [
            (0, 1.0),
            (1e6, 0.1),
            (num_iterations / 2, 0.01),
        ], outside_value=0.01
    )

def main(_):

    benchmark = gym.benchmark_spec('Atari40M')
    # Change the index to select a different game.
    task = benchmark.tasks[2]
    default_parameters(num_timesteps=task.max_timesteps)
    use_this_config()
    #collect_demonstration()
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.core_num

    # Run training
    seed = 0 # Use a seed of zero (you may want to randomize the seed!)
    env = get_env(task, seed)
    session = get_session()

    if FLAGS.learning_stage:
        atari_learn(env, session, num_timesteps=task.max_timesteps)
    else:
        atari_collect(env, session, num_timesteps=task.max_timesteps)

if __name__ == "__main__":
    tf.app.run()
