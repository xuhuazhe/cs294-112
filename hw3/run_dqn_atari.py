import argparse
import gym
from gym import wrappers
import os.path as osp
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import os

import dqn
from dqn_utils import *
from atari_wrappers import *
import sys
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean('ddqn', False,
                            """Enable double Q bellman Update""")
tf.app.flags.DEFINE_string('demo_mode', 'replay',
                           """hdf: load data from hdf, replay: from replay buffer pickle"""
                           """no_demo: skip the demo exp pool""")
tf.app.flags.DEFINE_string('method_name', 'vanilla',
                           """the method name we want to use""")
tf.app.flags.DEFINE_boolean('save_model', True,
                            """save the model of Q""")
# TODO: merge demo dir flag into one flag
tf.app.flags.DEFINE_string('demo_hdf_dir', '/data/hxu/cs294-112/hw3/DQfD/enduro-egs.h5',
                           """HDF Demonstration dir""")
tf.app.flags.DEFINE_string('pickle_dir', '/data/hxu/cs294-112/hw3/DQfD/enduro-egs.p',
                           """pickle Demonstration dir""")
tf.app.flags.DEFINE_boolean('collect_Q_experience', False,
                            """Do we want to add Q learning sample to the replay buffer""")
# TODO: make sure this flag is not used in the learning from demonstration case
tf.app.flags.DEFINE_integer('learning_starts', 50000,
                            """learning_starts point, 50000for Q learning, 0 for demonstration""")
def atari_model(img_in, num_actions, scope, reuse=False):
    # as described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
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
    # This is just a rough estimate
    # TODO: replace 4.0 by the framehistory
    num_iterations = float(num_timesteps) / 4.0

    lr_multiplier = 1.0
    # TODO: better learning rate schedules?
    lr_schedule = PiecewiseSchedule([
                                         (0,                   1e-4 * lr_multiplier),
                                         (num_iterations / 10, 1e-4 * lr_multiplier),
                                         (num_iterations / 2,  5e-5 * lr_multiplier),
                                    ],
                                    outside_value=5e-5 * lr_multiplier)
    # TODO: principle of Adam and more parameters
    optimizer = dqn.OptimizerSpec(
        constructor=tf.train.AdamOptimizer,
        kwargs=dict(epsilon=1e-4),
        lr_schedule=lr_schedule
    )

    # TODO: t input is not used here
    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

    # TODO: better exploration schedule?
    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1.0),
            (1e6, 0.1),
            (num_iterations / 2, 0.01),
        ], outside_value=0.01
    )

    # TODO: better hyper parameters here
    dqn.learn(
        env,
        q_func=atari_model,
        optimizer_spec=optimizer,
        session=session,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=1000000,
        batch_size=32,
        gamma=0.99,
        learning_starts=FLAGS.learning_starts,
        learning_freq=4,
        frame_history_len=int(sys.argv[2]),
        target_update_freq=10000,
        grad_norm_clipping=10
    )
    env.close()

def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']

def set_global_seeds(i):
    # TODO: cuda visable device should not be set here, and should be provided by a flag
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    try:
        import tensorflow as tf
    except ImportError:
        pass
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
    # TODO: the expt_dir might not be available on the machine
    expt_dir = os.path.join('/tmp/',FLAGS.method_name,'vid_dir2/')
    env = wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True)
    env = wrap_deepmind(env)

    return env
# TODO: this should be moved to the seperate config file
def common_setting():
    FLAGS.ddqn = False
    FLAGS.demo_mode = 'hdf'
    FLAGS.save_model = True #Note: this is only for debug
    FLAGS.collect_Q_experience = False
    FLAGS.learning_starts = 0
def main(_):
    # Get Atari games.
    common_setting()

    ### set all the names ###  #TODO: set a config file to set flags
    FLAGS.method_name = 'test_code'
    FLAGS.demo_hdf_dir = '/data/hxu/cs294-112/hw3/DQfD/enduro-egs.h5'

    benchmark = gym.benchmark_spec('Atari40M')

    # Change the index to select a different game.
    task = benchmark.tasks[2]

    # Run training
    seed = 0 # Use a seed of zero (you may want to randomize the seed!)
    env = get_env(task, seed)
    session = get_session()
    atari_learn(env, session, num_timesteps=task.max_timesteps)

if __name__ == "__main__":

    tf.app.run()
