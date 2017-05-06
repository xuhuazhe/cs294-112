import sys
import os
import inspect
sys.path.append('../')
import tensorflow as tf
FLAGS = tf.app.flags.FLAGS


def common_setting():
    FLAGS.ddqn = False

    FLAGS.demo_mode = 'hdf'
    FLAGS.demo_file_path = '/data/hxu/cs294-112/hw3/DQfD/enduro-egs.h5'
    FLAGS.collect_Q_experience = False

    FLAGS.eval_freq = 10000
    FLAGS.tiny_explore = 0.01

    FLAGS.learning_starts = 0


def hard_Q_on_demonstration():
    tag = inspect.stack()[0][3]
    FLAGS.method_name = tag
    common_setting()

    FLAGS.core_num = '0'
    # optionally set: lr_schedule, exploration_schedule

use_this_config = hard_Q_on_demonstration
