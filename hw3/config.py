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
    print("config name=", tag)
    FLAGS.method_name = tag
    common_setting()

    FLAGS.core_num = '0'
    FLAGS.hard_Q_loss_weight = 1
    # optionally set: lr_schedule, exploration_schedule



def supervised_learning():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag
    common_setting()
    
    FLAGS.core_num = '1'
    FLAGS.supervise_cross_entropy_loss_weight = 1

def soft_Q_on_demonstration():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag
    common_setting()

    FLAGS.core_num = '6'
    FLAGS.soft_Q_loss_weight = 1

def collect_demonstration():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag
    common_setting()

    FLAGS.core_num = '5'
    FLAGS.learning_starts = 0
    FLAGS.learning_stage = False
    FLAGS.ckpt_path = '/data/hxu/cs294-112/hw3/data/'
    FLAGS.tiny_explore = 0.01
    FLAGS.soft_Q_alpha = 0.01
    #FLAGS.summary_interval = 10

def DQfD():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag
    common_setting()

    FLAGS.core_num = '0'
    FLAGS.learning_starts = 0



use_this_config = collect_demonstration
