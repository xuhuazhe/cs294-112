import math
import sys
import os
import inspect
sys.path.append('../')
import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

def common_setting():
    FLAGS.ddqn = False
    FLAGS.demo_mode = 'hdf'
    FLAGS.use_env_reward = False
    FLAGS.save_model = True #Note: this is only for debug
    FLAGS.collect_Q_experience = False
    FLAGS.learning_starts = 0
    FLAGS.tiny_explore = 0.01
    FLAGS.eval_freq = 10000

def hard_Q_on_demonstration():
    print('Hard Q on Demo Config!')
    FLAGS.core_num = '0'
    FLAGS.method_name = 'hard_Q_demo'
    FLAGS.demo_hdf_dir = '/data/hxu/cs294-112/hw3/DQfD/enduro-egs.h5'
    FLAGS.temporal_diff_loss_weight = 1

def supervised_learning():
    print('Supervised Learning Config!')
    FLAGS.core_num = '1'
    FLAGS.method_name = 'supervised_learning'
    FLAGS.demo_hdf_dir = '/data/hxu/cs294-112/hw3/DQfD/enduro-egs.h5'
    FLAGS.cross_entropy_loss_weight = 1