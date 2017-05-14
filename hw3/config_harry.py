import sys
import os
import inspect
sys.path.append('../')
import tensorflow as tf
FLAGS = tf.app.flags.FLAGS


def common_setting():
    FLAGS.ddqn = False

    FLAGS.demo_mode = 'replay'
    FLAGS.demo_file_path = '/data/hxu/cs294-112/hw3/link_data/hardQ_expert.p'  #'/data/hxu/cs294-112/hw3/DQfD/enduro-egs.h5'
    FLAGS.bad_dir = '/data/hxu/cs294-112/hw3/link_data/bad_demo.p'
    FLAGS.collect_Q_experience = False

    FLAGS.eval_freq = 10000
    FLAGS.tiny_explore = 0.01

    FLAGS.learning_starts = 0
    FLAGS.learning_stage = True
    FLAGS.batch_size = 32




def cross_entropy_T():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion)
    common_setting()
    FLAGS.group_name = 'supervised'

    FLAGS.core_num = '4'
    FLAGS.supervise_cross_entropy_loss_weight = 1.0

def hinge_dqfd_T():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion)
    common_setting()
    FLAGS.group_name = 'supervised'

    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.core_num = '5'

def hinge_standard_T():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion)
    common_setting()
    FLAGS.group_name = 'supervised'

    FLAGS.supervise_hinge_standard_loss_weight = 1.0
    FLAGS.core_num = '6'

def hard_Q_T():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion)
    common_setting()
    FLAGS.group_name = 'rl'

    FLAGS.core_num = '7'
    FLAGS.hard_Q_loss_weight = 1.0
    # optionally set: lr_schedule, exploration_schedule

def soft_Q_T():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion)
    common_setting()
    FLAGS.group_name = 'rl'

    FLAGS.core_num = '4'
    FLAGS.soft_Q_loss_weight = 1.0
def DQFD_no_l2_T():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion)
    common_setting()
    FLAGS.group_name = 'dqfd'

    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.core_num = '5'

def DQfD_full_T():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion)
    common_setting()
    FLAGS.group_name = 'dqfd'

    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.l2_regularization_loss_weight = 1.0e-5
    FLAGS.core_num = '6'

def policy_gradient_soft_1_step_T():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion)
    common_setting()
    FLAGS.group_name = 'rl'

    FLAGS.policy_gradient_soft_1_step = 1.0
    FLAGS.core_num = '7'

def policy_gradient_soft_1_step_new_T():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion)
    common_setting()
    FLAGS.group_name = 'rl'

    FLAGS.policy_gradient_soft_1_step = 1.0
    FLAGS.core_num = '7'

def test_test():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion)
    common_setting()

    #FLAGS.dataset_size = 100
    FLAGS.core_num = '0'
    FLAGS.supervise_cross_entropy_loss_weight = 1.0


def collect_demonstration():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion)
    common_setting()

    FLAGS.core_num = '5'
    FLAGS.learning_starts = 0
    FLAGS.learning_stage = False
    FLAGS.ckpt_path = '/data/hxu/cs294-112/hw3/data/'
    FLAGS.tiny_explore = 0.1
    FLAGS.soft_Q_alpha = 0.01
    FLAGS.m_bad = 300
    FLAGS.m_good = 300
    #FLAGS.summary_interval = 10


def hard_Q_in_env():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion)
    common_setting()

    FLAGS.core_num = '0'

    # Q learning specific
    FLAGS.eval_freq = -1
    FLAGS.demo_mode = "no_demo"
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 10000

def soft_Q_in_env():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion)
    common_setting()
    FLAGS.core_num = '0'

    # Q learning specific
    FLAGS.eval_freq = -1
    FLAGS.demo_mode = "no_demo"
    FLAGS.soft_Q_loss_weight = 1.0
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 10000
    FLAGS.soft_Q_alpha = 0.1


use_this_config = test_test

