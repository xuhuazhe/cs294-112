import sys
import os
import inspect
sys.path.append('../')
import tensorflow as tf
from dqn_utils import *
FLAGS = tf.app.flags.FLAGS


def common_setting():
    FLAGS.ddqn = False

    FLAGS.demo_mode = 'hdf'
    FLAGS.demo_file_path = '/data/hxu/cs294-112/hw3/link_data/hardQ_expert.p'
    #FLAGS.demo_file_path   = '/backup/hxu/cs294-112/hw3/link_data/dmformat_demo/enduro-dm-egs-00.h5'  #'/data/hxu/cs294-112/hw3/DQfD/enduro-egs.h5'
    #FLAGS.demo_file_path_1 = '/backup/hxu/cs294-112/hw3/link_data/dmformat_demo/enduro-dm-egs-01.h5'
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
    FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion) + '_dm'
    common_setting()
    #FLAGS.group_name = 'supervised'

    FLAGS.core_num = '0'
    FLAGS.supervise_cross_entropy_loss_weight = 1.0

def hinge_dqfd_T():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion) + '_dm'
    common_setting()
    #FLAGS.group_name = 'supervised'

    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.core_num = '1'

def hinge_standard_T():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion) + '_dm'
    common_setting()
    #FLAGS.group_name = 'supervised'

    FLAGS.supervise_hinge_standard_loss_weight = 1.0
    FLAGS.core_num = '2'

def hard_Q_T():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion) + '_dm'
    common_setting()
    #FLAGS.group_name = 'rl'

    FLAGS.core_num = '3'
    FLAGS.hard_Q_loss_weight = 1.0
    # optionally set: lr_schedule, exploration_schedule

def soft_Q_T():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion) + '_dm'
    common_setting()
    #FLAGS.group_name = 'rl'

    FLAGS.core_num = '0'
    FLAGS.soft_Q_loss_weight = 1.0
def DQFD_no_l2_T():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion) + '_dm'
    common_setting()
    #FLAGS.group_name = 'dqfd'

    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.core_num = '1'

def DQfD_full_T():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion) + '_dm'
    common_setting()
    #FLAGS.group_name = 'dqfd'

    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.l2_regularization_loss_weight = 1.0e-5
    FLAGS.core_num = '2'

def policy_gradient_soft_1_step_T():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion)
    common_setting()
    #FLAGS.group_name = 'rl'

    FLAGS.policy_gradient_soft_1_step = 1.0
    FLAGS.core_num = '3'

def policy_gradient_soft_1_step_new_T():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion) + '_dm'
    common_setting()
    #FLAGS.group_name = 'rl'

    FLAGS.policy_gradient_soft_1_step = 1.0
    FLAGS.core_num = '3'

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
    FLAGS.method_name = tag + 'enduro_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion)
    common_setting()
    FLAGS.core_num = '2'

    # Q learning specific
    FLAGS.eval_freq = -1
    FLAGS.demo_mode = "no_demo"
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000

def soft_Q_in_env():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    common_setting()
    FLAGS.core_num = '2'

    # Q learning specific
    FLAGS.eval_freq = -1
    FLAGS.demo_mode = "no_demo"
    FLAGS.soft_Q_loss_weight = 1.0
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000
    FLAGS.soft_Q_alpha = 0.001
    FLAGS.method_name = tag + 'enduro_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion) + '_' +str(FLAGS.soft_Q_alpha)

def cross_entropy_finetune():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion)
    common_setting()
    FLAGS.eval_freq = -1
    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000
    FLAGS.inenv_finetune = True

    FLAGS.ckpt_path = '/data/hxu/cs294-112/hw3/link_data/cross_entropy_T/'
    FLAGS.core_num = '1'
    FLAGS.hard_Q_loss_weight = 1.0


def hinge_standard_finetune():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion)
    common_setting()
    FLAGS.eval_freq = -1
    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000
    FLAGS.inenv_finetune = True

    FLAGS.ckpt_path = '/data/hxu/cs294-112/hw3/link_data/hinge_standard_T/'
    FLAGS.core_num = '2'
    FLAGS.hard_Q_loss_weight = 1.0

def hard_Q_finetune():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion)
    common_setting()
    FLAGS.eval_freq = -1
    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000
    FLAGS.inenv_finetune = True

    FLAGS.ckpt_path = '/data/hxu/cs294-112/hw3/link_data/hard_Q_T/'
    FLAGS.core_num = '3'
    FLAGS.hard_Q_loss_weight = 1.0

def soft_Q_finetune():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion)
    common_setting()
    FLAGS.eval_freq = -1
    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000

    FLAGS.ckpt_path = ''
    FLAGS.core_num = '3'
    FLAGS.soft_Q_loss_weight = 1.0

def DQfD_full_finetune():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion) + '_constant_e'
    common_setting()
    FLAGS.eval_freq = -1
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000

    FLAGS.ckpt_path = '/data/hxu/cs294-112/hw3/link_data/DQfD_full_T/'
    FLAGS.inenv_finetune = True
    FLAGS.demo_mode = "dqfd"
    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.l2_regularization_loss_weight = 1.0e-5
    FLAGS.core_num = '2'
    FLAGS.exploration_schedule = PiecewiseSchedule(
        [
            (0, 0.01),
            (1e6, 0.01),
            (1e7, 0.01),
        ], outside_value=0.01
    )

def policy_gradient_soft_1_step_new_finetune():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion)
    common_setting()
    FLAGS.eval_freq = -1
    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000

    FLAGS.policy_gradient_soft_1_step = 1.0
    FLAGS.core_num = '2'

#_______________________starts_here_the_new_experiments______________________






use_this_config = test_test

