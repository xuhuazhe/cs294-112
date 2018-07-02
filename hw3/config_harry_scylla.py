import sys
import os
import inspect
sys.path.append('../')
import tensorflow as tf
from dqn_utils import *
FLAGS = tf.app.flags.FLAGS
from datetime import datetime


def common_setting():
    FLAGS.ddqn = False
    FLAGS.demo_mode = 'replay'
    if FLAGS.tag_prefix == 'human':
        FLAGS.demo_mode = 'hdf'
        FLAGS.demo_file_path = '/data/hxu/cs294-112/hw3/link_data/dmformat_demo/enduro-dm-egs-00.h5, /data/hxu/cs294-112/hw3/link_data/dmformat_demo/enduro-dm-egs-01.h5'
    else:
        FLAGS.demo_file_path = '/data/hxu/cs294-112/hw3/link_data/300000enduro_perfect'
    #FLAGS.demo_file_path   = '/backup/hxu/cs294-112/hw3/link_data/dmformat_demo/enduro-dm-egs-00.h5'  #'/data/hxu/cs294-112/hw3/DQfD/enduro-egs.h5'
    #FLAGS.demo_file_path_1 = '/backup/hxu/cs294-112/hw3/link_data/dmformat_demo/enduro-dm-egs-01.h5'
    FLAGS.bad_dir = ''#'/data/hxu/cs294-112/hw3/link_data/bad_demo.p'
    FLAGS.collect_Q_experience = False

    FLAGS.eval_freq = 10000
    #FLAGS.tiny_explore = 0.01

    FLAGS.learning_starts = 0
    FLAGS.learning_stage = True
    FLAGS.batch_size = 32

def common_data_size(sz):
    FLAGS.dataset_size = int(sz)#30000
    FLAGS.method_name = FLAGS.method_name + "_" + str(FLAGS.dataset_size)




def cross_entropy_T():
    FLAGS.tag_prefix = 'Q_expert_s'
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    #FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion) + '_dm'
    yang_common_setting(tag)
    common_setting()
    #FLAGS.demo_mode = 'replay'
    #FLAGS.group_name = 'supervised'
    #FLAGS.demo_file_path = '/data/hxu/cs294-112/hw3/link_data/300000enduro_perfect'
    FLAGS.core_num = '0'
    FLAGS.supervise_cross_entropy_loss_weight = 1.0


def hinge_dqfd_T():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion) + '_dm'
    common_setting()
    #FLAGS.group_name = 'supervised'
    FLAGS.demo_mode = 'replay'
    FLAGS.demo_file_path = '/data/hxu/cs294-112/hw3/link_data/300000enduro_perfect'
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
    #FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion) + '_dm'
    common_setting()
    #FLAGS.group_name = 'dqfd'

    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.core_num = '1'

def DQfD_full_T():
    FLAGS.tag_prefix = 'Q_expert_'
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    #FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion) + '_dm'
    yang_common_setting(tag)
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

    FLAGS.core_num = '0'
    FLAGS.learning_starts = 0
    FLAGS.learning_stage = False
    FLAGS.ckpt_path = '/data/hxu/cs294-112/hw3/data/'
    #FLAGS.tiny_explore = 0.1
    FLAGS.soft_Q_alpha = 0.1
    FLAGS.m_bad = 1
    FLAGS.m_good = 5
    FLAGS.exploration_schedule = PiecewiseSchedule(
        [
            (0, 0.01),
            (1e6, 0.01),
            (1e7, 0.01),
        ], outside_value=0.01
    )
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

def cross_entropy_plot():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion) + '_medio'
    common_setting()
    #FLAGS.group_name = 'supervised'

    FLAGS.core_num = '0'
    FLAGS.supervise_cross_entropy_loss_weight = 1.0

def hinge_dqfd_plot():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion) + '_medio'
    common_setting()
    #FLAGS.group_name = 'supervised'

    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.core_num = '1'

def hard_Q_plot():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion) + '_medio'
    common_setting()
    #FLAGS.group_name = 'rl'

    FLAGS.core_num = '2'
    FLAGS.hard_Q_loss_weight = 1.0
    # optionally set: lr_schedule, exploration_schedule

def soft_Q_plot():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion) + '_medio'
    common_setting()
    #FLAGS.group_name = 'rl'

    FLAGS.core_num = '3'
    FLAGS.soft_Q_loss_weight = 1.0

def DQfD_no_l2_softQ_plot():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion) + '_medio'
    common_setting()
    #FLAGS.group_name = 'dqfd'

    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.soft_Q_loss_weight = 1.0
    FLAGS.core_num = '0'

def DQfD_no_l2_plot():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion) + '_medio'
    common_setting()
    #FLAGS.group_name = 'dqfd'

    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.core_num = '1'

def policy_gradient_soft_1_step_plot():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion) + '_medio'
    common_setting()
    #FLAGS.group_name = 'rl'

    FLAGS.policy_gradient_soft_1_step = 1.0
    FLAGS.core_num = '2'

def exp_policy_grad_weighting_plot():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion) + '_medio'
    common_setting()
    #FLAGS.group_name = 'rl'

    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.core_num = '3'

def exp_advantage_diff_learning_plot():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion) + '_medio'
    common_setting()
    #FLAGS.group_name = 'rl'

    FLAGS.exp_advantage_diff_learning = 1.0
    FLAGS.core_num = '2'

#_________________________double_Q_test________________________________
def DQfD_no_l2_doubleQ_plot():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion) + '_plot'
    common_setting()
    #FLAGS.group_name = 'dqfd'

    FLAGS.demo_file_path   = '/backup/hxu/cs294-112/hw3/link_data/dmformat_demo/enduro-dm-egs-00.h5'
    FLAGS.demo_file_path_1 = '/backup/hxu/cs294-112/hw3/link_data/dmformat_demo/enduro-dm-egs-01.h5'
    FLAGS.ddqn = True
    FLAGS.demo_mode = 'hdf'
    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.core_num = '0'

#________________________cross_entropy_finetune_dm________________
def cross_entropy_dm_finetune_small_explore():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion)
    common_setting()
    FLAGS.eval_freq = -1
    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000
    FLAGS.inenv_finetune = True
    FLAGS.exploration_schedule = PiecewiseSchedule(
        [
            (0, 0.01),
            (1e6, 0.01),
            (1e7, 0.01),
        ], outside_value=0.01
    )

    FLAGS.ckpt_path = '/data/hxu/cs294-112/hw3/link_data/cross_entropy_T_300000_0.0_dm/'
    FLAGS.core_num = '1'
    FLAGS.hard_Q_loss_weight = 1.0

def cross_entropy_dm_finetune_normal_explore():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion)
    common_setting()
    FLAGS.eval_freq = -1
    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000
    FLAGS.inenv_finetune = True


    FLAGS.ckpt_path = '/data/hxu/cs294-112/hw3/link_data/cross_entropy_T_300000_0.0_dm/'
    FLAGS.core_num = '2'
    FLAGS.hard_Q_loss_weight = 1.0




def policy_gradient_soft_1_step_finetune_small_explore_May16_redo_normal_explore():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion)
    common_setting()
    FLAGS.eval_freq = -1
    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000
    FLAGS.inenv_finetune = True
    FLAGS.force_original_exploration = True
    FLAGS.exploration_schedule = PiecewiseSchedule(
        [
            (0, 1),
            (1e6, 0.01),
            (1e7, 0.01),
        ], outside_value=0.01
    )

    FLAGS.ckpt_path = '/data/yang/code/rl_demonstration/hw3/link_data/newdata_redo_policy_gradient_soft_1_step_False_0.1'
    FLAGS.policy_gradient_soft_1_step = 1.0
    FLAGS.core_num = '0'

def policy_gradient_soft_1_step_finetune_normal_explore_May16():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion)
    common_setting()
    FLAGS.eval_freq = -1
    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000
    FLAGS.inenv_finetune = True
    FLAGS.force_original_exploration = True

    FLAGS.ckpt_path = '/data/yang/code/rl_demonstration/hw3/link_data/newdata_redo_policy_gradient_soft_1_step_False_0.1'
    FLAGS.policy_gradient_soft_1_step = 1.0
    FLAGS.core_num = '1'

def adv_learn_finetune_small_explore_May16():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion)
    common_setting()
    FLAGS.eval_freq = -1
    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.inenv_finetune = True
    FLAGS.learning_starts = 50000
    FLAGS.force_original_exploration = True
    FLAGS.exploration_schedule = PiecewiseSchedule(
        [
            (0, 0.01),
            (1e6, 0.01),
            (1e7, 0.01),
        ], outside_value=0.01
    )


    FLAGS.ckpt_path = '/data/yang/code/rl_demonstration/hw3/link_data/newdata_exp_advantage_diff_learning_False_0.1'
    FLAGS.exp_advantage_diff_learning = 1.0
    FLAGS.core_num = '2'

def adv_learn_finetune_normal_explore_May16():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion) + '_plot'
    common_setting()
    FLAGS.eval_freq = -1
    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000
    FLAGS.inenv_finetune = True
    FLAGS.force_original_exploration = True

    FLAGS.ckpt_path = '/data/yang/code/rl_demonstration/hw3/link_data/newdata_exp_advantage_diff_learning_False_0.1'
    FLAGS.exp_advantage_diff_learning = 1.0
    FLAGS.core_num = '3'

def exp_policy_grad_weighting_finetune_normal_explore_May16():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion) + '_plot'
    common_setting()
    FLAGS.eval_freq = -1
    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000
    FLAGS.inenv_finetune = True
    FLAGS.force_original_exploration = True

    FLAGS.ckpt_path = '/data/yang/code/rl_demonstration/hw3/link_data/newdata_redo_exp_policy_grad_weighting_False_0.1'
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.core_num = '0'

def exp_policy_grad_weighting_finetune_small_explore_May16():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion) + '_plot'
    common_setting()
    FLAGS.eval_freq = -1
    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000
    FLAGS.inenv_finetune = True
    FLAGS.force_original_exploration = True
    FLAGS.exploration_schedule = PiecewiseSchedule(
        [
            (0, 0.01),
            (1e6, 0.01),
            (1e7, 0.01),
        ], outside_value=0.01
    )

    FLAGS.ckpt_path = '/data/yang/code/rl_demonstration/hw3/link_data/newdata_redo_exp_policy_grad_weighting_False_0.1'
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.core_num = '1'
#________________________________from_here_to_visualize_factor_____________________
def exp_advantage_diff_learning_visualize():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion)
    common_setting()
    #FLAGS.group_name = 'rl'

    FLAGS.exp_advantage_diff_learning = 1.0
    FLAGS.core_num = '2'

def exp_policy_gradient_visualize():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion)
    common_setting()
    #FLAGS.group_name = 'rl'

    FLAGS.policy_gradient_soft_1_step = 1.0
    FLAGS.core_num = '3'

def DQfD_no_l2_finetune_small_explore_human_May16():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion) + '_constant_e'
    common_setting()
    FLAGS.eval_freq = -1
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000

    FLAGS.ckpt_path = '/data/hxu/cs294-112/hw3/link_data/DQFD_no_l2_T_300000_0.0_dm/'
    FLAGS.inenv_finetune = True
    FLAGS.demo_mode = "dqfd"
    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.core_num = '2'
    FLAGS.exploration_schedule = PiecewiseSchedule(
        [
            (0, 0.01),
            (1e6, 0.01),
            (1e7, 0.01),
        ], outside_value=0.01
    )

def DQfD_no_l2_finetune_normal_explore_human_May16():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion) + '_constant_e'
    common_setting()
    FLAGS.eval_freq = -1
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000

    FLAGS.ckpt_path = '/data/hxu/cs294-112/hw3/link_data/DQFD_no_l2_T_300000_0.0_dm/'
    FLAGS.inenv_finetune = True
    FLAGS.demo_mode = "dqfd"
    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.core_num = '3'

def collect_demonstration_onpolicy():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag + '_badstarts_' + str(FLAGS.bad_starts) + '_badexplore_' + str(FLAGS.tiny_explore)
    common_setting()

    #FLAGS.core_num = '1'
    FLAGS.learning_starts = 0
    FLAGS.learning_stage = False
    FLAGS.ckpt_path = '/data/hxu/cs294-112/hw3/data/'
    

    FLAGS.exploration_schedule = PiecewiseSchedule(
        [
            (0, 0.01),
            (FLAGS.bad_starts, 0.01),
            (FLAGS.bad_starts+1, FLAGS.tiny_explore),
            (3e5, FLAGS.tiny_explore),
        ], outside_value=0.01
    )
    #FLAGS.soft_Q_alpha = 0.01
    #FLAGS.m_bad = 300
    #FLAGS.m_good = 300
    #FLAGS.summary_interval = 10

def hard_Q_onpolicy_data():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    common_setting()
    #FLAGS.group_name = 'rl'
    #FLAGS.demo_file_path = '/data/hxu/cs294-112/hw3/link_data/bad_demo_50000.0_0.7'
    FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion) +'_'+ FLAGS.demo_file_path[-12:]

    FLAGS.core_num = '0'
    FLAGS.hard_Q_loss_weight = 1.0
    # optionally set: lr_schedule, exploration_schedule

def dueling_net_double_Q():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag + '_dueling_30min'
    common_setting()
    FLAGS.core_num = '0'

    # Q learning specific
    FLAGS.eval_freq = 10000
    FLAGS.demo_mode = "no_demo"
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.collect_Q_experience = True
    FLAGS.dueling = True
    FLAGS.learning_starts = 50000
    #FLAGS.learning_rate = 5e-5
    FLAGS.ddqn = True

def dueling_net_double_Q_eval():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag
    common_setting()
    FLAGS.core_num = '1'

    # Q learning specific
    FLAGS.eval_freq = 10000
    FLAGS.demo_mode = "no_demo"
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.collect_Q_experience = False
    FLAGS.dueling = True
    FLAGS.learning_starts = 0
    # FLAGS.learning_rate = 5e-5
    FLAGS.ddqn = True
    FLAGS.inenv_eval = True
    FLAGS.ckpt_path = '/data/hxu/cs294-112/hw3/link_data/dueling_net_double_Q_dueling/'

def urex_multistep():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag
    common_setting()
    FLAGS.core_num = '7'

    # Q learning specific
    FLAGS.multistep = True
    FLAGS.multistep_urex = True
    FLAGS.multistep_replay = False


def yang_common_setting(tag=None):

    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000
    FLAGS.tiny_explore = 0.01
    FLAGS.learning_starts = 0
    print(FLAGS.demo_file_path)

    if tag is not None:
        print("config name = ", tag)
        FLAGS.method_name = FLAGS.tag_prefix + tag + '_' + str(FLAGS.torcs_divider) +'_' + str(datetime.now())

def torcs_dqn_kx(divider, tag):
    yang_common_setting(tag)
    FLAGS.torcs_demo = True
    FLAGS.core_num = '0'

    # Q learning specific
    FLAGS.eval_freq = -1
    FLAGS.demo_mode = "no_demo"


    FLAGS.collect_Q_experience = True
    FLAGS.env_id="rltorcs-v0"

    # begin the divider attempt
    num_iterations = int(4e7) / 4 / divider
    FLAGS.learning_starts = 0#50000 / divider
    FLAGS.target_update_freq = 10000 / divider
    FLAGS.replay_buffer_size = int(1e6)
    FLAGS.max_timesteps = 2 * int(4e7) / divider
    FLAGS.dataset_size = int(3e5)

    FLAGS.lr_schedule = PiecewiseSchedule([
            (0, 1e-4),
            (num_iterations / 10, 1e-4),
            (num_iterations / 2, 5e-5)],
        outside_value=5e-5)

    FLAGS.exploration_schedule = PiecewiseSchedule([
            (0, 1.0),
            (1e6 / divider, 0.1),
            (num_iterations / 2, 0.01)],
        outside_value=0.01)
    # interaction purpose
    FLAGS.summary_interval = 10000 / divider

def torcs_collect(divider, tag):
    yang_common_setting(tag)
    FLAGS.core_num = '0'
    FLAGS.eval_freq = -1
    FLAGS.env_id = "rltorcs-v0"
    # param for demonstration
    FLAGS.learning_stage = False
    FLAGS.ckpt_path = '/data/yang/code/rl_demonstration/hw3/link_data/torcs_dqn_30x_False_0.1'
    FLAGS.replay_buffer_size = int(3e5)
    FLAGS.max_timesteps = int(3e5)/4#int(4e6) / 4
    num_iterations = int(4e7) / 4 / divider
    FLAGS.exploration_schedule = PiecewiseSchedule([
        (0, 0.1),
        (1e6 / divider, 0.1),
        (num_iterations / 2, 0.1)],
        outside_value=0.1)


def torcs_10x_BenReward():
    tag = inspect.stack()[0][3]
    torcs_dqn_kx(10, tag)
    FLAGS.custom_reward = "reward_ben"

def collect_torcs_demonstration_3e5():
    tag = inspect.stack()[0][3]
    FLAGS.torcs_divider = 1
    print(FLAGS.torcs_divider,'x torcs demo')
    torcs_collect(FLAGS.torcs_divider, tag)
    FLAGS.custom_reward = "reward_ben"
    FLAGS.core_num = '0'

def torcs_cross_entropy_demo():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider) +'_'+str(datetime.now())
    common_setting()
    FLAGS.demo_file_path = '/data/hxu/cs294-112/hw3/link_data/300000_torcs_0.1explore.p'
    torcs_dqn_kx(FLAGS.torcs_divider,tag)

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '1'
    FLAGS.supervise_cross_entropy_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)

def torcs_hinge_dqfd_demo():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/data/hxu/cs294-112/hw3/link_data/300000_torcs_0.1explore.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '0'
    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    common_data_size(300000)

def torcs_hard_Q_demo():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/data/hxu/cs294-112/hw3/link_data/300000_torcs_0.1explore.p'
    torcs_dqn_kx(FLAGS.torcs_divider,tag)

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '1'
    FLAGS.hard_Q_loss_weight = 1.0
    common_data_size(300000)

def torcs_soft_Q_demo():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/data/hxu/cs294-112/hw3/link_data/300000_torcs_0.1explore.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '2'
    FLAGS.soft_Q_loss_weight = 1.0
    common_data_size(300000)

def torcs_dqfd_full_demo():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/data/hxu/cs294-112/hw3/link_data/300000_torcs_0.1explore.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '3'

    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.l2_regularization_loss_weight = 1.0e-5
    common_data_size(300000)

def torcs_V_grounding_demo():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/data/hxu/cs294-112/hw3/link_data/300000_torcs_0.1explore.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '4'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False
    common_data_size(300000)

def torcs_V_grounding_no_weighting_demo():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/data/hxu/cs294-112/hw3/link_data/300000_torcs_0.1explore.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '5'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False

    FLAGS.disable_off_policy_weighting = True
    common_data_size(300000)

def torcs_PG_demo():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/data/hxu/cs294-112/hw3/link_data/300000_torcs_0.1explore.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '6'

    #FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False

    #FLAGS.disable_off_policy_weighting = True
    common_data_size(300000)

def torcs_PG_no_weighting_demo():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/data/hxu/cs294-112/hw3/link_data/300000_torcs_0.1explore.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '6'

    #FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False

    FLAGS.disable_off_policy_weighting = True
    common_data_size(300000)

def torcs_only_V_no_weighting_demo():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/data/hxu/cs294-112/hw3/link_data/300000_torcs_0.1explore.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '2'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 0.0#1.0
    FLAGS.critic_use_rapid_weighting = False

    FLAGS.disable_off_policy_weighting = True
    common_data_size(300000)

def torcs_Q_grounding_no_weighting_demo():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/data/hxu/cs294-112/hw3/link_data/300000_torcs_0.1explore.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '3'

    FLAGS.exp_value_critic_weighting = 0.0#1.0
    FLAGS.soft_Q_loss_weight = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False
    FLAGS.soft_Q_alpha = 0.1

    FLAGS.disable_off_policy_weighting = True
    common_data_size(300000)



def torcs_PG_Vloss_rapidWeighting_demo():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/data/hxu/cs294-112/hw3/link_data/300000_torcs.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000
    FLAGS.core_num = '4'

    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = True

def collect_demonstration_enduro():
    tag = inspect.stack()[0][3]
    yang_common_setting(tag)
    print("config name=", tag)

    common_setting()

    FLAGS.core_num = '0'
    FLAGS.demo_mode = 'no_demo'
    FLAGS.learning_starts = 0
    FLAGS.learning_stage = False
    FLAGS.ckpt_path = '/data/hxu/cs294-112/hw3/data/'
    #FLAGS.tiny_explore = 0.1
    FLAGS.soft_Q_alpha = 0.1
    FLAGS.replay_buffer_size = int(3e5)
    FLAGS.max_timesteps = int(3e5) / 4
    FLAGS.demo_name = 'enduro_perfect'
    FLAGS.exploration_schedule = PiecewiseSchedule(
        [
            (0, 0.01),
            (1e6, 0.01),
            (1e7, 0.01),
        ], outside_value=0.01
    )
    #FLAGS.summary_interval = 10
def V_grounding_no_weighting_T():
    FLAGS.tag_prefix = 'Q_expert_'
    tag = inspect.stack()[0][3]
    yang_common_setting(tag)
    #FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion) + '_dm'
    common_setting()
    #FLAGS.group_name = 'rl'

    FLAGS.core_num = '0'
    #FLAGS.hard_Q_loss_weight = 1.0
    # optionally set: lr_schedule, exploration_schedule
    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False

    FLAGS.disable_off_policy_weighting = True
    #common_data_size(300000)

def V_grounding_T():
    tag = inspect.stack()[0][3]
    yang_common_setting(tag)
    # FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion) + '_dm'
    common_setting()
    # FLAGS.group_name = 'rl'

    FLAGS.core_num = '1'
    #FLAGS.hard_Q_loss_weight = 1.0
    # optionally set: lr_schedule, exploration_schedule
    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False

    FLAGS.disable_off_policy_weighting = False
    #common_data_size(300000)

def PG_no_weighting_T():
    FLAGS.tag_prefix = 'Q_expert_'
    tag = inspect.stack()[0][3]
    yang_common_setting(tag)
    # FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion) + '_dm'
    common_setting()
    # FLAGS.group_name = 'rl'

    FLAGS.core_num = '2'
    #FLAGS.hard_Q_loss_weight = 1.0
    # optionally set: lr_schedule, exploration_schedule
    FLAGS.exp_value_critic_weighting = 0.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False

    FLAGS.disable_off_policy_weighting = True
    #common_data_size(300000)

def PG_T():
    tag = inspect.stack()[0][3]
    yang_common_setting(tag)
    # FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion) + '_dm'
    common_setting()
    # FLAGS.group_name = 'rl'

    FLAGS.core_num = '3'
    #FLAGS.hard_Q_loss_weight = 1.0
    # optionally set: lr_schedule, exploration_schedule
    FLAGS.exp_value_critic_weighting = 0.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False

    FLAGS.disable_off_policy_weighting = False
    #common_data_size(300000)

def torcs_V_grounding_inenv():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/data/hxu/cs294-112/hw3/link_data/300000_torcs_0.1explore.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000/FLAGS.torcs_divider
    FLAGS.eval_freq = -1
    FLAGS.torcs_demo = False
    FLAGS.core_num = '5'
    FLAGS.replay_buffer_size = int(3e5)
    FLAGS.exploration_schedule = PiecewiseSchedule(
        [
            (0, 0.01),
            (1e6, 0.01),
            (1e7, 0.01),
        ], outside_value=0.01
    )
    FLAGS.ckpt_path='/data/hxu/cs294-112/hw3/link_data/torcs_V_grounding_no_weighting_demo_30_2017-07-05 21:24:35.375801'
    FLAGS.inenv_finetune = True

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False

    FLAGS.disable_off_policy_weighting = True

    FLAGS.custom_reward = "reward_ben"

def torcs_DQFD_inenv_stage_2_simple(key):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()

    torcs_dqn_kx(FLAGS.torcs_divider, tag)
    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    #FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000 / FLAGS.torcs_divider
    FLAGS.eval_freq = -1
    FLAGS.torcs_demo = False
    FLAGS.core_num = '2'
    FLAGS.replay_buffer_size = int(3e5)
    FLAGS.exploration_schedule = PiecewiseSchedule(
        [
            (0, 0.01),
            (1e6, 0.01),
            (1e7, 0.01),
        ], outside_value=0.01
    )
    FLAGS.lr_schedule = PiecewiseSchedule([
        (0, 5e-5),
        (1e6 / 10, 5e-5),
        (1e7 / 2, 5e-5)],
        outside_value=5e-5)
    FLAGS.ckpt_path = ckpt_list_bad[key]#'/data/hxu/cs294-112/hw3/link_data/torcs_dqfd_full_demo_30_2017-07-05 21:24:15.244307'
    FLAGS.inenv_finetune = True

    # FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.l2_regularization_loss_weight = 1.0e-5

    FLAGS.custom_reward = "reward_ben"

    FLAGS.method_name = FLAGS.method_name + '_' + key
    FLAGS.demo_mode = 'dqfd'
    FLAGS.demo_portion = 0.1
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000autoback_simple_bad_08_25.p'
    FLAGS.autoback = True

def torcs_DQFD_inenv_no_hinge():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/data/hxu/cs294-112/hw3/link_data/300000_torcs_0.1explore.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000 / FLAGS.torcs_divider
    FLAGS.eval_freq = -1
    FLAGS.torcs_demo = False
    FLAGS.core_num = '3'
    FLAGS.replay_buffer_size = int(3e5)
    FLAGS.exploration_schedule = PiecewiseSchedule(
        [
            (0, 0.01),
            (1e6, 0.01),
            (1e7, 0.01),
        ], outside_value=0.01
    )
    FLAGS.ckpt_path = '/data/hxu/cs294-112/hw3/link_data/torcs_dqfd_full_demo_30_2017-07-05 21:24:15.244307'
    FLAGS.inenv_finetune = True

    #FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.l2_regularization_loss_weight = 1.0e-5
    
    FLAGS.custom_reward = "reward_ben"

# _____________damage_test_config________________
def collect_torcs_demonstration_damage_test():
    tag = inspect.stack()[0][3]
    FLAGS.torcs_divider = 1
    print(FLAGS.torcs_divider,'x torcs demo')
    torcs_collect(FLAGS.torcs_divider, tag)
    FLAGS.custom_reward = "reward_ben"
    FLAGS.core_num = '0'
# _____________use_for_write_up__________________________
def collect_torcs_demonstration_add_damage():
    tag = inspect.stack()[0][3]
    FLAGS.torcs_divider = 1
    print(FLAGS.torcs_divider,'x torcs demo')
    torcs_collect(FLAGS.torcs_divider, tag)
    FLAGS.custom_reward = "reward_ben"
    FLAGS.core_num = '0'
    FLAGS.demo_name = 'add_damage'
    num_iterations  = int(4e7) / 4 / FLAGS.torcs_divider
    FLAGS.exploration_schedule = PiecewiseSchedule([
        (0, 0.01),
        (1e6 / FLAGS.torcs_divider, 0.01),
        (num_iterations / 2, 0.01)],
        outside_value=0.01)
def torcs_cross_entropy_demo_stage_1():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider) +'_'+str(datetime.now())
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000add_damage.p'
    torcs_dqn_kx(FLAGS.torcs_divider,tag)

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '1'
    FLAGS.supervise_cross_entropy_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)

def torcs_hard_Q_demo_stage_1():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000add_damage.p'
    torcs_dqn_kx(FLAGS.torcs_divider,tag)

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '2'
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)

def torcs_dqfd_full_demo_stage_1():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000add_damage.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '3'

    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.l2_regularization_loss_weight = 1.0e-5
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)

def torcs_V_grounding_demo_stage_1():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000add_damage.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '0'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)

def torcs_V_grounding_no_weighting_demo_stage_1():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000add_damage.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '0'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False

    FLAGS.disable_off_policy_weighting = True
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)

# _____________finetune_model_writeup_use______________________
''' possible path
    /backup/hxu/modelRL/torcs_dqfd_full_demo_stage_1_30_2017-08-04 11:39:00.926601_300000
    /backup/hxu/modelRL/torcs_V_grounding_no_weighting_demo_stage_1_30_2017-08-04 12:11:08.140152_300000
    /backup/hxu/modelRL/torcs_V_grounding_demo_stage_1_30_2017-08-04 12:13:21.937080_300000
    /backup/hxu/modelRL/torcs_cross_entropy_demo_stage_1_30_2017-08-04 11:10:35.839296_300000
    /backup/hxu/modelRL/torcs_hard_Q_demo_stage_1_30_2017-08-04 11:33:57.696110_300000
'''
ckpt_list = {'dqfd':'/backup/hxu/modelRL/torcs_dqfd_full_demo_stage_1_30_2017-08-04 11:39:00.926601_300000',
             'V_grounding_no_weighting':'/backup/hxu/modelRL/torcs_V_grounding_no_weighting_demo_stage_1_30_2017-08-04 12:11:08.140152_300000',
             'V_grounding':'/backup/hxu/modelRL/torcs_V_grounding_demo_stage_1_30_2017-08-04 12:13:21.937080_300000',
             'cross_entropy':'/backup/hxu/modelRL/torcs_cross_entropy_demo_stage_1_30_2017-08-04 11:10:35.839296_300000',
             'hard_Q':'/backup/hxu/modelRL/torcs_hard_Q_demo_stage_1_30_2017-08-04 11:33:57.696110_300000',
             'soft_Q':'/backup/hxu/modelRL/torcs_soft_Q_simple_demo_stage_1_autoback_30_2017-08-30 14:19:07.869744_300000'}
def torcs_V_grounding_inenv_stage_2(key):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = ''
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000/FLAGS.torcs_divider
    FLAGS.eval_freq = -1
    FLAGS.torcs_demo = False
    FLAGS.core_num = '2'
    FLAGS.replay_buffer_size = int(3e5)
    FLAGS.exploration_schedule = PiecewiseSchedule(
        [
            (0, 0.01),
            (1e6, 0.01),
            (1e7, 0.01),
        ], outside_value=0.01
    )
    FLAGS.ckpt_path=ckpt_list[key]#'/backup/hxu/modelRL/torcs_dqfd_full_demo_stage_1_30_2017-08-04 11:39:00.926601_300000'
    FLAGS.inenv_finetune = True

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False

    FLAGS.disable_off_policy_weighting = True

    FLAGS.custom_reward = "reward_ben"

    FLAGS.method_name = FLAGS.method_name + '_' + key

def torcs_DQFD_inenv_stage_2(key):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()

    torcs_dqn_kx(FLAGS.torcs_divider, tag)
    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    #FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000 / FLAGS.torcs_divider
    FLAGS.eval_freq = -1
    FLAGS.torcs_demo = False
    FLAGS.core_num = '1'
    FLAGS.replay_buffer_size = int(3e5)
    FLAGS.exploration_schedule = PiecewiseSchedule(
        [
            (0, 0.01),
            (1e6, 0.01),
            (1e7, 0.01),
        ], outside_value=0.01
    )
    FLAGS.lr_schedule = PiecewiseSchedule([
        (0, 5e-5),
        (1e6 / 10, 5e-5),
        (1e7 / 2, 5e-5)],
        outside_value=5e-5)
    FLAGS.ckpt_path = ckpt_list_bad[key]#'/data/hxu/cs294-112/hw3/link_data/torcs_dqfd_full_demo_30_2017-07-05 21:24:15.244307'
    FLAGS.inenv_finetune = True

    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.l2_regularization_loss_weight = 1.0e-5
    #FLAGS.supervise_hinge_DQfD_loss_weight = 1.0

    FLAGS.custom_reward = "reward_ben"

    FLAGS.method_name = FLAGS.method_name + '_' + key
    FLAGS.demo_mode = 'dqfd'
    FLAGS.demo_portion = 0.1
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000add_damage.p'

    #FLAGS.autoback = True


def torcs_DQFD_inenv_no_hinge_stage_2(key):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = ''
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000 / FLAGS.torcs_divider
    FLAGS.eval_freq = -1
    FLAGS.torcs_demo = False
    FLAGS.core_num = '1'
    FLAGS.replay_buffer_size = int(3e5)
    FLAGS.exploration_schedule = PiecewiseSchedule(
        [
            (0, 0.01),
            (1e6, 0.01),
            (1e7, 0.01),
        ], outside_value=0.01
    )
    FLAGS.ckpt_path = ckpt_list[key]#'/data/hxu/cs294-112/hw3/link_data/torcs_dqfd_full_demo_30_2017-07-05 21:24:15.244307'
    FLAGS.inenv_finetune = True

    # FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.l2_regularization_loss_weight = 1.0e-5

    FLAGS.custom_reward = "reward_ben"

    FLAGS.method_name = FLAGS.method_name + '_' + key

def torcs_V_grounding_inenv_stage_2_large_replay(key):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 6
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = ''
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000/FLAGS.torcs_divider
    FLAGS.eval_freq = -1
    FLAGS.torcs_demo = False
    FLAGS.core_num = '0'
    FLAGS.replay_buffer_size = int(3e5)
    FLAGS.exploration_schedule = PiecewiseSchedule(
        [
            (0, 0.01),
            (1e6, 0.01),
            (1e7, 0.01),
        ], outside_value=0.01
    )
    FLAGS.lr_schedule = PiecewiseSchedule([
        (0, 5e-5),
        (1e6, 5e-5),
        (1e7, 5e-5)],
        outside_value=5e-5)
    FLAGS.ckpt_path=ckpt_list[key]#'/backup/hxu/modelRL/torcs_dqfd_full_demo_stage_1_30_2017-08-04 11:39:00.926601_300000'
    FLAGS.inenv_finetune = True

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False

    FLAGS.disable_off_policy_weighting = True

    FLAGS.custom_reward = "reward_ben"

    FLAGS.method_name = FLAGS.method_name + '_' + key


def torcs_DQFD_inenv_stage_2_large_replay(key):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 6
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()

    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    #FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000 / FLAGS.torcs_divider
    FLAGS.eval_freq = -1
    FLAGS.torcs_demo = False
    FLAGS.core_num = '0'
    FLAGS.replay_buffer_size = int(3e5)
    FLAGS.exploration_schedule = PiecewiseSchedule(
        [
            (0, 0.01),
            (1e6, 0.01),
            (1e7, 0.01),
        ], outside_value=0.01
    )
    FLAGS.lr_schedule = PiecewiseSchedule([
        (0, 5e-5),
        (1e6 / 10, 5e-5),
        (1e7 / 2, 5e-5)],
        outside_value=5e-5)
    FLAGS.ckpt_path = ckpt_list[key]#'/data/hxu/cs294-112/hw3/link_data/torcs_dqfd_full_demo_30_2017-07-05 21:24:15.244307'
    FLAGS.inenv_finetune = True

    # FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.l2_regularization_loss_weight = 1.0e-5

    FLAGS.custom_reward = "reward_ben"

    FLAGS.method_name = FLAGS.method_name + '_' + key
    FLAGS.demo_mode = 'dqfd'
    FLAGS.demo_portion = 0.1
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000add_damage.p'


# exp on complex torcs env __ FLAGS.game_config_fname
def torcs_hard_Q_complex():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = ''#'/backup/hxu/modelRL/300000add_damage.p'
    torcs_dqn_kx(FLAGS.torcs_divider,tag)

    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.torcs_demo = False
    FLAGS.eval_freq = -1
    FLAGS.learning_starts = 50000 / FLAGS.torcs_divider
    FLAGS.replay_buffer_size = int(3e5)
    FLAGS.train_from_scratch = True
    FLAGS.inenv_finetune = True
    FLAGS.core_num = '0'
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    FLAGS.game_config_fname = 'quickrace_discrete_multi_ushite-city.xml'
    #common_data_size(300000)
def collect_torcs_demonstration_complex():
    tag = inspect.stack()[0][3]
    FLAGS.torcs_divider = 1
    print(FLAGS.torcs_divider,'x torcs demo')
    torcs_collect(FLAGS.torcs_divider, tag)
    FLAGS.custom_reward = "reward_ben"
    FLAGS.core_num = '1'
    FLAGS.demo_name = 'complex'
    num_iterations  = int(4e7) / 4 / FLAGS.torcs_divider
    FLAGS.exploration_schedule = PiecewiseSchedule([
        (0, 0.01),
        (1e6 / FLAGS.torcs_divider, 0.01),
        (num_iterations / 2, 0.01)],
        outside_value=0.01)
    FLAGS.ckpt_path = '/data/hxu/modelRL/torcs_hard_Q_complex_30_2017-08-08 10:14:45.948102'
    FLAGS.game_config_fname = 'quickrace_discrete_multi_ushite-city.xml'

def torcs_cross_entropy_demo_stage_1_complex():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider) +'_'+str(datetime.now())
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000complex.p'
    torcs_dqn_kx(FLAGS.torcs_divider,tag)

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '0'
    FLAGS.supervise_cross_entropy_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.game_config_fname = 'quickrace_discrete_multi_ushite-city.xml'

def torcs_hard_Q_demo_stage_1_complex():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000complex.p'
    torcs_dqn_kx(FLAGS.torcs_divider,tag)

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '2'
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.game_config_fname = 'quickrace_discrete_multi_ushite-city.xml'

def torcs_dqfd_full_demo_stage_1_complex():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000complex.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '3'

    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.l2_regularization_loss_weight = 1.0e-5
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.game_config_fname = 'quickrace_discrete_multi_ushite-city.xml'

def torcs_V_grounding_demo_stage_1_complex():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000complex.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '0'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.game_config_fname = 'quickrace_discrete_multi_ushite-city.xml'

def torcs_V_grounding_no_weighting_demo_stage_1_complex():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000complex.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '0'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False

    FLAGS.disable_off_policy_weighting = True
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.game_config_fname = 'quickrace_discrete_multi_ushite-city.xml'
ckpt_list_complex = {'dqfd':'/backup/hxu/modelRL/torcs_dqfd_full_demo_stage_1_complex_30_2017-08-09 16:45:17.873451_300000',
             'V_grounding_no_weighting':'/backup/hxu/modelRL/torcs_V_grounding_no_weighting_demo_stage_1_complex_30_2017-08-09 16:45:37.708050_300000',
             'V_grounding':'/backup/hxu/modelRL/torcs_V_grounding_demo_stage_1_complex_30_2017-08-09 16:45:27.748450_300000',
             'cross_entropy':'/backup/hxu/modelRL/torcs_cross_entropy_demo_stage_1_complex_30_2017-08-09 16:44:58.232611_300000',
             'hard_Q':'/backup/hxu/modelRL/torcs_hard_Q_demo_stage_1_complex_30_2017-08-09 16:45:07.661204_300000'}

def torcs_V_grounding_inenv_stage_2_large_replay_complex(key):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 6
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = ''
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000/FLAGS.torcs_divider
    FLAGS.eval_freq = -1
    FLAGS.torcs_demo = False
    FLAGS.core_num = '0'
    FLAGS.replay_buffer_size = int(3e5)
    FLAGS.exploration_schedule = PiecewiseSchedule(
        [
            (0, 0.01),
            (1e6, 0.01),
            (1e7, 0.01),
        ], outside_value=0.01
    )
    FLAGS.lr_schedule = PiecewiseSchedule([
        (0, 5e-5),
        (1e6, 5e-5),
        (1e7, 5e-5)],
        outside_value=5e-5)
    FLAGS.ckpt_path=ckpt_list_complex[key]#'/backup/hxu/modelRL/torcs_dqfd_full_demo_stage_1_30_2017-08-04 11:39:00.926601_300000'
    FLAGS.inenv_finetune = True

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False

    FLAGS.disable_off_policy_weighting = True

    FLAGS.custom_reward = "reward_ben"

    FLAGS.method_name = FLAGS.method_name + '_' + key
    FLAGS.game_config_fname = 'quickrace_discrete_multi_ushite-city.xml'

def torcs_DQFD_inenv_stage_2_large_replay_complex(key):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 6
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()

    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    #FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000 / FLAGS.torcs_divider
    FLAGS.eval_freq = -1
    FLAGS.torcs_demo = False
    FLAGS.core_num = '0'
    FLAGS.replay_buffer_size = int(3e5)
    FLAGS.exploration_schedule = PiecewiseSchedule(
        [
            (0, 0.01),
            (1e6, 0.01),
            (1e7, 0.01),
        ], outside_value=0.01
    )
    FLAGS.lr_schedule = PiecewiseSchedule([
        (0, 5e-5),
        (1e6 / 10, 5e-5),
        (1e7 / 2, 5e-5)],
        outside_value=5e-5)
    FLAGS.ckpt_path = ckpt_list_complex[key]#'/data/hxu/cs294-112/hw3/link_data/torcs_dqfd_full_demo_30_2017-07-05 21:24:15.244307'
    FLAGS.inenv_finetune = True

    # FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.l2_regularization_loss_weight = 1.0e-5

    FLAGS.custom_reward = "reward_ben"

    FLAGS.method_name = FLAGS.method_name + '_' + key
    FLAGS.demo_mode = 'dqfd'
    FLAGS.demo_portion = 0.1
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000complex.p'
    FLAGS.game_config_fname = 'quickrace_discrete_multi_ushite-city.xml'

# here for slow car env
def torcs_hard_Q_slowcar():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = ''#'/backup/hxu/modelRL/300000add_damage.p'
    torcs_dqn_kx(FLAGS.torcs_divider,tag)

    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.torcs_demo = False
    FLAGS.eval_freq = -1
    FLAGS.learning_starts = 50000 / FLAGS.torcs_divider
    FLAGS.replay_buffer_size = int(3e5)
    FLAGS.train_from_scratch = True
    FLAGS.inenv_finetune = True
    FLAGS.core_num = '0'
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    FLAGS.game_config_fname = 'quickrace_discrete_slowmulti_ushite-city.xml'

def torcs_hard_Q_trial_perfect():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = ''#'/backup/hxu/modelRL/300000add_damage.p'
    torcs_dqn_kx(FLAGS.torcs_divider,tag)

    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.torcs_demo = False
    FLAGS.eval_freq = -1
    FLAGS.learning_starts = 50000 / FLAGS.torcs_divider
    FLAGS.replay_buffer_size = int(3e5)

    FLAGS.train_from_scratch = True
    #FLAGS.ckpt_path = '/data/hxu/modelRL/torcs_hard_Q_trial_perfect_30_2017-08-12 15:27:53.998032'

    FLAGS.inenv_finetune = True
    FLAGS.core_num = '0'
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    FLAGS.game_config_fname = 'quickrace_discrete_supermulti_ushite-city.xml'
    num_iterations = int(4e7) / FLAGS.torcs_divider
    FLAGS.lr_schedule = PiecewiseSchedule([
        (0, 1e-4),
        (num_iterations / 10, 1e-4),
        (num_iterations / 2, 5e-5),
        (num_iterations    , 5e-5),
        (2*num_iterations  , 0)],
        outside_value=0)

    FLAGS.exploration_schedule = PiecewiseSchedule([
        (0, 1.0),
        (1e6 / FLAGS.torcs_divider, 0.1),
        (num_iterations / 2, 0.01)],
        outside_value=0.01)


def torcs_hard_Q_trial_imperfect():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = ''#'/backup/hxu/modelRL/300000add_damage.p'
    torcs_dqn_kx(FLAGS.torcs_divider,tag)

    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.torcs_demo = False
    FLAGS.eval_freq = -1
    FLAGS.learning_starts = 50000 / FLAGS.torcs_divider
    FLAGS.replay_buffer_size = int(3e5)

    FLAGS.train_from_scratch = True#False
    #FLAGS.ckpt_path = '/data/hxu/modelRL/torcs_hard_Q_trial_imperfect_30_2017-08-12 15:28:03.788971'

    FLAGS.inenv_finetune = True
    FLAGS.core_num = '0'
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    FLAGS.game_config_fname = 'quickrace_discrete_supermulti_ushite-city.xml'
    num_iterations = int(4e7) / FLAGS.torcs_divider
    FLAGS.lr_schedule = PiecewiseSchedule([
        (0, 1e-4),
        (num_iterations / 10, 1e-4),
        (num_iterations / 2, 5e-5),
        (num_iterations, 5e-5),
        (2 * num_iterations, 0)],
        outside_value=0)

    FLAGS.exploration_schedule = PiecewiseSchedule([
        (0, 1.0),
        (1e6 / FLAGS.torcs_divider, 0.1),
        (num_iterations / 2, 0.1)],
        outside_value=0.1)

def torcs_hard_Q_trial_perfect_div_100():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 100
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = ''#'/backup/hxu/modelRL/300000add_damage.p'
    torcs_dqn_kx(FLAGS.torcs_divider,tag)

    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.torcs_demo = False
    FLAGS.eval_freq = -1
    FLAGS.learning_starts = 50000 / FLAGS.torcs_divider
    FLAGS.replay_buffer_size = int(1e4)

    FLAGS.train_from_scratch = True
    #FLAGS.ckpt_path = '/data/hxu/modelRL/torcs_hard_Q_trial_perfect_30_2017-08-12 15:27:53.998032'

    FLAGS.inenv_finetune = True
    FLAGS.core_num = '0'
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    FLAGS.game_config_fname = 'quickrace_discrete_supermulti_ushite-city.xml'
    num_iterations = int(4e7) / FLAGS.torcs_divider
    FLAGS.lr_schedule = PiecewiseSchedule([
        (0, 1e-4),
        (num_iterations / 10, 1e-4),
        (num_iterations / 2, 5e-5),
        (num_iterations, 5e-5),
        (2 * num_iterations, 0)],
        outside_value=0)

    FLAGS.exploration_schedule = PiecewiseSchedule([
        (0, 1.0),
        (1e6 / FLAGS.torcs_divider, 0.1),
        (num_iterations / 2, 0.01)],
        outside_value=0.01)


def torcs_hard_Q_trial_imperfect_div_100():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 100
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = ''#'/backup/hxu/modelRL/300000add_damage.p'
    torcs_dqn_kx(FLAGS.torcs_divider,tag)

    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.torcs_demo = False
    FLAGS.eval_freq = -1
    FLAGS.learning_starts = 50000 / FLAGS.torcs_divider
    FLAGS.replay_buffer_size = int(1e4)

    FLAGS.train_from_scratch = True#False
    #FLAGS.ckpt_path = '/data/hxu/modelRL/torcs_hard_Q_trial_imperfect_30_2017-08-12 15:28:03.788971'

    FLAGS.inenv_finetune = True
    FLAGS.core_num = '0'
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    FLAGS.game_config_fname = 'quickrace_discrete_supermulti_ushite-city.xml'
    num_iterations = int(4e7) / FLAGS.torcs_divider
    FLAGS.lr_schedule = PiecewiseSchedule([
        (0, 1e-4),
        (num_iterations / 10, 1e-4),
        (num_iterations / 2, 5e-5),
        (num_iterations, 5e-5),
        (2 * num_iterations, 0)],
        outside_value=0)

    FLAGS.exploration_schedule = PiecewiseSchedule([
        (0, 1.0),
        (1e6 / FLAGS.torcs_divider, 0.1),
        (num_iterations / 2, 0.1)],
        outside_value=0.1)
#______hard Q for simple reward __________#
def torcs_hard_Q_simple_reward_addconst():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = ''#'/backup/hxu/modelRL/300000add_damage.p'
    torcs_dqn_kx(FLAGS.torcs_divider,tag)

    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.torcs_demo = False
    FLAGS.eval_freq = -1
    FLAGS.learning_starts = 50000 / FLAGS.torcs_divider
    FLAGS.replay_buffer_size = int(3e5)

    FLAGS.train_from_scratch = True#False
    #FLAGS.ckpt_path = '/data/hxu/modelRL/torcs_hard_Q_trial_imperfect_30_2017-08-12 15:28:03.788971'

    FLAGS.inenv_finetune = True
    FLAGS.core_num = '0'
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_simple_addconst'
    FLAGS.game_config_fname = 'quickrace_discrete_single.xml'
    num_iterations = int(4e7) / FLAGS.torcs_divider
    FLAGS.lr_schedule = PiecewiseSchedule([
        (0, 1e-4),
        (num_iterations / 10, 1e-4),
        (num_iterations / 2, 5e-5),
        (num_iterations, 5e-5),
        (2 * num_iterations, 0)],
        outside_value=0)

    FLAGS.exploration_schedule = PiecewiseSchedule([
        (0, 1.0),
        (1e6 / FLAGS.torcs_divider, 0.1),
        (num_iterations / 2, 0.01)],
        outside_value=0.01)
def torcs_hard_Q_simple_reward_addconst_supermulti():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = ''#'/backup/hxu/modelRL/300000add_damage.p'
    torcs_dqn_kx(FLAGS.torcs_divider,tag)

    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.torcs_demo = False
    FLAGS.eval_freq = -1
    FLAGS.learning_starts = 50000 / FLAGS.torcs_divider
    FLAGS.replay_buffer_size = int(3e5)

    FLAGS.train_from_scratch = True#False
    #FLAGS.ckpt_path = '/data/hxu/modelRL/torcs_hard_Q_trial_imperfect_30_2017-08-12 15:28:03.788971'

    FLAGS.inenv_finetune = True
    FLAGS.core_num = '0'
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_simple_addconst'
    FLAGS.game_config_fname = 'quickrace_discrete_supermulti_ushite-city.xml'
    num_iterations = int(4e7) / FLAGS.torcs_divider
    FLAGS.lr_schedule = PiecewiseSchedule([
        (0, 1e-4),
        (num_iterations / 10, 1e-4),
        (num_iterations / 2, 5e-5),
        (num_iterations, 5e-5),
        (2 * num_iterations, 0)],
        outside_value=0)

    FLAGS.exploration_schedule = PiecewiseSchedule([
        (0, 1.0),
        (1e6 / FLAGS.torcs_divider, 0.1),
        (num_iterations / 2, 0.01)],
        outside_value=0.01)

def collect_torcs_demonstration_super_multi():
    tag = inspect.stack()[0][3]
    FLAGS.torcs_divider = 1
    print(FLAGS.torcs_divider,'x torcs demo')
    torcs_collect(FLAGS.torcs_divider, tag)
    FLAGS.custom_reward = "reward_ben"
    FLAGS.core_num = '1'
    FLAGS.demo_name = 'supermulti'
    num_iterations  = int(4e7) / 4 / FLAGS.torcs_divider
    FLAGS.exploration_schedule = PiecewiseSchedule([
        (0, 0.01),
        (1e6 / FLAGS.torcs_divider, 0.01),
        (num_iterations / 2, 0.01)],
        outside_value=0.01)
    FLAGS.ckpt_path = '/backup/hxu/modelRL/torcs_hard_Q_trial_perfect_30_2017-08-15 13:13:23.773991'
    FLAGS.game_config_fname = 'quickrace_discrete_supermulti_ushite-city.xml'

'''stage 1 for super multi starts here'''

def torcs_cross_entropy_demo_stage_1_supermulti():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider) +'_'+str(datetime.now())
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000supermulti.p'
    torcs_dqn_kx(FLAGS.torcs_divider,tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '1'
    FLAGS.supervise_cross_entropy_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.game_config_fname = 'huazhe.xml'

def torcs_hard_Q_demo_stage_1_supermulti():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000supermulti.p'
    torcs_dqn_kx(FLAGS.torcs_divider,tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '2'
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.game_config_fname = 'huazhe.xml'

def torcs_dqfd_full_demo_stage_1_supermulti():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000supermulti.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '3'

    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.l2_regularization_loss_weight = 1.0e-5
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.game_config_fname = 'huazhe.xml'

def torcs_V_grounding_demo_stage_1_supermulti():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000supermulti.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '0'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.game_config_fname = 'huazhe.xml'

def torcs_V_grounding_no_weighting_demo_stage_1_supermulti():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000supermulti.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '0'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False

    FLAGS.disable_off_policy_weighting = True
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.game_config_fname = 'huazhe.xml'

'''new autoback exp'''

def torcs_hard_Q_autoback():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = ''#'/backup/hxu/modelRL/300000add_damage.p'
    torcs_dqn_kx(FLAGS.torcs_divider,tag)

    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.torcs_demo = False
    FLAGS.eval_freq = -1
    FLAGS.learning_starts = 50000 / FLAGS.torcs_divider
    FLAGS.replay_buffer_size = int(3e5)
    FLAGS.train_from_scratch = True
    FLAGS.inenv_finetune = True
    FLAGS.core_num = '0'
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    FLAGS.autoback = True
    #FLAGS.game_config_fname = 'quickrace_discrete_multi_ushite-city.xml'

ckpt_list_supermulti = {'dqfd':'/backup/hxu/modelRL/torcs_dqfd_full_demo_stage_1_supermulti_30_2017-08-23 11:58:42.559185_300000',
             'V_grounding_no_weighting':'/backup/hxu/modelRL/torcs_V_grounding_no_weighting_demo_stage_1_supermulti_30_2017-08-23 11:59:43.117298_300000',
             'V_grounding':'/backup/hxu/modelRL/torcs_V_grounding_demo_stage_1_supermulti_30_2017-08-23 11:59:13.047646_300000',
             'cross_entropy':'/backup/hxu/modelRL/torcs_cross_entropy_demo_stage_1_supermulti_30_2017-08-23 11:57:42.629095_300000',
             'hard_Q':'/backup/hxu/modelRL/torcs_hard_Q_demo_stage_1_supermulti_30_2017-08-23 11:58:12.569207_300000'}

def torcs_V_grounding_inenv_stage_2_supermulti(key):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = ''
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000/FLAGS.torcs_divider
    FLAGS.eval_freq = -1
    FLAGS.torcs_demo = False
    FLAGS.core_num = '0'
    FLAGS.replay_buffer_size = int(3e5)
    FLAGS.exploration_schedule = PiecewiseSchedule(
        [
            (0, 0.01),
            (1e6, 0.01),
            (1e7, 0.01),
        ], outside_value=0.01
    )
    FLAGS.lr_schedule = PiecewiseSchedule([
        (0, 5e-5),
        (1e6, 5e-5),
        (1e7, 5e-5)],
        outside_value=5e-5)
    FLAGS.ckpt_path=ckpt_list_supermulti[key]#'/backup/hxu/modelRL/torcs_dqfd_full_demo_stage_1_30_2017-08-04 11:39:00.926601_300000'
    FLAGS.inenv_finetune = True

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False

    FLAGS.disable_off_policy_weighting = True

    FLAGS.custom_reward = "reward_ben"

    FLAGS.method_name = FLAGS.method_name + '_' + key

    FLAGS.game_config_fname = 'huazhe.xml'


def torcs_DQFD_inenv_stage_2_supermulti(key):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()

    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    #FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000 / FLAGS.torcs_divider
    FLAGS.eval_freq = -1
    FLAGS.torcs_demo = False
    FLAGS.core_num = '0'
    FLAGS.replay_buffer_size = int(3e5)
    FLAGS.exploration_schedule = PiecewiseSchedule(
        [
            (0, 0.01),
            (1e6, 0.01),
            (1e7, 0.01),
        ], outside_value=0.01
    )
    FLAGS.lr_schedule = PiecewiseSchedule([
        (0, 5e-5),
        (1e6 / 10, 5e-5),
        (1e7 / 2, 5e-5)],
        outside_value=5e-5)
    FLAGS.ckpt_path = ckpt_list_supermulti[key]#'/data/hxu/cs294-112/hw3/link_data/torcs_dqfd_full_demo_30_2017-07-05 21:24:15.244307'
    FLAGS.inenv_finetune = True

    # FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.l2_regularization_loss_weight = 1.0e-5

    FLAGS.custom_reward = "reward_ben"

    FLAGS.method_name = FLAGS.method_name + '_' + key
    FLAGS.demo_mode = 'dqfd'
    FLAGS.demo_portion = 0.1
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000supermulti.p'
    FLAGS.game_config_fname = 'huazhe.xml'

# hard case with autoback
def torcs_hard_Q_autoback_supermulti():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = ''#'/backup/hxu/modelRL/300000add_damage.p'
    torcs_dqn_kx(FLAGS.torcs_divider,tag)

    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.torcs_demo = False
    FLAGS.eval_freq = -1
    FLAGS.learning_starts = 50000 / FLAGS.torcs_divider
    FLAGS.replay_buffer_size = int(3e5)
    FLAGS.train_from_scratch = True
    FLAGS.inenv_finetune = True
    FLAGS.core_num = '0'
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    FLAGS.autoback = True
    FLAGS.game_config_fname = 'huazhe.xml'
    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

def collect_torcs_demonstration_autoback_simple():
    tag = inspect.stack()[0][3]
    FLAGS.torcs_divider = 1
    print(FLAGS.torcs_divider,'x torcs demo')
    torcs_collect(FLAGS.torcs_divider, tag)
    FLAGS.custom_reward = "reward_ben"
    FLAGS.core_num = '1'
    FLAGS.demo_name = 'autoback_simple_08_25'
    num_iterations  = int(4e7) / 4 / FLAGS.torcs_divider
    FLAGS.exploration_schedule = PiecewiseSchedule([
        (0, 0.01),
        (1e6 / FLAGS.torcs_divider, 0.01),
        (num_iterations / 2, 0.01)],
        outside_value=0.01)
    FLAGS.ckpt_path = '/backup/hxu/modelRL/torcs_hard_Q_autoback_30_2017-08-23 12:18:57.721148'
    FLAGS.autoback = True
    #FLAGS.game_config_fname = 'quickrace_discrete_supermulti_ushite-city.xml'

def torcs_cross_entropy_demo_stage_1_autoback_simple():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider) + '_' + str(datetime.now())
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000autoback_simple_08_25.p'
    torcs_dqn_kx(FLAGS.torcs_divider,tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '1'
    FLAGS.supervise_cross_entropy_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_hard_Q_demo_stage_1_autoback_simple():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000autoback_simple_08_25.p'
    torcs_dqn_kx(FLAGS.torcs_divider,tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '2'
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_dqfd_full_demo_stage_1_autoback_simple():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000autoback_simple_08_25.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '3'

    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.l2_regularization_loss_weight = 1.0e-5
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_V_grounding_demo_stage_1_autoback_simple():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000autoback_simple_08_25.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '0'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_V_grounding_no_weighting_demo_stage_1_autoback_simple():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000autoback_simple_08_25.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '0'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False

    FLAGS.disable_off_policy_weighting = True
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    #FLAGS.game_config_fname = 'huazhe.xml'

"""human torcs starts here"""

def torcs_human_cross_entropy_demo_stage_1():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider) + '_' + str(datetime.now())
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/human_torcs/torcsThu_Aug_24_15&%22&%31_PDT_2017.h5,/backup/hxu/modelRL/human_torcs/torcsThu_Aug_24_15&%38&%12_PDT_2017.h5,/backup/hxu/modelRL/human_torcs/torcsWed_Aug_23_21&%56&%23_PDT_2017.h5,/backup/hxu/modelRL/human_torcs/torcsThu_Aug_24_15&%27&%44_PDT_2017.h5,/backup/hxu/modelRL/human_torcs/torcsThu_Aug_24_15&%43&%15_PDT_2017.h5,/backup/hxu/modelRL/human_torcs/torcsThu_Aug_24_15&%32&%52_PDT_2017.h5,/backup/hxu/modelRL/human_torcs/torcsThu_Aug_24_15&%48&%37_PDT_2017.h5'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '1'
    FLAGS.supervise_cross_entropy_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    FLAGS.game_config_fname = 'huazhe.xml'

def torcs_human_hard_Q_demo_stage_1():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/human_torcs/torcsThu_Aug_24_15&%22&%31_PDT_2017.h5,/backup/hxu/modelRL/human_torcs/torcsThu_Aug_24_15&%38&%12_PDT_2017.h5,/backup/hxu/modelRL/human_torcs/torcsWed_Aug_23_21&%56&%23_PDT_2017.h5,/backup/hxu/modelRL/human_torcs/torcsThu_Aug_24_15&%27&%44_PDT_2017.h5,/backup/hxu/modelRL/human_torcs/torcsThu_Aug_24_15&%43&%15_PDT_2017.h5,/backup/hxu/modelRL/human_torcs/torcsThu_Aug_24_15&%32&%52_PDT_2017.h5,/backup/hxu/modelRL/human_torcs/torcsThu_Aug_24_15&%48&%37_PDT_2017.h5'
    torcs_dqn_kx(FLAGS.torcs_divider,tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '2'
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    FLAGS.game_config_fname = 'huazhe.xml'

def torcs_human_dqfd_full_demo_stage_1():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/human_torcs/torcsThu_Aug_24_15&%22&%31_PDT_2017.h5,/backup/hxu/modelRL/human_torcs/torcsThu_Aug_24_15&%38&%12_PDT_2017.h5,/backup/hxu/modelRL/human_torcs/torcsWed_Aug_23_21&%56&%23_PDT_2017.h5,/backup/hxu/modelRL/human_torcs/torcsThu_Aug_24_15&%27&%44_PDT_2017.h5,/backup/hxu/modelRL/human_torcs/torcsThu_Aug_24_15&%43&%15_PDT_2017.h5,/backup/hxu/modelRL/human_torcs/torcsThu_Aug_24_15&%32&%52_PDT_2017.h5,/backup/hxu/modelRL/human_torcs/torcsThu_Aug_24_15&%48&%37_PDT_2017.h5'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '3'

    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.l2_regularization_loss_weight = 1.0e-5
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    FLAGS.game_config_fname = 'huazhe.xml'

def torcs_human_V_grounding_demo_stage_1():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/human_torcs/torcsThu_Aug_24_15&%22&%31_PDT_2017.h5,/backup/hxu/modelRL/human_torcs/torcsThu_Aug_24_15&%38&%12_PDT_2017.h5,/backup/hxu/modelRL/human_torcs/torcsWed_Aug_23_21&%56&%23_PDT_2017.h5,/backup/hxu/modelRL/human_torcs/torcsThu_Aug_24_15&%27&%44_PDT_2017.h5,/backup/hxu/modelRL/human_torcs/torcsThu_Aug_24_15&%43&%15_PDT_2017.h5,/backup/hxu/modelRL/human_torcs/torcsThu_Aug_24_15&%32&%52_PDT_2017.h5,/backup/hxu/modelRL/human_torcs/torcsThu_Aug_24_15&%48&%37_PDT_2017.h5'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '0'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    FLAGS.game_config_fname = 'huazhe.xml'

def torcs_human_V_grounding_no_weighting_demo_stage_1():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/human_torcs/torcsThu_Aug_24_15&%22&%31_PDT_2017.h5,/backup/hxu/modelRL/human_torcs/torcsThu_Aug_24_15&%38&%12_PDT_2017.h5,/backup/hxu/modelRL/human_torcs/torcsWed_Aug_23_21&%56&%23_PDT_2017.h5,/backup/hxu/modelRL/human_torcs/torcsThu_Aug_24_15&%27&%44_PDT_2017.h5,/backup/hxu/modelRL/human_torcs/torcsThu_Aug_24_15&%43&%15_PDT_2017.h5,/backup/hxu/modelRL/human_torcs/torcsThu_Aug_24_15&%32&%52_PDT_2017.h5,/backup/hxu/modelRL/human_torcs/torcsThu_Aug_24_15&%48&%37_PDT_2017.h5'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '0'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False

    FLAGS.disable_off_policy_weighting = True
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    FLAGS.game_config_fname = 'huazhe.xml'

"""bad demonstration!"""
def collect_torcs_demonstration_autoback_simple_bad():
    tag = inspect.stack()[0][3]
    FLAGS.torcs_divider = 1
    print(FLAGS.torcs_divider,'x torcs demo')
    torcs_collect(FLAGS.torcs_divider, tag)
    FLAGS.custom_reward = "reward_ben"
    FLAGS.core_num = '1'
    FLAGS.demo_name = 'autoback_simple_bad_08_25'
    num_iterations  = int(4e7) / 4 / FLAGS.torcs_divider
    FLAGS.exploration_schedule = PiecewiseSchedule([
        (0, 0.01),
        (1e6 / FLAGS.torcs_divider, 0.01),
        (num_iterations / 2, 0.01)],
        outside_value=0.01)
    FLAGS.ckpt_path = '/backup/hxu/modelRL/torcs_hard_Q_autoback_30_2017-08-23 12:18:57.721148'
    FLAGS.autoback = True
    FLAGS.lf_bad_data = True
    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

def torcs_cross_entropy_demo_stage_1_autoback_simple_bad():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider) + '_' + str(datetime.now())
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000autoback_simple_bad_08_25.p'
    torcs_dqn_kx(FLAGS.torcs_divider,tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '1'
    FLAGS.supervise_cross_entropy_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_hard_Q_demo_stage_1_autoback_simple_bad():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000autoback_simple_bad_08_25.p'
    torcs_dqn_kx(FLAGS.torcs_divider,tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '1'
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_dqfd_full_demo_stage_1_autoback_simple_bad():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000autoback_simple_bad_08_25.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '1'

    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.l2_regularization_loss_weight = 1.0e-5
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_V_grounding_demo_stage_1_autoback_simple_bad():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000autoback_simple_bad_08_25.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '1'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_V_grounding_no_weighting_demo_stage_1_autoback_simple_bad():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000autoback_simple_bad_08_25.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '1'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False

    FLAGS.disable_off_policy_weighting = True
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    #FLAGS.game_config_fname = 'huazhe.xml'

#ckpt_list_bad = {'dqfd':'/backup/hxu/modelRL/torcs_dqfd_full_demo_stage_1_autoback_simple_bad_30_2017-08-26 11:16:01.909194_300000',
#             'V_grounding_no_weighting':'/backup/hxu/modelRL/torcs_V_grounding_no_weighting_demo_stage_1_autoback_simple_bad_30_2017-08-26 13:53:34.450986_300000',
#             'V_grounding':'/backup/hxu/modelRL/torcs_V_grounding_demo_stage_1_autoback_simple_bad_30_2017-08-26 11:16:31.907168_300000',
#             'cross_entropy':'/backup/hxu/modelRL/torcs_cross_entropy_demo_stage_1_autoback_simple_bad_30_2017-08-26 11:15:02.495110_300000',
#             'hard_Q':'/backup/hxu/modelRL/torcs_hard_Q_demo_stage_1_autoback_simple_bad_30_2017-08-26 11:15:32.077936_300000'}
ckpt_list_bad = {'dqfd':'/backup/hxu/modelRL/torcs_dqfd_full_demo_stage_1_autoback_simple_bad_30_2017-08-30 20:02:52.805614_300000',
             'V_grounding_no_weighting':'/backup/hxu/modelRL/torcs_V_grounding_no_weighting_demo_stage_1_autoback_simple_bad_30_2017-08-30 20:03:51.800961_300000',
             'V_grounding':'/backup/hxu/modelRL/torcs_V_grounding_demo_stage_1_autoback_simple_bad_30_2017-08-30 20:03:21.795297_300000',
             'cross_entropy':'/backup/hxu/modelRL/torcs_cross_entropy_demo_stage_1_autoback_simple_bad_30_2017-08-30 20:01:53.304739_300000',
             'hard_Q':'/backup/hxu/modelRL/torcs_hard_Q_demo_stage_1_autoback_simple_bad_30_2017-08-30 20:02:24.182666_300000',
             'soft_Q':'/backup/hxu/modelRL/torcs_soft_Q_simple_demo_stage_1_autoback_bad_30_2017-08-30 14:19:37.882011_300000'}

def torcs_V_grounding_inenv_stage_2_simple_bad(key):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = ''
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000/FLAGS.torcs_divider
    FLAGS.eval_freq = -1
    FLAGS.torcs_demo = False
    FLAGS.core_num = '3'
    FLAGS.replay_buffer_size = int(3e5)
    FLAGS.exploration_schedule = PiecewiseSchedule(
        [
            (0, 0.01),
            (1e6, 0.01),
            (1e7, 0.01),
        ], outside_value=0.01
    )
    FLAGS.lr_schedule = PiecewiseSchedule([
        (0, 5e-5),
        (1e6 / 10, 5e-5),
        (1e7 / 2, 5e-5)],
        outside_value=5e-5)
    FLAGS.ckpt_path=ckpt_list_bad[key]#'/backup/hxu/modelRL/torcs_dqfd_full_demo_stage_1_30_2017-08-04 11:39:00.926601_300000'
    FLAGS.inenv_finetune = True

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False

    FLAGS.disable_off_policy_weighting = True

    FLAGS.custom_reward = "reward_ben"

    FLAGS.method_name = FLAGS.method_name + '_' + key
    FLAGS.autoback = True


def torcs_DQFD_inenv_stage_2_simple_bad(key):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()

    torcs_dqn_kx(FLAGS.torcs_divider, tag)
    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    #FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000 / FLAGS.torcs_divider
    FLAGS.eval_freq = -1
    FLAGS.torcs_demo = False
    FLAGS.core_num = '2'
    FLAGS.replay_buffer_size = int(3e5)
    FLAGS.exploration_schedule = PiecewiseSchedule(
        [
            (0, 0.01),
            (1e6, 0.01),
            (1e7, 0.01),
        ], outside_value=0.01
    )
    FLAGS.lr_schedule = PiecewiseSchedule([
        (0, 5e-5),
        (1e6 / 10, 5e-5),
        (1e7 / 2, 5e-5)],
        outside_value=5e-5)
    FLAGS.ckpt_path = ckpt_list_bad[key]#'/data/hxu/cs294-112/hw3/link_data/torcs_dqfd_full_demo_30_2017-07-05 21:24:15.244307'
    FLAGS.inenv_finetune = True

    # FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.l2_regularization_loss_weight = 1.0e-5

    FLAGS.custom_reward = "reward_ben"

    FLAGS.method_name = FLAGS.method_name + '_' + key
    FLAGS.demo_mode = 'dqfd'
    FLAGS.demo_portion = 0.1
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000autoback_simple_bad_08_25.p'
    FLAGS.autoback = True


"""here we want to do ablation study for data amount"""
def torcs_cross_entropy_demo_stage_1_half_amount():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider) + '_' + str(datetime.now())
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000autoback_simple_08_25.p'
    torcs_dqn_kx(FLAGS.torcs_divider,tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '1'
    FLAGS.supervise_cross_entropy_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(150000)
    FLAGS.autoback = True
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_hard_Q_demo_stage_1_half_amount():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000autoback_simple_08_25.p'
    torcs_dqn_kx(FLAGS.torcs_divider,tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '1'
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(150000)
    FLAGS.autoback = True
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_dqfd_full_demo_stage_1_half_amount():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000autoback_simple_08_25.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '1'

    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.l2_regularization_loss_weight = 1.0e-5
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(150000)
    FLAGS.autoback = True
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_V_grounding_demo_stage_1_half_amount():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000autoback_simple_08_25.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '1'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(150000)
    FLAGS.autoback = True
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_V_grounding_no_weighting_demo_stage_1_half_amount():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000autoback_simple_08_25.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '1'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False

    FLAGS.disable_off_policy_weighting = True
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(150000)
    FLAGS.autoback = True
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_cross_entropy_demo_stage_1_50000_amount():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider) + '_' + str(datetime.now())
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000autoback_simple_08_25.p'
    torcs_dqn_kx(FLAGS.torcs_divider,tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '1'
    FLAGS.supervise_cross_entropy_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(50000)
    FLAGS.autoback = True
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_hard_Q_demo_stage_1_50000_amount():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000autoback_simple_08_25.p'
    torcs_dqn_kx(FLAGS.torcs_divider,tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '1'
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(50000)
    FLAGS.autoback = True
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_dqfd_full_demo_stage_1_50000_amount():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000autoback_simple_08_25.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '1'

    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.l2_regularization_loss_weight = 1.0e-5
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(50000)
    FLAGS.autoback = True
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_V_grounding_demo_stage_1_50000_amount():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000autoback_simple_08_25.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '1'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(50000)
    FLAGS.autoback = True
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_V_grounding_no_weighting_demo_stage_1_50000_amount():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000autoback_simple_08_25.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '1'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False

    FLAGS.disable_off_policy_weighting = True
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(50000)
    FLAGS.autoback = True
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_cross_entropy_demo_stage_1_10000_amount():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider) + '_' + str(datetime.now())
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000autoback_simple_08_25.p'
    torcs_dqn_kx(FLAGS.torcs_divider,tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '1'
    FLAGS.supervise_cross_entropy_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(10000)
    FLAGS.autoback = True
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_hard_Q_demo_stage_1_10000_amount():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000autoback_simple_08_25.p'
    torcs_dqn_kx(FLAGS.torcs_divider,tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '1'
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(10000)
    FLAGS.autoback = True
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_dqfd_full_demo_stage_1_10000_amount():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000autoback_simple_08_25.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '1'

    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.l2_regularization_loss_weight = 1.0e-5
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(10000)
    FLAGS.autoback = True
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_V_grounding_demo_stage_1_10000_amount():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000autoback_simple_08_25.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '1'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(10000)
    FLAGS.autoback = True
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_V_grounding_no_weighting_demo_stage_1_10000_amount():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000autoback_simple_08_25.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '1'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False

    FLAGS.disable_off_policy_weighting = True
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(10000)
    FLAGS.autoback = True
    #FLAGS.game_config_fname = 'huazhe.xml'

""""exp for high fps on kraken :)"""

def torcs_hard_Q_autoback_kraken():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = ''#'/backup/hxu/modelRL/300000add_damage.p'
    torcs_dqn_kx(FLAGS.torcs_divider,tag)

    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.torcs_demo = False
    FLAGS.eval_freq = -1
    FLAGS.learning_starts = 50000 / FLAGS.torcs_divider
    FLAGS.replay_buffer_size = int(3e5)
    FLAGS.train_from_scratch = True
    FLAGS.inenv_finetune = True
    FLAGS.core_num = '3'
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    FLAGS.autoback = True
    FLAGS.torcs_path = '/data/hxu/torcs_fps/rlTORCS/'
    #FLAGS.game_config_fname = 'quickrace_discrete_multi_ushite-city.xml'

"""here start to get baseline for soft Q """
def torcs_soft_Q_simple_demo_stage_1_autoback():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30

    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000autoback_simple_08_25.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '1'
    FLAGS.soft_Q_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True

def torcs_soft_Q_simple_demo_stage_1_autoback_bad():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30

    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000autoback_simple_bad_08_25.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '3'
    FLAGS.soft_Q_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True

def torcs_soft_Q_simple_demo_stage_1_half_amount():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30

    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000autoback_simple_08_25.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '1'
    FLAGS.soft_Q_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(150000)
    FLAGS.autoback = True

def torcs_soft_Q_simple_demo_stage_1_50000_amount():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30

    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000autoback_simple_08_25.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '3'
    FLAGS.soft_Q_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(50000)
    FLAGS.autoback = True

def torcs_soft_Q_simple_demo_stage_1_10000_amount():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30

    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000autoback_simple_08_25.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '3'
    FLAGS.soft_Q_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(10000)
    FLAGS.autoback = True


"""exp for reducing variance"""
def torcs_V_grounding_inenv_stage_2_simple_bad_reduce_variance(key):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = ''
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000/FLAGS.torcs_divider
    FLAGS.eval_freq = -1
    FLAGS.torcs_demo = False
    FLAGS.core_num = '0'
    FLAGS.replay_buffer_size = int(3e5)
    FLAGS.exploration_schedule = PiecewiseSchedule(
        [
            (0, 0.01),
            (1e6, 0.01),
            (1e7, 0.01),
        ], outside_value=0.01
    )
    FLAGS.lr_schedule = PiecewiseSchedule([
        (0, 0),
        (1e6 / 10, 0),
        (1e7 / 2, 0)],
        outside_value=0)
    FLAGS.ckpt_path=ckpt_list_bad[key]#'/backup/hxu/modelRL/torcs_dqfd_full_demo_stage_1_30_2017-08-04 11:39:00.926601_300000'
    FLAGS.inenv_finetune = True

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False

    FLAGS.disable_off_policy_weighting = True

    FLAGS.custom_reward = "reward_ben"

    FLAGS.method_name = FLAGS.method_name + '_' + key

"""human big data torcs starts here"""
def parse_name(path_name):
    with open(path_name,'r') as f:
        content = f.readlines()
    all_name = ''
    for item in content:
        name = item.strip()
        all_name += name + ','
    all_name = all_name[0:-1]
    return all_name

def torcs_human_all_cross_entropy_demo_stage_1():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider) + '_' + str(datetime.now())
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/human_all.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '1'
    FLAGS.supervise_cross_entropy_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_human_all_hard_Q_demo_stage_1():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/human_all.txt')
    torcs_dqn_kx(FLAGS.torcs_divider,tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '2'
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_human_all_dqfd_full_demo_stage_1():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/human_all.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '3'

    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.l2_regularization_loss_weight = 1.0e-5
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_human_all_V_grounding_demo_stage_1():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/human_all.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '4'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_human_all_V_grounding_no_weighting_demo_stage_1():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/human_all.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '5'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False

    FLAGS.disable_off_policy_weighting = True
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_human_all_soft_Q_demo_stage_1():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/human_all.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '6'

    FLAGS.soft_Q_loss_weight = 1.0
    #FLAGS.exp_value_critic_weighting = 1.0
    #FLAGS.exp_policy_grad_weighting = 1.0
    #FLAGS.critic_use_rapid_weighting = False

    #FLAGS.disable_off_policy_weighting = True
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    # FLAGS.game_config_fname = 'huazhe.xml'

"""demo collected from Harry"""

def torcs_human_all_cross_entropy_demo_stage_1_harry():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider) + '_' + str(datetime.now())
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/human_harry.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '1'
    FLAGS.supervise_cross_entropy_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_human_all_hard_Q_demo_stage_1_harry():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/human_harry.txt')
    torcs_dqn_kx(FLAGS.torcs_divider,tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '2'
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_human_all_dqfd_full_demo_stage_1_harry():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/human_harry.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '3'

    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.l2_regularization_loss_weight = 1.0e-5
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_human_all_V_grounding_demo_stage_1_harry():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/human_harry.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '4'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_human_all_V_grounding_no_weighting_demo_stage_1_harry():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/human_harry.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '5'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False

    FLAGS.disable_off_policy_weighting = True
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_human_all_soft_Q_demo_stage_1_harry():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/human_harry.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '6'

    FLAGS.soft_Q_loss_weight = 1.0
    #FLAGS.exp_value_critic_weighting = 1.0
    #FLAGS.exp_policy_grad_weighting = 1.0
    #FLAGS.critic_use_rapid_weighting = False

    #FLAGS.disable_off_policy_weighting = True
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    # FLAGS.game_config_fname = 'huazhe.xml'

def torcs_human_all_hard_Q_demo_stage_1_yang():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/human_yang.txt')
    torcs_dqn_kx(FLAGS.torcs_divider,tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '2'
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_human_all_dqfd_full_demo_stage_1_yang():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/human_yang.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '3'

    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.l2_regularization_loss_weight = 1.0e-5
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_human_all_V_grounding_demo_stage_1_yang():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/human_yang.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '4'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_human_all_V_grounding_no_weighting_demo_stage_1_yang():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/human_yang.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '5'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False

    FLAGS.disable_off_policy_weighting = True
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_human_all_soft_Q_demo_stage_1_yang():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/human_yang.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '6'

    FLAGS.soft_Q_loss_weight = 1.0
    #FLAGS.exp_value_critic_weighting = 1.0
    #FLAGS.exp_policy_grad_weighting = 1.0
    #FLAGS.critic_use_rapid_weighting = False

    #FLAGS.disable_off_policy_weighting = True
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    # FLAGS.game_config_fname = 'huazhe.xml'

'''rerun large scale environment'''
def collect_torcs_demonstration_add_damage_rerun():
    tag = inspect.stack()[0][3]
    FLAGS.torcs_divider = 1
    print(FLAGS.torcs_divider,'x torcs demo')
    torcs_collect(FLAGS.torcs_divider, tag)
    FLAGS.custom_reward = "reward_ben"
    FLAGS.core_num = '0'
    FLAGS.demo_name = 'add_damage'
    num_iterations  = int(4e7) / 4 / FLAGS.torcs_divider
    FLAGS.exploration_schedule = PiecewiseSchedule([
        (0, 0.01),
        (1e6 / FLAGS.torcs_divider, 0.01),
        (num_iterations / 2, 0.01)],
        outside_value=0.01)
def torcs_cross_entropy_demo_stage_1_rerun(index):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider) +'_'+str(datetime.now())
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000add_damage.p'
    torcs_dqn_kx(FLAGS.torcs_divider,tag)

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '1'
    FLAGS.supervise_cross_entropy_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.torcs_path = '/data/yang/code/rlTORCS'
    FLAGS.method_name = tag + '_' + index

def torcs_hard_Q_demo_stage_1_rerun(index):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000add_damage.p'
    torcs_dqn_kx(FLAGS.torcs_divider,tag)

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '2'
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.torcs_path = '/data/yang/code/rlTORCS'
    FLAGS.method_name = tag + '_' + index

def torcs_dqfd_full_demo_stage_1_rerun(index):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000add_damage.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '3'

    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.l2_regularization_loss_weight = 1.0e-5
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.torcs_path = '/data/yang/code/rlTORCS'
    FLAGS.method_name = tag + '_' + index

def torcs_V_grounding_demo_stage_1_rerun(index):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000add_damage.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '4'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.torcs_path = '/data/yang/code/rlTORCS'
    FLAGS.method_name = tag + '_' + index

def torcs_V_grounding_no_weighting_demo_stage_1_rerun(index):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000add_damage.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '5'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False

    FLAGS.disable_off_policy_weighting = True
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.torcs_path = '/data/yang/code/rlTORCS'
    FLAGS.method_name = tag + '_' + index

def torcs_soft_Q_simple_demo_stage_1_rerun(index):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30

    common_setting()
    FLAGS.demo_file_path = '/data2/hxu/modelRL/300000add_damage.p' #'/backup/hxu/modelRL/300000autoback_simple_08_25.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data2/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '0'
    FLAGS.soft_Q_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    #FLAGS.autoback = True
    FLAGS.method_name = tag + '_' + index

    # added by Yang, for evaluation only. (Second evaluation trail / batch)
    FLAGS.inenv_eval = True
    FLAGS.ckpt_path = "/backup/hxu/modelRL/torcs_soft_Q_simple_demo_stage_1_rerun"
    FLAGS.eval_freq = 1
    FLAGS.torcs_path = "/data2/hxu/rlTORCS"
    # adding the val settings, put for training first
    FLAGS.val_set = True
    FLAGS.val_set_file = "/data2/hxu/modelRL/demo_large/linji_torcsTue_Oct_24_21:54:25_PDT_2017.h5"  # only for bellman error
    FLAGS.eval_only = True

''' possible path
    /backup/hxu/modelRL/torcs_dqfd_full_demo_stage_1_30_2017-08-04 11:39:00.926601_300000
    /backup/hxu/modelRL/torcs_V_grounding_no_weighting_demo_stage_1_30_2017-08-04 12:11:08.140152_300000
    /backup/hxu/modelRL/torcs_V_grounding_demo_stage_1_30_2017-08-04 12:13:21.937080_300000
    /backup/hxu/modelRL/torcs_cross_entropy_demo_stage_1_30_2017-08-04 11:10:35.839296_300000
    /backup/hxu/modelRL/torcs_hard_Q_demo_stage_1_30_2017-08-04 11:33:57.696110_300000
'''
#ckpt_list = {'dqfd':'/backup/hxu/modelRL/torcs_dqfd_full_demo_stage_1_30_2017-08-04 11:39:00.926601_300000',
#             'V_grounding_no_weighting':'/backup/hxu/modelRL/torcs_V_grounding_no_weighting_demo_stage_1_30_2017-08-04 12:11:08.140152_300000',
#             'V_grounding':'/backup/hxu/modelRL/torcs_V_grounding_demo_stage_1_30_2017-08-04 12:13:21.937080_300000',
#             'cross_entropy':'/backup/hxu/modelRL/torcs_cross_entropy_demo_stage_1_30_2017-08-04 11:10:35.839296_300000',
#             'hard_Q':'/backup/hxu/modelRL/torcs_hard_Q_demo_stage_1_30_2017-08-04 11:33:57.696110_300000',
#             'soft_Q':'/backup/hxu/modelRL/torcs_soft_Q_simple_demo_stage_1_autoback_30_2017-08-30 14:19:07.869744_300000'}
ckpt_list_rerun = {'v_grounding': '/backup/hxu/modelRL/torcs_V_grounding_demo_stage_1_rerun',
                   'v_grounding_no_weighting': '/backup/hxu/modelRL/torcs_V_grounding_no_weighting_demo_stage_1_rerun',
                   'cross_entropy': '/backup/hxu/modelRL/torcs_cross_entropy_demo_stage_1_rerun',
                   'dqfd': '/backup/hxu/modelRL/torcs_dqfd_full_demo_stage_1_rerun',
                   'hard_Q': '/backup/hxu/modelRL/torcs_hard_Q_demo_stage_1_rerun',
                   'soft_Q': '/backup/hxu/modelRL/torcs_soft_Q_simple_demo_stage_1_rerun'}
def get_rerun_list(config):
    import glob
    f = glob.glob('/backup/hxu/modelRL/torcs_' + config + '*stage_1*09-22*')
    return f


def torcs_V_grounding_has_weighting_inenv_stage_2_rerun(key):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = ''
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'
    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000 / FLAGS.torcs_divider
    FLAGS.eval_freq = -1
    FLAGS.torcs_demo = False
    FLAGS.core_num = '0'
    FLAGS.replay_buffer_size = int(3e5)
    FLAGS.exploration_schedule = PiecewiseSchedule(
        [
            (0, 0.01),
            (1e6, 0.01),
            (1e7, 0.01),
        ], outside_value=0.01
    )

    FLAGS.ckpt_path = ckpt_list_rerun['v_grounding'] + '_' + str(key)

    #FLAGS.ckpt_path = ckpt_list[int(key)]  # '/backup/hxu/modelRL/torcs_dqfd_full_demo_stage_1_30_2017-08-04 11:39:00.926601_300000'
    FLAGS.inenv_finetune = True

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False

    #FLAGS.disable_off_policy_weighting = False#True

    FLAGS.custom_reward = "reward_ben"

    FLAGS.method_name = tag + '_' + key + '_new'

def torcs_soft_Q_inenv_stage_2_rerun(key):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = ''
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000 / FLAGS.torcs_divider
    FLAGS.eval_freq = -1
    FLAGS.torcs_demo = False
    FLAGS.core_num = '1'
    FLAGS.replay_buffer_size = int(3e5)
    FLAGS.exploration_schedule = PiecewiseSchedule(
        [
            (0, 0.01),
            (1e6, 0.01),
            (1e7, 0.01),
        ], outside_value=0.01
    )

    FLAGS.ckpt_path = ckpt_list_rerun['soft_Q'] + '_' + str(key)

    FLAGS.inenv_finetune = True

    FLAGS.soft_Q_loss_weight = 1.0

    FLAGS.custom_reward = "reward_ben"

    FLAGS.method_name = tag + '_' + key + '_new'

def torcs_hard_Q_inenv_stage_2_rerun(key):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = ''
    torcs_dqn_kx(FLAGS.torcs_divider, tag)
    FLAGS.torcs_path = '/data/yang/code/rlTORCS'
    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000 / FLAGS.torcs_divider
    FLAGS.eval_freq = -1
    FLAGS.torcs_demo = False
    FLAGS.core_num = '2'
    FLAGS.replay_buffer_size = int(3e5)
    FLAGS.exploration_schedule = PiecewiseSchedule(
        [
            (0, 0.01),
            (1e6, 0.01),
            (1e7, 0.01),
        ], outside_value=0.01
    )
    FLAGS.ckpt_path = ckpt_list_rerun['hard_Q'] + '_' + str(key)

    FLAGS.inenv_finetune = True

    FLAGS.hard_Q_loss_weight = 1.0

    FLAGS.custom_reward = "reward_ben"

    FLAGS.method_name = tag + '_' + key + '_new'

def torcs_hard_Q_inenv_stage_2_with_cross_ent_rerun(key):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = ''
    torcs_dqn_kx(FLAGS.torcs_divider, tag)
    FLAGS.torcs_path = '/data/yang/code/rlTORCS'
    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000 / FLAGS.torcs_divider
    FLAGS.eval_freq = -1
    FLAGS.torcs_demo = False
    FLAGS.core_num = '0'
    FLAGS.replay_buffer_size = int(3e5)
    FLAGS.exploration_schedule = PiecewiseSchedule(
        [
            (0, 0.01),
            (1e6, 0.01),
            (1e7, 0.01),
        ], outside_value=0.01
    )
    FLAGS.ckpt_path = ckpt_list_rerun['cross_entropy'] + '_' + str(key)

    FLAGS.inenv_finetune = True

    FLAGS.hard_Q_loss_weight = 1.0

    FLAGS.custom_reward = "reward_ben"

    FLAGS.method_name = tag + '_' + key + '_new'

def torcs_V_grounding_no_weighting_inenv_stage_2_rerun(key):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = ''
    torcs_dqn_kx(FLAGS.torcs_divider, tag)
    FLAGS.torcs_path = '/data/yang/code/rlTORCS'
    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000/FLAGS.torcs_divider
    FLAGS.eval_freq = -1
    FLAGS.torcs_demo = False
    FLAGS.core_num = '1'
    FLAGS.replay_buffer_size = int(3e5)
    FLAGS.exploration_schedule = PiecewiseSchedule(
        [
            (0, 0.01),
            (1e6, 0.01),
            (1e7, 0.01),
        ], outside_value=0.01
    )
    FLAGS.ckpt_path = ckpt_list_rerun['v_grounding_no_weighting'] + '_' + str(key)
    FLAGS.inenv_finetune = True

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False

    FLAGS.disable_off_policy_weighting = True

    FLAGS.custom_reward = "reward_ben"

    FLAGS.method_name = tag + '_' + key + '_new'

def torcs_DQFD_inenv_stage_2_rerun(key):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()

    torcs_dqn_kx(FLAGS.torcs_divider, tag)
    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    #FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000 / FLAGS.torcs_divider
    FLAGS.eval_freq = -1
    FLAGS.torcs_demo = False
    FLAGS.core_num = '2'
    FLAGS.replay_buffer_size = int(3e5)
    FLAGS.exploration_schedule = PiecewiseSchedule(
        [
            (0, 0.01),
            (1e6, 0.01),
            (1e7, 0.01),
        ], outside_value=0.01
    )
    FLAGS.lr_schedule = PiecewiseSchedule([
        (0, 5e-5),
        (1e6 / 10, 5e-5),
        (1e7 / 2, 5e-5)],
        outside_value=5e-5)

    FLAGS.ckpt_path = ckpt_list_rerun['dqfd'] + '_' + str(key)


    FLAGS.inenv_finetune = True

    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.l2_regularization_loss_weight = 1.0e-5
    #FLAGS.supervise_hinge_DQfD_loss_weight = 1.0

    FLAGS.custom_reward = "reward_ben"

    FLAGS.method_name = tag + '_' + key + '_new'
    FLAGS.demo_mode = 'dqfd'
    FLAGS.demo_portion = 0.1
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000add_damage.p'

    #FLAGS.autoback = True


"""rerun yang with new data"""

def torcs_human_all_cross_entropy_demo_stage_1_yang_new():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider) + '_' + str(datetime.now())
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/human_all.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '1'
    FLAGS.supervise_cross_entropy_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    #FLAGS.game_config_fname = 'huazhe.xml'
def torcs_human_all_hard_Q_demo_stage_1_yang_new():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/human_yang.txt')
    torcs_dqn_kx(FLAGS.torcs_divider,tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '2'
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_human_all_dqfd_full_demo_stage_1_yang_new():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/human_yang.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '3'

    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.l2_regularization_loss_weight = 1.0e-5
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_human_all_V_grounding_demo_stage_1_yang_new():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/human_yang.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '4'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_human_all_V_grounding_no_weighting_demo_stage_1_yang_new():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/human_yang.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '5'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False

    FLAGS.disable_off_policy_weighting = True
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_human_all_soft_Q_demo_stage_1_yang_new():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/human_yang.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '6'

    FLAGS.soft_Q_loss_weight = 1.0
    #FLAGS.exp_value_critic_weighting = 1.0
    #FLAGS.exp_policy_grad_weighting = 1.0
    #FLAGS.critic_use_rapid_weighting = False

    #FLAGS.disable_off_policy_weighting = True
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    # FLAGS.game_config_fname = 'huazhe.xml'

def torcs_human_all_soft_Q_demo_stage_1_yang_new_PCL(index):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/data2/hxu/modelRL/300000add_damage.p'#parse_name('/data/hxu/modelRL/human_yang.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data2/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '0'
    FLAGS.soft_Q_loss_weight = 1.0
    FLAGS.pi_v_model = True
    #FLAGS.exp_value_critic_weighting = 1.0
    #FLAGS.exp_policy_grad_weighting = 1.0
    #FLAGS.critic_use_rapid_weighting = False

    #FLAGS.disable_off_policy_weighting = True
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.torcs_demo = True
    FLAGS.method_name = tag + '_' + index
    #FLAGS.human_torcs = True
    # FLAGS.game_config_fname = 'huazhe.xml'

def torcs_human_all_cross_entropy_demo_stage_1_yang_new_second():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider) + '_' + str(datetime.now())
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/human_yang_new.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '1'
    FLAGS.supervise_cross_entropy_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    #FLAGS.game_config_fname = 'huazhe.xml'
def torcs_human_all_hard_Q_demo_stage_1_yang_new_second():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/human_yang_new.txt')
    torcs_dqn_kx(FLAGS.torcs_divider,tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '2'
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_human_all_dqfd_full_demo_stage_1_yang_new_second():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/human_yang_new.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '3'

    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.l2_regularization_loss_weight = 1.0e-5
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_human_all_V_grounding_demo_stage_1_yang_new_second():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/human_yang_new.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '4'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_human_all_V_grounding_no_weighting_demo_stage_1_yang_new_second():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/human_yang_new.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '5'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False

    FLAGS.disable_off_policy_weighting = True
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_human_all_soft_Q_demo_stage_1_yang_new_second():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/human_yang_new.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '6'

    FLAGS.soft_Q_loss_weight = 1.0
    #FLAGS.exp_value_critic_weighting = 1.0
    #FLAGS.exp_policy_grad_weighting = 1.0
    #FLAGS.critic_use_rapid_weighting = False

    #FLAGS.disable_off_policy_weighting = True
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    # FLAGS.game_config_fname = 'huazhe.xml'

def torcs_human_all_cross_entropy_demo_stage_1_hxu_slow():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider) + '_' + str(datetime.now())
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/hxu_slow.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '1'
    FLAGS.supervise_cross_entropy_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    #FLAGS.game_config_fname = 'huazhe.xml'
def torcs_human_all_hard_Q_demo_stage_1_hxu_slow():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/hxu_slow.txt')
    torcs_dqn_kx(FLAGS.torcs_divider,tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '2'
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_human_all_dqfd_full_demo_stage_1_hxu_slow():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/hxu_slow.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '3'

    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.l2_regularization_loss_weight = 1.0e-5
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_human_all_V_grounding_demo_stage_1_hxu_slow():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/hxu_slow.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '4'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_human_all_V_grounding_no_weighting_demo_stage_1_hxu_slow():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/hxu_slow.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '5'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False

    FLAGS.disable_off_policy_weighting = True
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_human_all_soft_Q_demo_stage_1_hxu_slow():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/hxu_slow.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '6'

    FLAGS.soft_Q_loss_weight = 1.0
    #FLAGS.exp_value_critic_weighting = 1.0
    #FLAGS.exp_policy_grad_weighting = 1.0
    #FLAGS.critic_use_rapid_weighting = False

    #FLAGS.disable_off_policy_weighting = True
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    # FLAGS.game_config_fname = 'huazhe.xml'

"""enduro rerun with hdf!"""
def enduro_cross_entropy_rerun():
    FLAGS.tag_prefix = 'human'
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    #FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion) + '_dm'
    yang_common_setting(tag)
    common_setting()
    #FLAGS.demo_mode = 'replay'
    #FLAGS.group_name = 'supervised'
    #FLAGS.demo_file_path = '/data/hxu/cs294-112/hw3/link_data/300000enduro_perfect'
    FLAGS.core_num = '0'
    FLAGS.supervise_cross_entropy_loss_weight = 1.0
    FLAGS.method_name = tag + '2017-10-08'

def enduro_DQfD_full_rerun():
    FLAGS.tag_prefix = 'human'
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    #FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion) + '_dm'
    yang_common_setting(tag)
    common_setting()
    #FLAGS.group_name = 'dqfd'

    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.l2_regularization_loss_weight = 1.0e-5
    FLAGS.core_num = '1'
    FLAGS.method_name = tag + '2017-10-08'

def enduro_V_grounding_no_weighting_rerun():
    FLAGS.tag_prefix = 'human'
    tag = inspect.stack()[0][3]
    print("config name=", tag)

    yang_common_setting(tag)
    common_setting()

    FLAGS.core_num = '2'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False

    FLAGS.disable_off_policy_weighting = True
    FLAGS.method_name = tag + '2017-10-08'

"""weight decay"""
def torcs_human_all_cross_entropy_demo_stage_1_hxu_slow_decay(num):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider) + '_' + str(datetime.now())
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/hxu_slow.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '1'
    FLAGS.supervise_cross_entropy_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    #FLAGS.game_config_fname = 'huazhe.xml'
    #FLAGS.weight_decay = True
    #FLAGS.WEIGHT_DECAY_FACTOR = num
    FLAGS.l2_regularization_loss_weight = num
    FLAGS.method_name = tag + '_' + str(num)


def torcs_human_all_hard_Q_demo_stage_1_hxu_slow_decay(num):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/hxu_slow.txt')
    torcs_dqn_kx(FLAGS.torcs_divider,tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '2'
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    #FLAGS.game_config_fname = 'huazhe.xml'
    #FLAGS.weight_decay = True
    #FLAGS.WEIGHT_DECAY_FACTOR = num
    FLAGS.l2_regularization_loss_weight = num
    FLAGS.method_name = tag + '_' + str(num)

def torcs_human_all_dqfd_full_demo_stage_1_hxu_slow_decay(num):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/hxu_slow.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '3'

    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.l2_regularization_loss_weight = 1.0e-5
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    #FLAGS.game_config_fname = 'huazhe.xml'
    #FLAGS.weight_decay = True
    #FLAGS.WEIGHT_DECAY_FACTOR = num
    FLAGS.l2_regularization_loss_weight = num
    FLAGS.method_name = tag + '_' + str(num)

def torcs_human_all_V_grounding_demo_stage_1_hxu_slow_decay(num):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/hxu_slow.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '4'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    #FLAGS.game_config_fname = 'huazhe.xml'
    #FLAGS.weight_decay = True
    #FLAGS.WEIGHT_DECAY_FACTOR = num
    FLAGS.l2_regularization_loss_weight = num
    FLAGS.method_name = tag + '_' + str(num)

def torcs_human_all_V_grounding_no_weighting_demo_stage_1_hxu_slow_decay(num):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/hxu_slow.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '5'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False

    FLAGS.disable_off_policy_weighting = True
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    #FLAGS.game_config_fname = 'huazhe.xml'
    #FLAGS.weight_decay = True
    #FLAGS.WEIGHT_DECAY_FACTOR = num
    FLAGS.l2_regularization_loss_weight = num
    FLAGS.method_name = tag + '_' + str(num)

def torcs_human_all_soft_Q_demo_stage_1_hxu_slow_decay(num):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/hxu_slow.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '6'

    FLAGS.soft_Q_loss_weight = 1.0
    #FLAGS.exp_value_critic_weighting = 1.0
    #FLAGS.exp_policy_grad_weighting = 1.0
    #FLAGS.critic_use_rapid_weighting = False

    #FLAGS.disable_off_policy_weighting = True
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    # FLAGS.game_config_fname = 'huazhe.xml'
    #FLAGS.weight_decay = True
    #FLAGS.WEIGHT_DECAY_FACTOR = num
    FLAGS.l2_regularization_loss_weight = num
    FLAGS.method_name = tag + '_' + str(num)

def torcs_human_all_cross_entropy_demo_stage_1_hxu_slow_val():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider) + '_' + str(datetime.now())
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/hxu_slow.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '1'
    FLAGS.supervise_cross_entropy_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    #FLAGS.game_config_fname = 'huazhe.xml'
    FLAGS.val_set = True
    FLAGS.val_set_file = parse_name('/data/hxu/modelRL/val_set.txt')

def torcs_human_all_hard_Q_demo_stage_1_hxu_slow_val():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/hxu_slow.txt')
    torcs_dqn_kx(FLAGS.torcs_divider,tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '2'
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    #FLAGS.game_config_fname = 'huazhe.xml'
    FLAGS.val_set = True
    FLAGS.val_set_file = parse_name('/data/hxu/modelRL/val_set.txt')

def torcs_human_all_dqfd_full_demo_stage_1_hxu_slow_val():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/hxu_slow.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '3'

    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.l2_regularization_loss_weight = 1.0e-5
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    #FLAGS.game_config_fname = 'huazhe.xml'
    FLAGS.val_set = True
    FLAGS.val_set_file = parse_name('/data/hxu/modelRL/val_set.txt')

def torcs_human_all_V_grounding_demo_stage_1_hxu_slow_val():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/hxu_slow.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '4'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    #FLAGS.game_config_fname = 'huazhe.xml'
    FLAGS.val_set = True
    FLAGS.val_set_file = parse_name('/data/hxu/modelRL/val_set.txt')

def torcs_human_all_V_grounding_no_weighting_demo_stage_1_hxu_slow_val():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/hxu_slow.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '5'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False

    FLAGS.disable_off_policy_weighting = True
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    #FLAGS.game_config_fname = 'huazhe.xml'
    FLAGS.val_set = True
    FLAGS.val_set_file = parse_name('/data/hxu/modelRL/val_set.txt')

def torcs_human_all_soft_Q_demo_stage_1_hxu_slow_val():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/hxu_slow.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '6'

    FLAGS.soft_Q_loss_weight = 1.0
    #FLAGS.exp_value_critic_weighting = 1.0
    #FLAGS.exp_policy_grad_weighting = 1.0
    #FLAGS.critic_use_rapid_weighting = False

    #FLAGS.disable_off_policy_weighting = True
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    # FLAGS.game_config_fname = 'huazhe.xml'
    FLAGS.val_set = True
    FLAGS.val_set_file = parse_name('/data/hxu/modelRL/val_set.txt')


#### diverse behavior

def torcs_human_all_cross_entropy_demo_stage_1_hxu_slow_diverse():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider) + '_' + str(datetime.now())
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/hxu_slow_diverse.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '0'
    FLAGS.supervise_cross_entropy_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    #FLAGS.game_config_fname = 'huazhe.xml'
def torcs_human_all_hard_Q_demo_stage_1_hxu_slow_diverse():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/hxu_slow_diverse.txt')
    torcs_dqn_kx(FLAGS.torcs_divider,tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '1'
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_human_all_dqfd_full_demo_stage_1_hxu_slow_diverse():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/hxu_slow_diverse.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '2'

    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.l2_regularization_loss_weight = 1.0e-5
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_human_all_V_grounding_demo_stage_1_hxu_slow_diverse():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/hxu_slow_diverse.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '3'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_human_all_V_grounding_no_weighting_demo_stage_1_hxu_slow_diverse():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/hxu_slow_diverse.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '0'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False

    FLAGS.disable_off_policy_weighting = True
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_human_all_soft_Q_demo_stage_1_hxu_slow_diverse():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/hxu_slow_diverse.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '1'

    FLAGS.soft_Q_loss_weight = 1.0
    #FLAGS.exp_value_critic_weighting = 1.0
    #FLAGS.exp_policy_grad_weighting = 1.0
    #FLAGS.critic_use_rapid_weighting = False

    #FLAGS.disable_off_policy_weighting = True
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    # FLAGS.game_config_fname = 'huazhe.xml'

#### all behavior

def torcs_human_all_cross_entropy_demo_stage_1_hxu_slow_all():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider) + '_' + str(datetime.now())
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/hxu_slow_all.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '0'
    FLAGS.supervise_cross_entropy_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    #FLAGS.game_config_fname = 'huazhe.xml'
def torcs_human_all_hard_Q_demo_stage_1_hxu_slow_all():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/hxu_slow_all.txt')
    torcs_dqn_kx(FLAGS.torcs_divider,tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '1'
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_human_all_dqfd_full_demo_stage_1_hxu_slow_all():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/hxu_slow_all.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '2'

    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.l2_regularization_loss_weight = 1.0e-5
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_human_all_V_grounding_demo_stage_1_hxu_slow_all():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/hxu_slow_all.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '3'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_human_all_V_grounding_no_weighting_demo_stage_1_hxu_slow_all():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/hxu_slow_all.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '2'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False

    FLAGS.disable_off_policy_weighting = True
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_human_all_soft_Q_demo_stage_1_hxu_slow_all():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/hxu_slow_all.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '3'

    FLAGS.soft_Q_loss_weight = 1.0
    #FLAGS.exp_value_critic_weighting = 1.0
    #FLAGS.exp_policy_grad_weighting = 1.0
    #FLAGS.critic_use_rapid_weighting = False

    #FLAGS.disable_off_policy_weighting = True
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    # FLAGS.game_config_fname = 'huazhe.xml'

#### bad behavior
def torcs_human_all_cross_entropy_demo_stage_1_hxu_slow_badmix():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider) + '_' + str(datetime.now())
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/hxu_slow_badmix.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '0'
    FLAGS.supervise_cross_entropy_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    # FLAGS.game_config_fname = 'huazhe.xml'

def torcs_human_all_hard_Q_demo_stage_1_hxu_slow_badmix():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/hxu_slow_badmix.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '1'
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    # FLAGS.game_config_fname = 'huazhe.xml'

def torcs_human_all_dqfd_full_demo_stage_1_hxu_slow_badmix():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/hxu_slow_badmix.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '2'

    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.l2_regularization_loss_weight = 1.0e-5
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    # FLAGS.game_config_fname = 'huazhe.xml'

def torcs_human_all_V_grounding_demo_stage_1_hxu_slow_badmix():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/hxu_slow_badmix.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '3'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    # FLAGS.game_config_fname = 'huazhe.xml'

def torcs_human_all_V_grounding_no_weighting_demo_stage_1_hxu_slow_badmix():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/hxu_slow_badmix.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '2'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False

    FLAGS.disable_off_policy_weighting = True
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    # FLAGS.game_config_fname = 'huazhe.xml'

def torcs_human_all_soft_Q_demo_stage_1_hxu_slow_badmix():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/hxu_slow_badmix.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '3'

    FLAGS.soft_Q_loss_weight = 1.0
    # FLAGS.exp_value_critic_weighting = 1.0
    # FLAGS.exp_policy_grad_weighting = 1.0
    # FLAGS.critic_use_rapid_weighting = False

    # FLAGS.disable_off_policy_weighting = True
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    # FLAGS.game_config_fname = 'huazhe.xml'

#### badall behavior
def torcs_human_all_cross_entropy_demo_stage_1_hxu_slow_badall():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider) + '_' + str(datetime.now())
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/hxu_slow_badall.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '0'
    FLAGS.supervise_cross_entropy_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    # FLAGS.game_config_fname = 'huazhe.xml'

def torcs_human_all_hard_Q_demo_stage_1_hxu_slow_badall():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/hxu_slow_badall.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '1'
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    # FLAGS.game_config_fname = 'huazhe.xml'

def torcs_human_all_dqfd_full_demo_stage_1_hxu_slow_badall():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/hxu_slow_badall.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '2'

    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.l2_regularization_loss_weight = 1.0e-5
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    # FLAGS.game_config_fname = 'huazhe.xml'

def torcs_human_all_V_grounding_demo_stage_1_hxu_slow_badall():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/hxu_slow_badall.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '3'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    # FLAGS.game_config_fname = 'huazhe.xml'

def torcs_human_all_V_grounding_no_weighting_demo_stage_1_hxu_slow_badall():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/hxu_slow_badall.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '2'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False

    FLAGS.disable_off_policy_weighting = True
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    # FLAGS.game_config_fname = 'huazhe.xml'

def torcs_human_all_soft_Q_demo_stage_1_hxu_slow_badall():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/hxu_slow_badall.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '3'

    FLAGS.soft_Q_loss_weight = 1.0
    # FLAGS.exp_value_critic_weighting = 1.0
    # FLAGS.exp_policy_grad_weighting = 1.0
    # FLAGS.critic_use_rapid_weighting = False

    # FLAGS.disable_off_policy_weighting = True
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    # FLAGS.game_config_fname = 'huazhe.xml'

"""bad demonstration!"""
def collect_torcs_demonstration_autoback_simple_bad(num):
    tag = inspect.stack()[0][3]
    FLAGS.torcs_divider = 1
    print(FLAGS.torcs_divider,'x torcs demo')
    torcs_collect(FLAGS.torcs_divider, tag)
    FLAGS.custom_reward = "reward_ben"
    FLAGS.core_num = '0'

    num_iterations  = int(4e7) / 4 / FLAGS.torcs_divider
    FLAGS.exploration_schedule = PiecewiseSchedule([
        (0, 0.01),
        (1e6 / FLAGS.torcs_divider, 0.01),
        (num_iterations / 2, 0.01)],
        outside_value=0.01)
    FLAGS.ckpt_path = '/backup/hxu/modelRL/torcs_hard_Q_autoback_30_2017-08-23 12:18:57.721148'
    FLAGS.autoback = True
    FLAGS.lf_bad_data = True
    FLAGS.torcs_path = '/data/yang/code/rlTORCS'
    FLAGS.max_timesteps = int((3e5)/4)
    FLAGS.final_bad_portion = num
    FLAGS.demo_name = 'block_bad_10_21' + '_' + str(FLAGS.final_bad_portion)

def collect_torcs_demonstration_autoback_random_bad(num):
    tag = inspect.stack()[0][3]
    FLAGS.torcs_divider = 1
    print(FLAGS.torcs_divider,'x torcs demo')
    torcs_collect(FLAGS.torcs_divider, tag)
    FLAGS.custom_reward = "reward_ben"
    FLAGS.core_num = '1'

    num_iterations  = int(4e7) / 4 / FLAGS.torcs_divider
    FLAGS.exploration_schedule = PiecewiseSchedule([
        (0, 0.01),
        (1e6 / FLAGS.torcs_divider, 0.01),
        (num_iterations / 2, 0.01)],
        outside_value=0.01)
    FLAGS.ckpt_path = '/backup/hxu/modelRL/torcs_hard_Q_autoback_30_2017-08-23 12:18:57.721148'
    FLAGS.autoback = True
    FLAGS.lf_bad_data = True
    FLAGS.torcs_path = '/data/yang/code/rlTORCS'
    FLAGS.max_timesteps = int((3e5)/4)
    FLAGS.final_bad_portion = num
    FLAGS.demo_name = 'random_bad_10_21' + '_' + str(FLAGS.final_bad_portion)
    FLAGS.bad_type = 'random'

def collect_torcs_demonstration_autoback_block_bad(num):
    tag = inspect.stack()[0][3]
    FLAGS.torcs_divider = 1
    print(FLAGS.torcs_divider,'x torcs demo')
    torcs_collect(FLAGS.torcs_divider, tag)
    FLAGS.custom_reward = "reward_ben"
    FLAGS.core_num = '2'

    num_iterations  = int(4e7) / 4 / FLAGS.torcs_divider
    FLAGS.exploration_schedule = PiecewiseSchedule([
        (0, 0.01),
        (1e6 / FLAGS.torcs_divider, 0.01),
        (num_iterations / 2, 0.01)],
        outside_value=0.01)
    FLAGS.ckpt_path = '/backup/hxu/modelRL/torcs_hard_Q_autoback_30_2017-08-23 12:18:57.721148'
    FLAGS.autoback = True
    FLAGS.lf_bad_data = True
    FLAGS.torcs_path = '/data/yang/code/rlTORCS'
    FLAGS.max_timesteps = int((3e5)/4)
    FLAGS.final_bad_portion = num
    FLAGS.demo_name = 'block_bad_fix_max' + '_' + str(FLAGS.final_bad_portion)
    FLAGS.bad_type = 'block'
    FLAGS.method_name = tag + '_' + str(num)

def collect_torcs_demonstration_segment_bad(num):
    tag = inspect.stack()[0][3]
    FLAGS.torcs_divider = 1
    print(FLAGS.torcs_divider,'x torcs demo')
    torcs_collect(FLAGS.torcs_divider, tag)
    FLAGS.custom_reward = "reward_ben"
    FLAGS.core_num = '0'

    num_iterations  = int(4e7) / 4 / FLAGS.torcs_divider
    FLAGS.exploration_schedule = PiecewiseSchedule([
        (0, 0.01),
        (1e6 / FLAGS.torcs_divider, 0.01),
        (num_iterations / 2, 0.01)],
        outside_value=0.01)
    FLAGS.ckpt_path = '/backup/hxu/modelRL/torcs_hard_Q_autoback_30_2017-08-23 12:18:57.721148'
    FLAGS.autoback = True
    FLAGS.lf_bad_data = True
    FLAGS.torcs_path = '/data/yang/code/rlTORCS'
    FLAGS.max_timesteps = int((3e5)/4)
    #FLAGS.final_bad_portion = num
    FLAGS.period = 300
    FLAGS.bad_period = num
    FLAGS.demo_name = 'segment' + '_' + str(FLAGS.bad_period)
    FLAGS.bad_type = 'segment'
    FLAGS.method_name = tag + '_' + str(num)

def torcs_cross_entropy_demo_stage_1_bad(demo_prefix, num, data_num):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider) + '_' + str(datetime.now())
    common_setting()
    #demo_prefix = block_bad_fix_max_ or random_bad_10_21_
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000_'+ demo_prefix +str(num)+'.p'
    torcs_dqn_kx(FLAGS.torcs_divider,tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '0'
    FLAGS.supervise_cross_entropy_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    common_data_size(data_num)
    FLAGS.method_name = tag+ '_' + demo_prefix + '_' +str(num) + '_' + str(data_num)

    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_hard_Q_demo_stage_1_bad(demo_prefix, num, data_num):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    # demo_prefix = block_bad_fix_max_ or random_bad_10_21_
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000_'+ demo_prefix +str(num)+'.p'
    torcs_dqn_kx(FLAGS.torcs_divider,tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '1'
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    common_data_size(data_num)
    FLAGS.method_name = tag + '_' + demo_prefix + '_' + str(num) + '_' + str(data_num)
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_dqfd_full_demo_stage_1_bad(demo_prefix, num, data_num):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    # demo_prefix = block_bad_fix_max_ or random_bad_10_21_
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000_' + demo_prefix + str(num) + '.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '2'

    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.l2_regularization_loss_weight = 1.0e-5
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    common_data_size(data_num)
    FLAGS.method_name = tag + '_' + demo_prefix + '_' + str(num) + '_' + str(data_num)
    #FLAGS.game_config_fname = 'huazhe.xml'

'''def torcs_V_grounding_demo_stage_1_bad(demo_prefix, num):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    # demo_prefix = block_bad_fix_max_ or random_bad_10_21_
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000_' + demo_prefix + str(num) + '.p'
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000autoback_simple_bad_08_25.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '1'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    #FLAGS.game_config_fname = 'huazhe.xml'  '''

def torcs_V_grounding_no_weighting_demo_stage_1_bad(demo_prefix, num, data_num):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    # demo_prefix = block_bad_fix_max_ or random_bad_10_21_
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000_' + demo_prefix + str(num) + '.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '3'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False

    FLAGS.disable_off_policy_weighting = True
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    common_data_size(data_num)
    FLAGS.method_name = tag + '_' + demo_prefix + '_' + str(num) + '_' + str(data_num)

def torcs_soft_Q_demo_stage_1_bad(demo_prefix, num, data_num):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30

    common_setting()
    # demo_prefix = block_bad_fix_max_ or random_bad_10_21_
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000_' + demo_prefix + str(num) + '.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '0'
    FLAGS.soft_Q_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    common_data_size(data_num)
    FLAGS.method_name = tag + '_' + demo_prefix + '_' + str(num) + '_' + str(data_num)

    # added by Yang, for evaluation only. (Second evaluation trail / batch)
    FLAGS.inenv_eval = True
    FLAGS.ckpt_path = "/backup/hxu/modelRL/torcs_soft_Q_demo_stage_1_bad_segment__90_150000"
    FLAGS.eval_freq = 1
    FLAGS.torcs_path = "/data2/hxu/rlTORCS"
    # adding the val settings, put for training first
    FLAGS.val_set = True
    FLAGS.val_set_file = "/data2/hxu/modelRL/demo_large/linji_torcsTue_Oct_24_21:54:25_PDT_2017.h5"  # only for bellman error
    FLAGS.eval_only = True

"""batch exp for data amount"""

def torcs_cross_entropy_demo_stage_1_amount(num, exp_num):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider) + '_' + str(datetime.now())
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000autoback_simple_08_25.p'
    torcs_dqn_kx(FLAGS.torcs_divider,tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '0'
    FLAGS.supervise_cross_entropy_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(num)
    FLAGS.autoback = True
    FLAGS.method_name = tag + '_' + str(num) + '_' + str(exp_num).zfill(2) + '_new'
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_hard_Q_demo_stage_1_amount(num, exp_num):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000autoback_simple_08_25.p'
    torcs_dqn_kx(FLAGS.torcs_divider,tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '3'
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(num)
    FLAGS.autoback = True
    FLAGS.method_name = tag + '_' + str(num) + '_' + str(exp_num).zfill(2) + '_new'
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_dqfd_full_demo_stage_1_amount(num, exp_num):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000autoback_simple_08_25.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '1'

    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.l2_regularization_loss_weight = 1.0e-5
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(num)
    FLAGS.autoback = True
    FLAGS.method_name = tag + '_' + str(num) + '_' + str(exp_num).zfill(2) + '_new'
    #FLAGS.game_config_fname = 'huazhe.xml'


def torcs_V_grounding_no_weighting_demo_stage_1_amount(num, exp_num):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000autoback_simple_08_25.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '2'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False

    FLAGS.disable_off_policy_weighting = True
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(num)
    FLAGS.autoback = True
    FLAGS.method_name = tag + '_' + str(num) + '_' + str(exp_num).zfill(2) + '_new'
    #FLAGS.game_config_fname = 'huazhe.xml'


    # added by Yang, for evaluation only. (Second evaluation trail / batch)
    FLAGS.inenv_eval = True
    FLAGS.ckpt_path = "/backup/hxu/modelRL/torcs_V_grounding_no_weighting_demo_stage_1_amount_150000_09_new"
    FLAGS.eval_freq = 1
    FLAGS.torcs_path = "/data2/hxu/rlTORCS"
    # adding the val settings, put for training first
    FLAGS.val_set = True
    FLAGS.val_set_file = "/data2/hxu/modelRL/demo_large/linji_torcsTue_Oct_24_21:54:25_PDT_2017.h5" # only for bellman error
    FLAGS.eval_only = True

def torcs_soft_Q_demo_stage_1_amount(num,exp_num):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000autoback_simple_08_25.p'
    torcs_dqn_kx(FLAGS.torcs_divider,tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '4'
    FLAGS.soft_Q_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(num)
    FLAGS.autoback = True
    FLAGS.method_name = tag + '_' + str(num) + '_' + str(exp_num).zfill(2) + '_new'
    #FLAGS.game_config_fname = 'huazhe.xml'


### batch for bad
def torcs_cross_entropy_demo_stage_1_bad_batch(demo_prefix, num, data_num, exp_num):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider) + '_' + str(datetime.now())
    common_setting()
    #demo_prefix = block_bad_fix_max_ or random_bad_10_21_
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000_'+ demo_prefix +str(num)+'.p'
    torcs_dqn_kx(FLAGS.torcs_divider,tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '3'
    FLAGS.supervise_cross_entropy_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    common_data_size(data_num)
    FLAGS.method_name = tag+ '_' + demo_prefix + '_' +str(num) + '_' + str(data_num) + '_' + str(exp_num).zfill(2)

    #FLAGS.game_config_fname = 'huazhe.xml'

    # addedbyYang,forevaluationonly
    FLAGS.inenv_eval = True
    FLAGS.ckpt_path = "/backup/hxu/modelRL/torcs_cross_entropy_demo_stage_1_bad_batch_segment__90_300000_00"
    FLAGS.eval_freq = 1
    FLAGS.torcs_path = "/data2/hxu/rlTORCS"

def torcs_hard_Q_demo_stage_1_bad_batch(demo_prefix, num, data_num, exp_num):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    # demo_prefix = block_bad_fix_max_ or random_bad_10_21_
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000_'+ demo_prefix +str(num)+'.p'
    torcs_dqn_kx(FLAGS.torcs_divider,tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '5'
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    common_data_size(data_num)
    FLAGS.method_name = tag + '_' + demo_prefix + '_' + str(num) + '_' + str(data_num) + '_' + str(exp_num).zfill(2)
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_dqfd_full_demo_stage_1_bad_batch(demo_prefix, num, data_num, exp_num):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    #FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    # demo_prefix = block_bad_fix_max_ or random_bad_10_21_
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000_' + demo_prefix + str(num) + '.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '4'

    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.l2_regularization_loss_weight = 1.0e-5
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    common_data_size(data_num)
    FLAGS.method_name = tag + '_' + demo_prefix + '_' + str(num) + '_' + str(data_num) + '_' + str(exp_num).zfill(2)
    #FLAGS.game_config_fname = 'huazhe.xml'

    # addedbyYang,forevaluationonly
    FLAGS.inenv_eval = True
    FLAGS.ckpt_path = "/backup/hxu/modelRL/torcs_dqfd_full_demo_stage_1_bad_batch_segment__90_300000_00"
    FLAGS.eval_freq = 1
    FLAGS.torcs_path = "/data2/hxu/rlTORCS"


def torcs_V_grounding_no_weighting_demo_stage_1_bad_batch(demo_prefix, num, data_num, exp_num):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    # demo_prefix = block_bad_fix_max_ or random_bad_10_21_
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000_' + demo_prefix + str(num) + '.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '5'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False

    FLAGS.disable_off_policy_weighting = True
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    common_data_size(data_num)
    FLAGS.method_name = tag + '_' + demo_prefix + '_' + str(num) + '_' + str(data_num) + '_' + str(exp_num).zfill(2)

    # added by Yang, for evaluation only
    FLAGS.inenv_eval = True
    FLAGS.ckpt_path = "/backup/hxu/modelRL/torcs_V_grounding_no_weighting_demo_stage_1_bad_batch_segment__90_300000_00"
    FLAGS.eval_freq = 1
    FLAGS.torcs_path = "/data2/hxu/rlTORCS"

def torcs_soft_Q_demo_stage_1_bad_batch(demo_prefix, num, data_num, exp_num):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30

    common_setting()
    # demo_prefix = block_bad_fix_max_ or random_bad_10_21_
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000_' + demo_prefix + str(num) + '.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '4'
    FLAGS.soft_Q_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    common_data_size(data_num)
    FLAGS.method_name = tag + '_' + demo_prefix + '_' + str(num) + '_' + str(data_num) + '_' + str(exp_num).zfill(2)



#### final behavior

def torcs_human_all_cross_entropy_demo_stage_1_hxu_slow_final(core, exp_num):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider) + '_' + str(datetime.now())
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/hxu_slow_all.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = core
    FLAGS.supervise_cross_entropy_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    FLAGS.method_name = tag + '_' + str(exp_num) + '_new'
    # FLAGS.game_config_fname = 'huazhe.xml'



def torcs_human_all_dqfd_full_demo_stage_1_hxu_slow_final(core, exp_num):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/hxu_slow_all.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = core

    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.l2_regularization_loss_weight = 1.0e-5
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    FLAGS.method_name = tag + '_' + str(exp_num) + '_new'
    # FLAGS.game_config_fname = 'huazhe.xml'



def torcs_human_all_V_grounding_no_weighting_demo_stage_1_hxu_slow_final(core, exp_num):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = parse_name('/data/hxu/modelRL/hxu_slow_all.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "hdf"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = core

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False

    FLAGS.disable_off_policy_weighting = True
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.human_torcs = True
    FLAGS.method_name = tag + '_' + str(exp_num) + '_new'
    # FLAGS.game_config_fname = 'huazhe.xml'

    # added by Yang, for evaluation only. (Second evaluation trail / batch)
    FLAGS.inenv_eval = True
    FLAGS.ckpt_path = "/backup/hxu/modelRL/torcs_human_all_V_grounding_no_weighting_demo_stage_1_hxu_slow_final_00_new"
    FLAGS.eval_freq = 1
    FLAGS.torcs_path = "/data2/hxu/rlTORCS"
    # adding the val settings, put for training first
    FLAGS.val_set = True
    FLAGS.val_set_file = "/data2/hxu/modelRL/demo_large/linji_torcsTue_Oct_24_21:54:25_PDT_2017.h5"
    FLAGS.eval_only = True


# Yang digging out the NAC config
def torcs_V_grounding_no_weighting_demo_stage_1_amount_revive(num, exp_num):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000autoback_simple_08_25.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '0'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False

    FLAGS.disable_off_policy_weighting = True
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(num)
    FLAGS.autoback = True
    FLAGS.method_name = tag + '_' + str(num) + '_' + str(exp_num).zfill(2) + '_new'
    #FLAGS.game_config_fname = 'huazhe.xml'

def torcs_V_grounding_no_weighting_demo_stage_1_amount_revive_piv(num, exp_num):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000autoback_simple_08_25.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '0'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False

    FLAGS.disable_off_policy_weighting = True
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(num)
    FLAGS.autoback = True
    FLAGS.method_name = tag + '_' + str(num) + '_' + str(exp_num).zfill(2) + '_new'
    #FLAGS.game_config_fname = 'huazhe.xml'

    FLAGS.pi_v_model = True

# try advantage learning
def torcs_V_grounding_no_weighting_demo_stage_1_amount_advantage(num, exp_num):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000autoback_simple_08_25.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '0'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 0.0
    FLAGS.exp_advantage_diff_learning = 1.0
    FLAGS.critic_use_rapid_weighting = False

    FLAGS.disable_off_policy_weighting = True

    FLAGS.custom_reward = 'reward_ben'
    common_data_size(num)
    FLAGS.autoback = True
    FLAGS.method_name = tag + '_' + str(num) + '_' + str(exp_num).zfill(2) + '_new'

# the advantage second stage.
def torcs_V_grounding_no_weighting_inenv_stage_2_rerun_advantage(key):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = ''
    torcs_dqn_kx(FLAGS.torcs_divider, tag)
    FLAGS.torcs_path = '/data/yang/code/rlTORCS'
    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000/FLAGS.torcs_divider
    FLAGS.eval_freq = -1
    FLAGS.torcs_demo = False
    FLAGS.core_num = '0'
    FLAGS.replay_buffer_size = int(3e5)
    FLAGS.exploration_schedule = PiecewiseSchedule(
        [
            (0, 0.01),
            (1e6, 0.01),
            (1e7, 0.01),
        ], outside_value=0.01
    )
    # todo: change this line
    FLAGS.ckpt_path = "/data/yang/code/rl_demonstration/hw3/link_data/torcs_V_grounding_no_weighting_demo_stage_1_amount_advantage_300000_00_new"

    FLAGS.inenv_finetune = True

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 0.0
    FLAGS.exp_advantage_diff_learning = 1.0
    FLAGS.critic_use_rapid_weighting = False

    FLAGS.disable_off_policy_weighting = True

    FLAGS.custom_reward = "reward_ben"

    FLAGS.method_name = tag + '_' + key + '_new'

# run second stage with weighting
def torcs_V_grounding_no_weighting_inenv_stage_2_rerun_Critic_Weighting(critic_weight):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = ''
    torcs_dqn_kx(FLAGS.torcs_divider, tag)
    FLAGS.torcs_path = '/data/yang/code/rlTORCS'
    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000/FLAGS.torcs_divider
    FLAGS.eval_freq = -1
    FLAGS.torcs_demo = False
    FLAGS.core_num = '0'
    FLAGS.replay_buffer_size = int(3e5)
    FLAGS.exploration_schedule = PiecewiseSchedule(
        [
            (0, 0.01),
            (1e6, 0.01),
            (1e7, 0.01),
        ], outside_value=0.01
    )
    # todo: change this line
    FLAGS.ckpt_path = "/data/yang/code/rl_demonstration/hw3/link_data/torcs_V_grounding_no_weighting_demo_stage_1_amount_revive_300000_00_new"

    FLAGS.inenv_finetune = True

    FLAGS.exp_value_critic_weighting = float(critic_weight)
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.exp_advantage_diff_learning = 0.0
    FLAGS.critic_use_rapid_weighting = False

    FLAGS.disable_off_policy_weighting = False
    FLAGS.disable_off_policy_weighting_PG = True

    FLAGS.custom_reward = "reward_ben"

    FLAGS.method_name = tag + '_' + str(critic_weight) + '_new'

# try advantage learning
def torcs_V_grounding_no_weighting_demo_stage_1_amount_advantage_nocritic(num, exp_num):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000autoback_simple_08_25.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '1'

    FLAGS.exp_value_critic_weighting = 0.0
    FLAGS.exp_policy_grad_weighting = 0.0
    FLAGS.exp_advantage_diff_learning = 1.0
    FLAGS.critic_use_rapid_weighting = False

    FLAGS.disable_off_policy_weighting = True

    FLAGS.custom_reward = 'reward_ben'
    common_data_size(num)
    FLAGS.autoback = True
    FLAGS.method_name = tag + '_' + str(num) + '_' + str(exp_num).zfill(2) + '_new'

use_this_config = test_test
