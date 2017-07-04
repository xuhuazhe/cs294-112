import sys
import os
import inspect
sys.path.append('../')
import tensorflow as tf
from dqn_utils import *
FLAGS = tf.app.flags.FLAGS


def common_setting():
    FLAGS.ddqn = False

    FLAGS.demo_mode = 'replay'
    FLAGS.demo_file_path = '/data/hxu/cs294-112/hw3/link_data/hardQ_expert.p'
    #FLAGS.demo_file_path   = '/backup/hxu/cs294-112/hw3/link_data/dmformat_demo/enduro-dm-egs-00.h5'  #'/data/hxu/cs294-112/hw3/DQfD/enduro-egs.h5'
    #FLAGS.demo_file_path_1 = '/backup/hxu/cs294-112/hw3/link_data/dmformat_demo/enduro-dm-egs-01.h5'
    FLAGS.bad_dir = '/data/hxu/cs294-112/hw3/link_data/bad_demo.p'
    FLAGS.collect_Q_experience = False

    FLAGS.eval_freq = 10000
    #FLAGS.tiny_explore = 0.01

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
    #FLAGS.demo_mode = 'hdf'
    # FLAGS.demo_file_path = '/data/hxu/cs294-112/hw3/DQfD/enduro-egs.h5'
    # FLAGS.demo_file_path = '/backup/hxu/cs294-112/hw3/link_data/dmformat_demo/enduro-dm-egs-00.h5, /backup/hxu/cs294-112/hw3/link_data/dmformat_demo/enduro-dm-egs-01.h5'
    # FLAGS.demo_file_path = '/backup/hxu/cs294-112/hw3/link_data/dmformat_demo/enduro.h5'
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000
    FLAGS.tiny_explore = 0.01
    FLAGS.learning_starts = 0
    print(FLAGS.demo_file_path)
    FLAGS.learning_stage = True
    FLAGS.batch_size = 32

    if tag is not None:
        print("config name = ", tag)
        FLAGS.method_name = FLAGS.tag_prefix + tag + "_" + str(FLAGS.ddqn) + "_" + str(FLAGS.soft_Q_alpha) + "_" + str(FLAGS.torcs_divider)

def torcs_dqn_kx(divider, tag):
    yang_common_setting(tag)
    FLAGS.torcs_demo = True
    FLAGS.core_num = '0'

    # Q learning specific
    FLAGS.eval_freq = -1
    FLAGS.demo_mode = "no_demo"
    #FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.collect_Q_experience = True

    FLAGS.env_id="rltorcs-v0"

    # begin the divider attempt
    num_iterations = int(4e7) / 4 / divider
    FLAGS.learning_starts = 50000 / divider
    FLAGS.target_update_freq = 10000 / divider
    FLAGS.replay_buffer_size = 1000000 / divider
    FLAGS.max_timesteps = int(4e7) / divider
    FLAGS.dataset_size = 1000000 / divider

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
    FLAGS.core_num = '7'
    FLAGS.eval_freq = -1
    FLAGS.env_id = "rltorcs-v0"
    # param for demonstration
    FLAGS.learning_stage = False
    FLAGS.ckpt_path = '/data/yang/code/rl_demonstration/hw3/link_data/torcs_dqn_30x_False_0.1'
    FLAGS.replay_buffer_size = 1000000 / divider
    FLAGS.max_timesteps = int(4e6) / 4 / divider
    num_iterations = int(4e7) / 4 / divider
    FLAGS.exploration_schedule = PiecewiseSchedule([
        (0, 0.01),
        (1e6 / divider, 0.01),
        (num_iterations / 2, 0.01)],
        outside_value=0.01)


def torcs_10x_BenReward():
    tag = inspect.stack()[0][3]
    torcs_dqn_kx(10, tag)
    FLAGS.custom_reward = "reward_ben"

def collect_torcs_demonstration():
    tag = inspect.stack()[0][3]
    FLAGS.torcs_divider = 1
    print(FLAGS.torcs_divider,'x torcs demo')
    torcs_collect(FLAGS.torcs_divider, tag)
    FLAGS.custom_reward = "reward_ben"

def torcs_cross_entropy_demo():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/data/hxu/cs294-112/hw3/link_data/30torcs_new.p'
    yang_common_setting()
    torcs_dqn_kx(FLAGS.torcs_divider,tag)

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '6'
    FLAGS.supervise_cross_entropy_loss_weight = 1.0

def torcs_hinge_dqfd_demo():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/data/hxu/cs294-112/hw3/link_data/30torcs_new.p'
    yang_common_setting()
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '0'
    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0

def torcs_hard_Q_demo():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/data/hxu/cs294-112/hw3/link_data/30torcs_new.p'
    yang_common_setting()
    torcs_dqn_kx(FLAGS.torcs_divider,tag)

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '1'
    FLAGS.hard_Q_loss_weight = 1.0

def torcs_soft_Q_demo():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/data/hxu/cs294-112/hw3/link_data/30torcs_new.p'
    yang_common_setting()
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '2'
    FLAGS.soft_Q_loss_weight = 1.0

def torcs_dqfd_full_demo():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/data/hxu/cs294-112/hw3/link_data/30torcs_new.p'
    yang_common_setting()
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '3'

    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.l2_regularization_loss_weight = 1.0e-5

def torcs_PG_Vloss_rapidWeighting_demo():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/data/hxu/cs294-112/hw3/link_data/30torcs_new.p'
    yang_common_setting()
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000
    FLAGS.core_num = '4'

    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = True



use_this_config = test_test

