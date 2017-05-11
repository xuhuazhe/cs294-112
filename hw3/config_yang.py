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

def yang_common_setting():
    common_setting()
    FLAGS.learning_stage = True
    FLAGS.batch_size = 32

def yang_cross_entropy():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag
    yang_common_setting()

    FLAGS.supervise_cross_entropy_loss_weight = 1.0
    FLAGS.core_num = '0'

def yang_hinge_dqfd():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag
    yang_common_setting()

    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.core_num = '0'

def yang_hinge_standard():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag
    yang_common_setting()

    FLAGS.supervise_hinge_standard_loss_weight = 1.0
    FLAGS.core_num = '1'

def yang_hard_Q():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag
    yang_common_setting()

    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.core_num = '1'

def yang_soft_Q():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag
    yang_common_setting()

    FLAGS.soft_Q_loss_weight = 1.0
    FLAGS.core_num = '2'

def yang_DQfD_no_l2():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag
    yang_common_setting()

    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.core_num = '2'

def yang_DQfD_full():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag
    yang_common_setting()

    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.l2_regularization_loss_weight = 1.0e-5
    FLAGS.core_num = '3'

def yang_policy_gradient_soft_1_step_fast():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag
    yang_common_setting()

    FLAGS.policy_gradient_soft_1_step = 1.0
    FLAGS.core_num = '1'
    FLAGS.eval_freq = 100000


def yang_policy_gradient_soft_1_step_fast_largeBatch():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag
    yang_common_setting()

    FLAGS.policy_gradient_soft_1_step = 1.0
    FLAGS.core_num = '2'
    FLAGS.eval_freq = 100000
    FLAGS.batch_size = 512

def yang_policy_gradient_soft_1_step_surrogate_stopgrad():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag
    yang_common_setting()

    FLAGS.policy_gradient_soft_1_step = 1.0
    FLAGS.core_num = '3'
    FLAGS.eval_freq = 100000

def yang_policy_gradient_soft_1_step_reproduce_overfit():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag
    yang_common_setting()

    FLAGS.policy_gradient_soft_1_step = 1.0
    FLAGS.core_num = '4'
    FLAGS.eval_freq = 100000

def yang_soft_Q_in_env():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag
    yang_common_setting()

    FLAGS.core_num = '5'

    # Q learning specific
    FLAGS.eval_freq = -1
    FLAGS.demo_mode = "no_demo"
    FLAGS.soft_Q_loss_weight = 1.0
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000


use_this_config = yang_soft_Q_in_env
