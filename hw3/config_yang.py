import sys
import os
import inspect
sys.path.append('../')
import tensorflow as tf
from dqn_utils import PiecewiseSchedule
FLAGS = tf.app.flags.FLAGS


def common_setting():
    FLAGS.demo_mode = 'hdf'
    #FLAGS.demo_file_path = '/data/hxu/cs294-112/hw3/DQfD/enduro-egs.h5'
    #FLAGS.demo_file_path = '/backup/hxu/cs294-112/hw3/link_data/dmformat_demo/enduro-dm-egs-00.h5, /backup/hxu/cs294-112/hw3/link_data/dmformat_demo/enduro-dm-egs-01.h5'
    #FLAGS.demo_file_path = '/backup/hxu/cs294-112/hw3/link_data/dmformat_demo/enduro.h5'
    FLAGS.collect_Q_experience = False

    FLAGS.eval_freq = 10000
    FLAGS.tiny_explore = 0.01

    FLAGS.learning_starts = 0
    print(FLAGS.demo_file_path)

def yang_common_setting(tag=None):
    common_setting()
    FLAGS.learning_stage = True
    FLAGS.batch_size = 32

    if tag is not None:
        print("config name = ", tag)
        FLAGS.method_name = FLAGS.tag_prefix + tag + "_" + str(FLAGS.ddqn) + "_" + str(FLAGS.soft_Q_alpha)

def yang_cross_entropy():
    tag = inspect.stack()[0][3]
    yang_common_setting(tag)

    FLAGS.supervise_cross_entropy_loss_weight = 1.0
    FLAGS.core_num = '2'

def yang_hinge_dqfd():
    tag = inspect.stack()[0][3]
    yang_common_setting(tag)

    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.core_num = '3'

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

    FLAGS.core_num = '0'

    # Q learning specific
    FLAGS.eval_freq = -1
    FLAGS.demo_mode = "no_demo"
    FLAGS.soft_Q_loss_weight = 1.0
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000
    FLAGS.soft_Q_alpha = 0.1


def yang_soft_Q_in_env_tinyExplore():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag
    yang_common_setting()

    FLAGS.core_num = '0'

    # Q learning specific
    FLAGS.eval_freq = -1
    FLAGS.demo_mode = "no_demo"
    FLAGS.soft_Q_loss_weight = 1.0
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000
    FLAGS.soft_Q_alpha = 0.1

    FLAGS.tiny_explore = 0.01

def yang_soft_Q_in_env_lowtemp():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag
    yang_common_setting()

    FLAGS.core_num = '1'

    # Q learning specific
    FLAGS.eval_freq = -1
    FLAGS.demo_mode = "no_demo"
    FLAGS.soft_Q_loss_weight = 1.0
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000
    FLAGS.soft_Q_alpha = 0.01


def yang_soft_Q_in_env_tinyExplore_lowtemp():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag
    yang_common_setting()

    FLAGS.core_num = '1'

    # Q learning specific
    FLAGS.eval_freq = -1
    FLAGS.demo_mode = "no_demo"
    FLAGS.soft_Q_loss_weight = 1.0
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000
    FLAGS.soft_Q_alpha = 0.01

    FLAGS.tiny_explore = 0.01

def yang_hard_Q_in_env():
    tag = inspect.stack()[0][3]
    yang_common_setting(tag)

    FLAGS.core_num = '0'

    # Q learning specific
    FLAGS.eval_freq = -1
    FLAGS.demo_mode = "no_demo"
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000

def yang_DQfD_no_l2_softQ():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag
    yang_common_setting()

    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.soft_Q_loss_weight = 1.0
    FLAGS.core_num = '0'

def yang_exp_soft_Q_bellman_alpha01():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag
    yang_common_setting()

    FLAGS.exp_soft_Q_bellman = 1.0
    FLAGS.core_num = '2'
    FLAGS.soft_Q_alpha = 0.1

def yang_exp_soft_Q_bellman_alpha10():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag
    yang_common_setting()

    FLAGS.exp_soft_Q_bellman = 1.0
    FLAGS.core_num = '2'
    FLAGS.soft_Q_alpha = 1.0

def yang_exp_soft_Q_bellman_mixed():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag
    yang_common_setting()

    FLAGS.exp_soft_Q_bellman = 1.0
    FLAGS.core_num = '2'
    FLAGS.soft_Q_alpha = 1.0

def yang_exp_policy_grad_weighting():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag
    yang_common_setting()

    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.core_num = '0'
    FLAGS.soft_Q_alpha = 1.0

def yang_exp_policy_grad_weighting_01():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag
    yang_common_setting()

    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.core_num = '0'
    FLAGS.soft_Q_alpha = 0.1

def yang_exp_policy_grad_weighting_001():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag
    yang_common_setting()

    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.core_num = '0'
    FLAGS.soft_Q_alpha = 0.01

def yang_exp_policy_grad_weighting_batch512():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag
    yang_common_setting()

    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.core_num = '0'
    FLAGS.soft_Q_alpha = 1.0
    FLAGS.batch_size = 512

def yang_exp_policy_grad_weighting_surrogate():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag
    yang_common_setting()

    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.core_num = '0'
    FLAGS.soft_Q_alpha = 1.0

def yang_exp_policy_grad_weighting_batch512_surrogate():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag
    yang_common_setting()

    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.core_num = '0'
    FLAGS.soft_Q_alpha = 1.0
    FLAGS.batch_size = 512

# TODO rerun all soft Q experiment with corrected Q2V
def redo_soft_Q():
    tag = inspect.stack()[0][3]
    yang_common_setting(tag)

    FLAGS.soft_Q_loss_weight = 1.0
    FLAGS.core_num = '0'

def redo_exp_soft_Q_bellman():
    tag = inspect.stack()[0][3]
    yang_common_setting(tag)

    FLAGS.exp_soft_Q_bellman = 1.0
    FLAGS.core_num = '1'

def redo_exp_soft_Q_bellman_mixed():
    tag = inspect.stack()[0][3]
    yang_common_setting(tag)

    FLAGS.exp_soft_Q_bellman = 1.0
    FLAGS.core_num = '1'

def redo_policy_gradient_soft_1_step_surrogate():
    tag = inspect.stack()[0][3]
    yang_common_setting(tag)

    FLAGS.policy_gradient_soft_1_step_surrogate = 1.0
    FLAGS.core_num = '0'

# comparison experiments
def redo_DQfD_no_l2():
    tag = inspect.stack()[0][3]
    yang_common_setting(tag)

    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.core_num = '0'

def redo_DQfD_no_l2_softQ():
    tag = inspect.stack()[0][3]
    yang_common_setting(tag)

    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.soft_Q_loss_weight = 1.0
    FLAGS.core_num = '1'

def redo_policy_gradient_soft_1_step():
    tag = inspect.stack()[0][3]
    yang_common_setting(tag)

    FLAGS.policy_gradient_soft_1_step = 1.0
    FLAGS.core_num = '2'

def redo_exp_policy_grad_weighting():
    tag = inspect.stack()[0][3]
    yang_common_setting(tag)

    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.core_num = '3'

# new sets of experiments
def exp_advantage_diff_learning():
    tag = inspect.stack()[0][3]
    yang_common_setting(tag)

    FLAGS.exp_advantage_diff_learning = 1.0
    FLAGS.core_num = '1'

# tune soft Q learning with force_original_exploration=True
def yang_soft_Q_in_env_original_explore():
    tag = inspect.stack()[0][3]
    yang_common_setting(tag)

    FLAGS.core_num = '0'

    # Q learning specific
    FLAGS.eval_freq = -1
    FLAGS.demo_mode = "no_demo"
    FLAGS.soft_Q_loss_weight = 1.0
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000

    FLAGS.force_original_exploration = True

def DQfD_no_l2_official_double():
    tag = inspect.stack()[0][3]
    yang_common_setting(tag)

    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.core_num = '0'

    FLAGS.target_update_freq = 30000
    FLAGS.ddqn = True

def yang_DDQN_in_env_official_double_longtarget():
    FLAGS.ddqn = True

    tag = inspect.stack()[0][3]
    yang_common_setting(tag)

    FLAGS.core_num = '0'

    # Q learning specific
    FLAGS.eval_freq = -1
    FLAGS.demo_mode = "no_demo"
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000

    FLAGS.target_update_freq = 30000

def roadrunner():
    tag = inspect.stack()[0][3]
    yang_common_setting(tag)

    FLAGS.core_num = '0'

    # Q learning specific
    FLAGS.eval_freq = -1
    FLAGS.demo_mode = "no_demo"
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000

    FLAGS.env_id="RoadRunnerNoFrameskip-v3"
    # set ddqn flag in command line

def roadrunner_ddqn():
    FLAGS.ddqn = True

    tag = inspect.stack()[0][3]
    yang_common_setting(tag)

    FLAGS.core_num = '0'

    # Q learning specific
    FLAGS.eval_freq = -1
    FLAGS.demo_mode = "no_demo"
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000

    FLAGS.env_id="RoadRunnerNoFrameskip-v3"

    FLAGS.target_update_freq = 30000

def exp_policy_grad_weighting_ratio_env():
    tag = inspect.stack()[0][3]
    yang_common_setting(tag)

    FLAGS.core_num = '0'

    # Q learning specific
    FLAGS.eval_freq = -1
    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000

    #FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.force_original_exploration = True

# from this point, we change the evaluation to Standard Eval.
# i.e. eps=0.05, and max emulator time 30 mins
def exp_policy_grad_weighting_ratio_env_pong():
    tag = inspect.stack()[0][3]
    yang_common_setting(tag)

    FLAGS.core_num = '0'

    # Q learning specific
    FLAGS.eval_freq = 10000
    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000

    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.env_id="PongNoFrameskip-v3"

def exp_policy_grad_weighting_ratio_env_smallLR():
    tag = inspect.stack()[0][3]
    yang_common_setting(tag)

    FLAGS.core_num = '0'

    # Q learning specific
    FLAGS.eval_freq = 10000
    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000

    FLAGS.exp_policy_grad_weighting = 1.0

    num_iterations = float(4e7) / 4.0
    lr_multiplier = 0.1
    FLAGS.lr_schedule = PiecewiseSchedule([
        (0, 1e-4 * lr_multiplier),
        (num_iterations / 10, 1e-4 * lr_multiplier),
        (num_iterations / 2, 5e-5 * lr_multiplier),
        ],
        outside_value=5e-5 * lr_multiplier)

def exp_policy_grad_weighting_ratio_env_plus_Qloss():
    tag = inspect.stack()[0][3]
    yang_common_setting(tag)

    FLAGS.core_num = '0'

    # Q learning specific
    FLAGS.eval_freq = 10000
    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000

    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.soft_Q_loss_weight = 1.0

def soft_Q_base():
    tag = inspect.stack()[0][3]
    yang_common_setting(tag)

    FLAGS.core_num = '0'

    # Q learning specific
    FLAGS.eval_freq = 10000
    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000

    FLAGS.soft_Q_loss_weight = 1.0

def PG_Vloss():
    tag = inspect.stack()[0][3]
    yang_common_setting(tag)

    FLAGS.core_num = '0'

    # Q learning specific
    FLAGS.eval_freq = 10000
    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000

    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.exp_value_critic_weighting = 1.0

def soft_Q_soft_explore():
    tag = inspect.stack()[0][3]
    yang_common_setting(tag)

    FLAGS.core_num = '0'

    # Q learning specific
    FLAGS.eval_freq = 10000
    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000

    FLAGS.soft_Q_loss_weight = 1.0
    FLAGS.greedy_method = "soft"
    FLAGS.explore_value_method = "normal"

    FLAGS.env_id = "PongNoFrameskip-v3"

def hard_Q_paper_explore():
    tag = inspect.stack()[0][3]
    yang_common_setting(tag)

    FLAGS.core_num = '0'

    # Q learning specific
    FLAGS.eval_freq = 10000
    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000

    FLAGS.hard_Q_loss_weight = 1.0

    # config exploration schedule
    FLAGS.exploration_schedule = PiecewiseSchedule(
        [
            (0, 1.0),
            (1e6, 0.1),
        ], outside_value=0.1
    )

def PG_Vloss_rapidWeighting():
    tag = inspect.stack()[0][3]
    yang_common_setting(tag)

    FLAGS.core_num = '1'

    # Q learning specific
    FLAGS.eval_freq = 10000
    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000

    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = True

############ Begin the TORCS Game ################
def torcs_dqn():
    tag = inspect.stack()[0][3]
    yang_common_setting(tag)

    FLAGS.core_num = '0'

    # Q learning specific
    FLAGS.eval_freq = -1
    FLAGS.demo_mode = "no_demo"
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000

    FLAGS.env_id="rltorcs-v0"

def torcs_dqn_debug():
    tag = inspect.stack()[0][3]
    yang_common_setting(tag)

    FLAGS.core_num = '0'

    # Q learning specific
    FLAGS.eval_freq = -1
    FLAGS.demo_mode = "no_demo"
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 10

    FLAGS.env_id="rltorcs-v0"
