import sys
import os
import inspect
sys.path.append('../')
import tensorflow as tf
from dqn_utils import PiecewiseSchedule
FLAGS = tf.app.flags.FLAGS


def common_setting():
    FLAGS.demo_mode = 'hdf'
    # The non synced version of demonstration
    #FLAGS.demo_file_path = '/data/hxu/cs294-112/hw3/DQfD/enduro-egs.h5'
    # The synced version
    #FLAGS.demo_file_path = '/backup/hxu/cs294-112/hw3/link_data/dmformat_demo/enduro-dm-egs-00.h5, /backup/hxu/cs294-112/hw3/link_data/dmformat_demo/enduro-dm-egs-01.h5'
    # The harry's version, the quality is much worse
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

    #FLAGS.inenv_finetune = True
    #FLAGS.ckpt_path = FLAGS.method_name

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

def torcs_dqn_84x84():
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

    FLAGS.torcs_resolution="84x84"

def torcs_dqn_kx(divider, tag):
    yang_common_setting(tag)

    FLAGS.core_num = '1'

    # Q learning specific
    FLAGS.eval_freq = -1
    FLAGS.demo_mode = "no_demo"
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.collect_Q_experience = True

    FLAGS.env_id="rltorcs-v0"

    # begin the divider attempt
    num_iterations = int(4e7) / 4 / divider
    FLAGS.learning_starts = 50000 / divider
    FLAGS.target_update_freq = 10000 / divider
    FLAGS.replay_buffer_size = 1000000 / divider
    FLAGS.max_timesteps = int(4e7) / divider

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

def torcs_dqn_3x():
    tag = inspect.stack()[0][3]
    torcs_dqn_kx(3, tag)

def torcs_dqn_10x():
    tag = inspect.stack()[0][3]
    torcs_dqn_kx(10, tag)

def torcs_dqn_30x():
    tag = inspect.stack()[0][3]
    torcs_dqn_kx(30, tag)

def torcs_dqn_100x():
    tag = inspect.stack()[0][3]
    torcs_dqn_kx(100, tag)

def torcs_dqn_sensible_kx(divider, tag):
    yang_common_setting(tag)

    FLAGS.core_num = '0'

    # Q learning specific
    FLAGS.eval_freq = -1
    FLAGS.demo_mode = "no_demo"
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.collect_Q_experience = True

    FLAGS.env_id="rltorcs-v0"

    # begin the divider attempt
    num_iterations = int(4e7) / 4 / divider
    FLAGS.learning_starts = 50000
    FLAGS.target_update_freq = 10000
    FLAGS.replay_buffer_size = 1000000 / divider
    FLAGS.max_timesteps = int(4e7) / divider

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

def torcs_dqn_snesible_3x():
    tag = inspect.stack()[0][3]
    torcs_dqn_sensible_kx(3, tag)

def torcs_dqn_sensible_10x():
    tag = inspect.stack()[0][3]
    torcs_dqn_sensible_kx(10, tag)

def torcs_dqn_snesible_30x():
    tag = inspect.stack()[0][3]
    torcs_dqn_sensible_kx(30, tag)

def torcs_dqn_sensible_100x():
    tag = inspect.stack()[0][3]
    torcs_dqn_sensible_kx(100, tag)

def torcs_10x_BenReward():
    tag = inspect.stack()[0][3]
    torcs_dqn_kx(10, tag)
    FLAGS.custom_reward = "reward_ben"

def torcs_config(tag):
    torcs_dqn_kx(30, tag)
    FLAGS.custom_reward = "reward_ben"
    # need to set this for every torcs instance
    FLAGS.torcs_path = "/data/yang/code/rlTORCS"

def torcs_scan_RB():
    tag = inspect.stack()[0][3]
    if FLAGS.exp_policy_grad_weighting > 0:
        type = "PGweighting"
    elif FLAGS.policy_gradient_soft_1_step > 0:
        type = "PG"
    else:
        raise ValueError("flag not correctly set")
    tag = tag + "_RB" + str(FLAGS.replay_buffer_size) + "_" + type

    RB_commandline = FLAGS.replay_buffer_size
    torcs_config(tag)
    FLAGS.hard_Q_loss_weight = 0
    FLAGS.replay_buffer_size = RB_commandline


def enduro_config(tag):
    yang_common_setting(tag)

    FLAGS.core_num = '1'

    # Q learning specific
    FLAGS.eval_freq = -1
    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000

    # FLAGS.hard_Q_loss_weight = 1.0
    # need to define your own loss

def enduro_scan_RB():
    tag = inspect.stack()[0][3]
    if FLAGS.exp_policy_grad_weighting > 0:
        type = "PGweighting"
    elif FLAGS.policy_gradient_soft_1_step > 0:
        type = "PG"
    else:
        raise ValueError("flag not correctly set")
    tag = tag + "_RB" + str(FLAGS.replay_buffer_size) + "_" + type

    RB_commandline = FLAGS.replay_buffer_size
    enduro_config(tag)
    FLAGS.hard_Q_loss_weight = 0
    FLAGS.replay_buffer_size = RB_commandline

    FLAGS.env_id="EnduroNoFrameskip-v4"

def torcs_V_grounding_consistent_target():
    tag = inspect.stack()[0][3]
    RB = FLAGS.replay_buffer_size
    tag = tag + "_RB" + str(RB)
    torcs_config(tag)
    FLAGS.hard_Q_loss_weight = 0

    # use a small replay buffer to simulate on line case
    FLAGS.replay_buffer_size = RB

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False

def torcs_V_grounding_consistent_rapid():
    tag = inspect.stack()[0][3]
    RB = FLAGS.replay_buffer_size
    tag = tag + "_RB" + str(RB)
    torcs_config(tag)
    FLAGS.hard_Q_loss_weight = 0

    # use a small replay buffer to simulate on line case
    FLAGS.replay_buffer_size = RB

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = True

def torcs_V_grounding_no_weighting():
    tag = inspect.stack()[0][3]
    RB = FLAGS.replay_buffer_size
    tag = tag + "_RB" + str(RB)
    torcs_config(tag)
    FLAGS.hard_Q_loss_weight = 0

    # use a small replay buffer to simulate on line case
    FLAGS.replay_buffer_size = RB

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False

    FLAGS.disable_off_policy_weighting = True

def torcs_V_grounding_baseline():
    tag = inspect.stack()[0][3]
    RB = FLAGS.replay_buffer_size
    tag = tag + "_RB" + str(RB)
    torcs_config(tag)
    FLAGS.hard_Q_loss_weight = 1.0

    # use a small replay buffer to simulate on line case
    FLAGS.replay_buffer_size = RB

def torcs_V_grounding_only():
    tag = inspect.stack()[0][3]
    RB = 10000
    tag = tag + "_RB" + str(RB)
    torcs_config(tag)
    FLAGS.hard_Q_loss_weight = 0

    # use a small replay buffer to simulate on line case
    FLAGS.replay_buffer_size = RB

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 0

    FLAGS.disable_off_policy_weighting = True

def enduro_demo_Q_grounding():
    tag = inspect.stack()[0][3]
    yang_common_setting(tag)

    FLAGS.core_num = '1'

    FLAGS.demo_file_path = '/backup/hxu/cs294-112/hw3/link_data/dmformat_demo/enduro-dm-egs-00.h5, /backup/hxu/cs294-112/hw3/link_data/dmformat_demo/enduro-dm-egs-01.h5'

    FLAGS.soft_Q_loss_weight = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.disable_off_policy_weighting = True

# exploring the PCL 1 step
def torcs_dqn_kx_demo(divider, tag):
    yang_common_setting(tag)

    FLAGS.env_id="rltorcs-v0"

    # begin the divider attempt
    num_iterations = int(4e7) / 4 / divider
    FLAGS.learning_starts = 50000 / divider
    FLAGS.target_update_freq = 10000 / divider
    FLAGS.replay_buffer_size = 1000000 / divider
    FLAGS.max_timesteps = int(4e7) / divider

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

def torcs_config_demo(tag):
    torcs_dqn_kx_demo(30, tag)
    FLAGS.custom_reward = "reward_ben"
    # need to set this for every torcs instance
    FLAGS.torcs_path = "/data/yang/code/rlTORCS"
    FLAGS.torcs_demo = True

def torcs_config_ft(tag):
    torcs_config(tag)
    FLAGS.torcs_demo = False
    FLAGS.exploration_schedule = PiecewiseSchedule([
        (0, 0.05),
        (10, 0.05)],
        outside_value=0.05)
    # some in env flags


def torcs_demo_PCL_PiV_rapidNet():
    tag = inspect.stack()[0][3]
    torcs_config_demo(tag)

    FLAGS.core_num = '5'
    FLAGS.demo_mode="replay"
    FLAGS.demo_file_path = '/data/hxu/cs294-112/hw3/link_data/300000_torcs_0.1explore.p'

    FLAGS.PCL_1_step_weighting = 1.0
    FLAGS.pi_v_model = True

def torcs_demo_PCL_PiV_targetNet():
    tag = inspect.stack()[0][3]
    torcs_config_demo(tag)

    FLAGS.core_num = '1'
    FLAGS.demo_mode = "replay"
    FLAGS.demo_file_path = '/data/hxu/cs294-112/hw3/link_data/300000_torcs_0.1explore.p'

    FLAGS.soft_Q_loss_weight = 1.0
    FLAGS.pi_v_model = True

def torcs_demo_PCL_Q_rapidNet():
    tag = inspect.stack()[0][3]
    torcs_config_demo(tag)

    FLAGS.core_num = '2'
    FLAGS.demo_mode="replay"
    FLAGS.demo_file_path = '/data/hxu/cs294-112/hw3/link_data/300000_torcs_0.1explore.p'

    FLAGS.PCL_1_step_weighting = 1.0
    FLAGS.pi_v_model = False

def torcs_demo_PCL_Q_targetNet():
    tag = inspect.stack()[0][3]
    torcs_config_demo(tag)

    FLAGS.core_num = '3'
    FLAGS.demo_mode = "replay"
    FLAGS.demo_file_path = '/data/hxu/cs294-112/hw3/link_data/300000_torcs_0.1explore.p'

    FLAGS.soft_Q_loss_weight = 1.0
    FLAGS.pi_v_model = False

def torcs_demo_V_grounding():
    tag = inspect.stack()[0][3]
    torcs_config_demo(tag)

    FLAGS.core_num = '6'
    FLAGS.demo_mode = "replay"
    FLAGS.demo_file_path = '/data/hxu/cs294-112/hw3/link_data/300000_torcs_0.1explore.p'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    #FLAGS.critic_use_rapid_weighting = False

    FLAGS.disable_off_policy_weighting = True

# begin the intuitive idea experiments
def torcs_intuitive_2_stages():
    tag = inspect.stack()[0][3]
    phase = 3

    if phase == 1:
        # learning from demo part
        torcs_config_demo(tag)

        FLAGS.demo_mode = "replay"
        FLAGS.demo_file_path = '/data/hxu/cs294-112/hw3/link_data/300000_torcs_0.1explore.p'

        FLAGS.supervise_cross_entropy_loss_weight = 1.0
    elif phase == 2 or phase == 3:
        # finetuning V only
        torcs_config(tag)

        FLAGS.hard_Q_loss_weight = -1.0
        FLAGS.soft_Q_loss_weight = 1.0

        FLAGS.exploration_schedule = PiecewiseSchedule([
            (0, 0.01),
            (10, 0.01)],
            outside_value=0.01)

        FLAGS.inenv_finetune = True
        FLAGS.ckpt_path = "/data/yang/data/link_data_rl/torcs_intuitive_2_stages_False_0.1_phase" + str(phase-1)
    else:
        raise ValueError("wrong phase")

    if phase == 2:
        FLAGS.optimize_V_only = True

    FLAGS.method_name = FLAGS.method_name + "_phase" + str(phase)
    FLAGS.core_num = '0'
    FLAGS.pi_v_model = True

def enduro_config_demo(tag):
    yang_common_setting(tag)
    FLAGS.demo_file_path = '/backup/hxu/cs294-112/hw3/link_data/dmformat_demo/enduro-dm-egs-00.h5, /backup/hxu/cs294-112/hw3/link_data/dmformat_demo/enduro-dm-egs-01.h5'

def enduro_config_inenv(tag):
    yang_common_setting(tag)
    FLAGS.eval_freq = -1
    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000

def enduro_config_ft(tag, ckpt_path):
    enduro_config_inenv(tag)
    # ft related
    finetune_config(ckpt_path)

def finetune_config(ckpt_path):
    FLAGS.inenv_finetune = True
    FLAGS.ckpt_path = ckpt_path
    FLAGS.exploration_schedule = PiecewiseSchedule([
        (0, 0.01),
        (10, 0.01)],
        outside_value=0.01)
    # TODO: learning rate should also be smaller, doesn't change here for now
    FLAGS.lr_schedule = PiecewiseSchedule([
        (0, 5e-5),
        (10, 5e-5)],
        outside_value=5e-5)


def enduro_intuitive_3_stages():
    tag = inspect.stack()[0][3]
    phase = 3

    if phase == 1:
        # learning from demo part
        enduro_config_demo(tag)

        FLAGS.supervise_cross_entropy_loss_weight = 1.0
    elif phase == 2 or phase == 3:
        # finetuning V only
        enduro_config_ft(tag,
                         "/data/yang/data/link_data_rl/" + "enduro_intuitive_3_stages_False_0.1" + "_phase" + str(phase - 1))

        FLAGS.hard_Q_loss_weight = -1.0
        FLAGS.soft_Q_loss_weight = 1.0
    else:
        raise ValueError("wrong phase")

    if phase == 2:
        FLAGS.optimize_V_only = True

    FLAGS.method_name = FLAGS.method_name + "_phase" + str(phase)
    FLAGS.core_num = '0'
    FLAGS.pi_v_model = True


def enduro_intuitive_3_stages_combined():
    tag = inspect.stack()[0][3]
    phase = 3

    if phase == 1:
        # first stage should be reused
        pass
    elif phase == 2:
        ckpt_path = "/data/yang/data/link_data_rl/" + "enduro_intuitive_3_stages_False_0.1" + "_phase1"
        enduro_config_demo(tag)
        finetune_config(ckpt_path)

        FLAGS.hard_Q_loss_weight = -1.0
        FLAGS.soft_Q_loss_weight = 1.0
    elif phase == 3:
        # finetuning V only
        ckpt_path = "/data/yang/data/link_data_rl/" + "enduro_intuitive_3_stages_combined_False_0.1" + "_phase2"
        enduro_config_ft(tag, ckpt_path)

        FLAGS.hard_Q_loss_weight = -1.0
        FLAGS.soft_Q_loss_weight = 1.0
    else:
        raise ValueError("wrong phase")

    if phase == 2:
        FLAGS.optimize_V_only = True

    FLAGS.method_name = FLAGS.method_name + "_phase" + str(phase)
    FLAGS.core_num = '0'
    FLAGS.pi_v_model = True


def frozen_dqn_kx(divider, tag):
    yang_common_setting(tag)

    FLAGS.core_num = '1'

    # Q learning specific
    FLAGS.eval_freq = -1
    FLAGS.demo_mode = "no_demo"
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.collect_Q_experience = True

    FLAGS.env_id="frozen-v0"

    # begin the divider attempt
    num_iterations = int(4e7) / 4 / divider
    FLAGS.learning_starts = 50000 / divider
    FLAGS.target_update_freq = 10000 / divider
    FLAGS.replay_buffer_size = 1000000 / divider
    FLAGS.max_timesteps = int(4e7) / divider

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

def frozen_dqn_v9_gamma0_5():
    tag = inspect.stack()[0][3]

    frozen_dqn_kx(30, tag)
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.core_num = "0"

    FLAGS.frame_history_len = 1
    FLAGS.tabular = True
    FLAGS.discount_factor = 0.5

def collect_demonstration_frozenlake_suboptimal():
    tag = inspect.stack()[0][3]

    frozen_dqn_kx(30, tag)
    #FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.core_num = "0"

    FLAGS.frame_history_len = 1
    FLAGS.tabular = True
    #FLAGS.discount_factor = 0.5

    FLAGS.learning_stage = False
    FLAGS.ckpt_path = '/data/yang/code/rl_demonstration/hw3/link_data/frozen_dqn_v7_gamma0_9_False_0.1'
    FLAGS.exploration_schedule = PiecewiseSchedule([], outside_value=0.01)

    FLAGS.replay_buffer_size = 300
    FLAGS.max_timesteps = FLAGS.replay_buffer_size / 4
    FLAGS.demo_name = "frozenlake_suboptimal"

def frozenlake_dqfd_stage_1_v2():
    tag = inspect.stack()[0][3]

    frozen_dqn_kx(30, tag)
    FLAGS.hard_Q_loss_weight = -1.0
    FLAGS.core_num = "0"

    FLAGS.frame_history_len = 1
    FLAGS.tabular = True
    # FLAGS.discount_factor = 0.5

    FLAGS.demo_file_path = "/data/yang/code/rl_demonstration/hw3/link_data/300_frozenlake_suboptimal.p"
    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False

    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.l2_regularization_loss_weight = 1.0e-5

    FLAGS.dataset_size = 300
    FLAGS.eval_freq = 1000
    FLAGS.learning_starts = 0

def frozenlake_dqfd_stage_2_v2():
    tag = inspect.stack()[0][3]

    frozen_dqn_kx(30, tag)
    FLAGS.hard_Q_loss_weight = -1.0
    FLAGS.core_num = "0"

    FLAGS.frame_history_len = 1
    FLAGS.tabular = True
    # FLAGS.discount_factor = 0.5

    FLAGS.demo_file_path = "/data/yang/code/rl_demonstration/hw3/link_data/300_frozenlake_suboptimal.p"
    FLAGS.demo_mode = "dqfd"
    FLAGS.collect_Q_experience = True

    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.l2_regularization_loss_weight = 1.0e-5

    FLAGS.dataset_size = 300
    FLAGS.eval_freq = -1

    FLAGS.exploration_schedule = PiecewiseSchedule([], outside_value=0.01)
    FLAGS.lr_schedule = PiecewiseSchedule([], outside_value=5e-5)

    FLAGS.ckpt_path = "/data/yang/code/rl_demonstration/hw3/link_data/frozenlake_dqfd_stage_1_v2_False_0.1/"

    FLAGS.inenv_finetune = True
    FLAGS.demo_portion = 0.1

    # TODO: unable to load the checkpoint


def frozenlake_nac_meta(tag):
    frozen_dqn_kx(30, tag)
    FLAGS.hard_Q_loss_weight = -1.0
    FLAGS.core_num = "0"

    FLAGS.frame_history_len = 1
    FLAGS.tabular = True
    # FLAGS.discount_factor = 0.5

    FLAGS.demo_file_path = "/data/yang/code/rl_demonstration/hw3/link_data/300_frozenlake_suboptimal.p"
    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False

    FLAGS.disable_off_policy_weighting = True
    FLAGS.dataset_size = 300
    FLAGS.eval_freq = 1000
    FLAGS.learning_starts = 0

def frozenlake_nac_stage_1_v4():
    tag = inspect.stack()[0][3]
    frozenlake_nac_meta(tag)

def frozenlake_nac_small_RB():
    tag = inspect.stack()[0][3]
    frozenlake_nac_meta(tag)
    FLAGS.replay_buffer_size = 600

def frozenlake_nac_stage_2():
    tag = inspect.stack()[0][3]
    frozen_dqn_kx(30, tag)
    FLAGS.hard_Q_loss_weight = -1.0
    FLAGS.core_num = "0"

    FLAGS.frame_history_len = 1
    FLAGS.tabular = True
    # FLAGS.discount_factor = 0.5

    FLAGS.demo_file_path = ""
    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False
    FLAGS.disable_off_policy_weighting = True

    FLAGS.eval_freq = -1

    FLAGS.exploration_schedule = PiecewiseSchedule([], outside_value=0.01)
    FLAGS.lr_schedule = PiecewiseSchedule([], outside_value=5e-5)

    FLAGS.ckpt_path = "/data/yang/code/rl_demonstration/hw3/link_data/frozenlake_nac_stage_1_v4_False_0.1/"
    FLAGS.inenv_finetune = True
