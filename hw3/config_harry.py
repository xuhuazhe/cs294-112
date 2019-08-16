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
    # FLAGS.demo_file_path   = '/backup/hxu/cs294-112/hw3/link_data/dmformat_demo/enduro-dm-egs-00.h5'  #'/data/hxu/cs294-112/hw3/DQfD/enduro-egs.h5'
    # FLAGS.demo_file_path_1 = '/backup/hxu/cs294-112/hw3/link_data/dmformat_demo/enduro-dm-egs-01.h5'
    FLAGS.bad_dir = ''  # '/data/hxu/cs294-112/hw3/link_data/bad_demo.p'
    FLAGS.collect_Q_experience = False

    FLAGS.eval_freq = 10000
    # FLAGS.tiny_explore = 0.01

    FLAGS.learning_starts = 0
    FLAGS.learning_stage = True
    FLAGS.batch_size = 32


def common_data_size(sz):
    FLAGS.dataset_size = int(sz)  # 30000
    FLAGS.method_name = FLAGS.method_name + "_" + str(FLAGS.dataset_size)

def collect_demonstration_onpolicy():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.method_name = tag + '_badstarts_' + str(FLAGS.bad_starts) + '_badexplore_' + str(FLAGS.tiny_explore)
    common_setting()

    # FLAGS.core_num = '1'
    FLAGS.learning_starts = 0
    FLAGS.learning_stage = False
    FLAGS.ckpt_path = '/data/hxu/cs294-112/hw3/data/'

    FLAGS.exploration_schedule = PiecewiseSchedule(
        [
            (0, 0.01),
            (FLAGS.bad_starts, 0.01),
            (FLAGS.bad_starts + 1, FLAGS.tiny_explore),
            (3e5, FLAGS.tiny_explore),
        ], outside_value=0.01
    )
    # FLAGS.soft_Q_alpha = 0.01
    # FLAGS.m_bad = 300
    # FLAGS.m_good = 300
    # FLAGS.summary_interval = 10


def hard_Q_onpolicy_data():
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    common_setting()
    # FLAGS.group_name = 'rl'
    # FLAGS.demo_file_path = '/data/hxu/cs294-112/hw3/link_data/bad_demo_50000.0_0.7'
    FLAGS.method_name = tag + '_' + str(FLAGS.dataset_size) + '_' + str(FLAGS.bad_portion) + '_' + FLAGS.demo_file_path[
                                                                                                   -12:]

    FLAGS.core_num = '0'
    FLAGS.hard_Q_loss_weight = 1.0
    # optionally set: lr_schedule, exploration_schedule



def yang_common_setting(tag=None):
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000
    FLAGS.tiny_explore = 0.01
    FLAGS.learning_starts = 0
    print(FLAGS.demo_file_path)

    if tag is not None:
        print("config name = ", tag)
        FLAGS.method_name = FLAGS.tag_prefix + tag + '_' + str(FLAGS.torcs_divider) + '_' + str(datetime.now())


def torcs_dqn_kx(divider, tag):
    yang_common_setting(tag)
    FLAGS.torcs_demo = False
    FLAGS.core_num = '0'

    # Q learning specific
    FLAGS.eval_freq = -1
    FLAGS.demo_mode = "no_demo"

    FLAGS.collect_Q_experience = True
    FLAGS.env_id = "rltorcs-v0"

    # begin the divider attempt
    num_iterations = int(4e7) / 4 / divider
    FLAGS.learning_starts = 0  # 50000 / divider
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
    FLAGS.max_timesteps = int(3e5) / 4  # int(4e6) / 4
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
    print(FLAGS.torcs_divider, 'x torcs demo')
    torcs_collect(FLAGS.torcs_divider, tag)
    FLAGS.custom_reward = "reward_ben"
    FLAGS.core_num = '0'


# _____________use_for_write_up__________________________
def collect_torcs_demonstration_add_damage():
    tag = inspect.stack()[0][3]
    FLAGS.torcs_divider = 1
    print(FLAGS.torcs_divider, 'x torcs demo')
    torcs_collect(FLAGS.torcs_divider, tag)
    FLAGS.custom_reward = "reward_ben"
    FLAGS.core_num = '0'
    FLAGS.demo_name = 'add_damage'
    num_iterations = int(4e7) / 4 / FLAGS.torcs_divider
    FLAGS.exploration_schedule = PiecewiseSchedule([
        (0, 0.01),
        (1e6 / FLAGS.torcs_divider, 0.01),
        (num_iterations / 2, 0.01)],
        outside_value=0.01)


'''rerun large scale environment'''


def collect_torcs_demonstration_add_damage_rerun():
    tag = inspect.stack()[0][3]
    FLAGS.torcs_divider = 1
    print(FLAGS.torcs_divider, 'x torcs demo')
    torcs_collect(FLAGS.torcs_divider, tag)
    FLAGS.custom_reward = "reward_ben"
    FLAGS.core_num = '0'
    FLAGS.demo_name = 'add_damage'
    num_iterations = int(4e7) / 4 / FLAGS.torcs_divider
    FLAGS.exploration_schedule = PiecewiseSchedule([
        (0, 0.01),
        (1e6 / FLAGS.torcs_divider, 0.01),
        (num_iterations / 2, 0.01)],
        outside_value=0.01)


def torcs_cross_entropy_demo_stage_1_rerun(index):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider) +'_'+str(datetime.now())
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000add_damage.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

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
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000add_damage.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

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
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
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
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
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
    FLAGS.demo_file_path = '/data2/hxu/modelRL/300000add_damage.p'  # '/backup/hxu/modelRL/300000autoback_simple_08_25.p'
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data2/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '0'
    FLAGS.soft_Q_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    # FLAGS.autoback = True
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
# ckpt_list = {'dqfd':'/backup/hxu/modelRL/torcs_dqfd_full_demo_stage_1_30_2017-08-04 11:39:00.926601_300000',
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

    # FLAGS.ckpt_path = ckpt_list[int(key)]  # '/backup/hxu/modelRL/torcs_dqfd_full_demo_stage_1_30_2017-08-04 11:39:00.926601_300000'
    FLAGS.inenv_finetune = True

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False

    # FLAGS.disable_off_policy_weighting = False#True

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

    # FLAGS.demo_mode = "no_demo"
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
    # FLAGS.supervise_hinge_DQfD_loss_weight = 1.0

    FLAGS.custom_reward = "reward_ben"

    FLAGS.method_name = tag + '_' + key + '_new'
    FLAGS.demo_mode = 'dqfd'
    FLAGS.demo_portion = 0.1
    FLAGS.demo_file_path = '/backup/hxu/modelRL/300000add_damage.p'

    # FLAGS.autoback = True


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
    FLAGS.can_deal_with_human_demo = True
    # FLAGS.game_config_fname = 'huazhe.xml'


def torcs_human_all_hard_Q_demo_stage_1_yang_new():
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

    FLAGS.core_num = '2'
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.can_deal_with_human_demo = True
    # FLAGS.game_config_fname = 'huazhe.xml'


def torcs_human_all_dqfd_full_demo_stage_1_yang_new():
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

    FLAGS.core_num = '3'

    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.l2_regularization_loss_weight = 1.0e-5
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.can_deal_with_human_demo = True
    # FLAGS.game_config_fname = 'huazhe.xml'


def torcs_human_all_V_grounding_demo_stage_1_yang_new():
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

    FLAGS.core_num = '4'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.can_deal_with_human_demo = True
    # FLAGS.game_config_fname = 'huazhe.xml'


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
    FLAGS.can_deal_with_human_demo = True
    # FLAGS.game_config_fname = 'huazhe.xml'


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
    # FLAGS.exp_value_critic_weighting = 1.0
    # FLAGS.exp_policy_grad_weighting = 1.0
    # FLAGS.critic_use_rapid_weighting = False

    # FLAGS.disable_off_policy_weighting = True
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.can_deal_with_human_demo = True
    # FLAGS.game_config_fname = 'huazhe.xml'


def torcs_human_all_soft_Q_demo_stage_1_yang_new_PCL(index):
    tag = inspect.stack()[0][3]
    print("config name=", tag)
    FLAGS.torcs_divider = 30
    # FLAGS.method_name = tag + str(FLAGS.torcs_divider)
    common_setting()
    FLAGS.demo_file_path = '/data2/hxu/modelRL/300000add_damage.p'  # parse_name('/data/hxu/modelRL/human_yang.txt')
    torcs_dqn_kx(FLAGS.torcs_divider, tag)

    FLAGS.torcs_path = '/data2/yang/code/rlTORCS'

    FLAGS.demo_mode = "replay"
    FLAGS.collect_Q_experience = False
    FLAGS.eval_freq = 10000

    FLAGS.core_num = '0'
    FLAGS.soft_Q_loss_weight = 1.0
    FLAGS.pi_v_model = True
    # FLAGS.exp_value_critic_weighting = 1.0
    # FLAGS.exp_policy_grad_weighting = 1.0
    # FLAGS.critic_use_rapid_weighting = False

    # FLAGS.disable_off_policy_weighting = True
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.torcs_demo = True
    FLAGS.method_name = tag + '_' + index
    # FLAGS.can_deal_with_human_demo = True
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
    FLAGS.can_deal_with_human_demo = True
    # FLAGS.game_config_fname = 'huazhe.xml'


def torcs_human_all_hard_Q_demo_stage_1_yang_new_second():
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

    FLAGS.core_num = '2'
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.can_deal_with_human_demo = True
    # FLAGS.game_config_fname = 'huazhe.xml'


def torcs_human_all_dqfd_full_demo_stage_1_yang_new_second():
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

    FLAGS.core_num = '3'

    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.l2_regularization_loss_weight = 1.0e-5
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.can_deal_with_human_demo = True
    # FLAGS.game_config_fname = 'huazhe.xml'


def torcs_human_all_V_grounding_demo_stage_1_yang_new_second():
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

    FLAGS.core_num = '4'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.can_deal_with_human_demo = True
    # FLAGS.game_config_fname = 'huazhe.xml'


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
    FLAGS.can_deal_with_human_demo = True
    # FLAGS.game_config_fname = 'huazhe.xml'


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
    # FLAGS.exp_value_critic_weighting = 1.0
    # FLAGS.exp_policy_grad_weighting = 1.0
    # FLAGS.critic_use_rapid_weighting = False

    # FLAGS.disable_off_policy_weighting = True
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.can_deal_with_human_demo = True
    # FLAGS.game_config_fname = 'huazhe.xml'





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
    FLAGS.can_deal_with_human_demo = True
    # FLAGS.game_config_fname = 'huazhe.xml'
    # FLAGS.weight_decay = True
    # FLAGS.WEIGHT_DECAY_FACTOR = num
    FLAGS.l2_regularization_loss_weight = num
    FLAGS.method_name = tag + '_' + str(num)


def torcs_human_all_hard_Q_demo_stage_1_hxu_slow_decay(num):
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

    FLAGS.core_num = '2'
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.can_deal_with_human_demo = True
    # FLAGS.game_config_fname = 'huazhe.xml'
    # FLAGS.weight_decay = True
    # FLAGS.WEIGHT_DECAY_FACTOR = num
    FLAGS.l2_regularization_loss_weight = num
    FLAGS.method_name = tag + '_' + str(num)



def torcs_human_all_dqfd_full_demo_stage_1_hxu_slow_val():
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

    FLAGS.core_num = '3'

    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.l2_regularization_loss_weight = 1.0e-5
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.can_deal_with_human_demo = True
    # FLAGS.game_config_fname = 'huazhe.xml'
    FLAGS.val_set = True
    FLAGS.val_set_file = parse_name('/data/hxu/modelRL/val_set.txt')


def torcs_human_all_V_grounding_demo_stage_1_hxu_slow_val():
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

    FLAGS.core_num = '4'

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.can_deal_with_human_demo = True
    # FLAGS.game_config_fname = 'huazhe.xml'
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
    FLAGS.can_deal_with_human_demo = True
    # FLAGS.game_config_fname = 'huazhe.xml'
    FLAGS.val_set = True
    FLAGS.val_set_file = parse_name('/data/hxu/modelRL/val_set.txt')





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
    FLAGS.can_deal_with_human_demo = True
    # FLAGS.game_config_fname = 'huazhe.xml'


def torcs_human_all_hard_Q_demo_stage_1_hxu_slow_all():
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

    FLAGS.core_num = '1'
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.can_deal_with_human_demo = True
    # FLAGS.game_config_fname = 'huazhe.xml'


def torcs_human_all_dqfd_full_demo_stage_1_hxu_slow_all():
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

    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.l2_regularization_loss_weight = 1.0e-5
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.can_deal_with_human_demo = True
    # FLAGS.game_config_fname = 'huazhe.xml'


def torcs_human_all_V_grounding_demo_stage_1_hxu_slow_all():
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

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.critic_use_rapid_weighting = False
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.can_deal_with_human_demo = True
    # FLAGS.game_config_fname = 'huazhe.xml'


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
    FLAGS.can_deal_with_human_demo = True
    # FLAGS.game_config_fname = 'huazhe.xml'


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
    # FLAGS.exp_value_critic_weighting = 1.0
    # FLAGS.exp_policy_grad_weighting = 1.0
    # FLAGS.critic_use_rapid_weighting = False

    # FLAGS.disable_off_policy_weighting = True
    FLAGS.custom_reward = 'reward_ben'
    common_data_size(300000)
    FLAGS.autoback = True
    FLAGS.can_deal_with_human_demo = True
    # FLAGS.game_config_fname = 'huazhe.xml'


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
    FLAGS.can_deal_with_human_demo = True
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
    FLAGS.can_deal_with_human_demo = True
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
    FLAGS.can_deal_with_human_demo = True
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
    # FLAGS.game_config_fname = 'huazhe.xml'


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
    # FLAGS.game_config_fname = 'huazhe.xml'

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
