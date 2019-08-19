import inspect, sys, os, datetime
import tensorflow as tf
FLAGS = tf.app.flags.FLAGS
from dqn_utils import *

# resources list
TORCS_PATH = '/data/yang/code/rlTORCS'
CKPT_BASE = "/data/yang/code/rl_demonstration/hw3/link_data"
TORCS_HUMAN_DEMO_LIST = '/data/hxu/modelRL/hxu_slow_all.txt'
TORCS_MACHINE_DEMO = '/data2/hxu/modelRL/300000autoback_simple_08_25.p'
# end of resource list

# hierachy of flags, group them into several smaller functionality
# each method takes in previous set flags, instead of use them directly, to be clear about the dependency order
def set_unconditioned():
    # Torcs related
    # TODO: resume how to use torcs
    FLAGS.torcs_path = TORCS_PATH
    FLAGS.custom_reward = 'reward_ben'
    FLAGS.autoback = True
    # fix the human demonstration of torcs data loading problem, no other usage, should always set to True if we are using torcs
    FLAGS.can_deal_with_human_demo = True

    FLAGS.torcs_divider = 30
    divider = FLAGS.torcs_divider

    # in data collection stage, it should be False
    FLAGS.learning_stage = True
    FLAGS.batch_size = 32
    FLAGS.tiny_explore = 0.01

    FLAGS.env_id = "rltorcs-v0"

    # begin the divider attempt
    FLAGS.target_update_freq = 10000 / divider
    FLAGS.replay_buffer_size = int(1e6)
    FLAGS.max_timesteps = 2 * int(4e7) / divider
    FLAGS.dataset_size = int(3e5)

    # interaction purpose
    FLAGS.summary_interval = 10000 / divider


# three types: learn from demo, finetune, and learning from scratch
def set_stage(stage,
              ckpt_base=CKPT_BASE):
    # this depends on FLAGS.torcs_divider, FLAGS.method_name
    assert (FLAGS.torcs_divider > 0)

    # learning rate for stage1
    num_iterations = int(4e7) / 4 / FLAGS.torcs_divider # 3.3E5
    FLAGS.lr_schedule = PiecewiseSchedule([
        (0, 1e-4),
        (num_iterations / 10, 1e-4), # 3.3E4
        (num_iterations / 2, 5e-5)], # 1.7E5
        outside_value=5e-5)

    if stage == "stage1":
        # since we are learning from demo only, we don't need the interaction
        FLAGS.collect_Q_experience = False
        # since torcs only has one instance, we can evaluate in the learning from demo phase, but not from env phase
        FLAGS.eval_freq = 10000
        # this sets train env=None, and test env=one env
        FLAGS.torcs_demo = True
        # 0 if on demonstration, 50000/divider if from the environment.
        FLAGS.learning_starts = 0

    elif stage == "stage2" or stage == "from_scratch":
        FLAGS.collect_Q_experience = True
        FLAGS.eval_freq = -1
        FLAGS.torcs_demo = False
        FLAGS.learning_starts = 50000 / FLAGS.torcs_divider # 1.7E3

        if stage == "stage2":
            FLAGS.exploration_schedule = ConstantSchedule(0.01)
            # TODO: fine tune stage does not have a small learning rate, this might be a problem
            method_name = get_method_name_without_stage()
            FLAGS.ckpt_path = ckpt_base + "/" + method_name + "_stage1"
            # the only use of this flag is to load the checkpoint
            FLAGS.inenv_finetune = True
        else:
            FLAGS.exploration_schedule = PiecewiseSchedule([
                (0, 1.0),
                (1e6 / FLAGS.torcs_divider, 0.1),
                (num_iterations / 2, 0.01)],
                outside_value=0.01)
            FLAGS.inenv_finetune = False
    else:
        raise

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

def set_demo_type(type):
    if type == "human":
        FLAGS.demo_mode = 'hdf'
        FLAGS.demo_file_path = parse_name(TORCS_HUMAN_DEMO_LIST)
    elif type == "machine":
        FLAGS.demo_mode = 'replay'
        FLAGS.demo_file_path = TORCS_MACHINE_DEMO
    elif type == "no":
        FLAGS.demo_mode = 'no_demo'
        FLAGS.demo_file_path = ''
    else:
        raise
    # TODO: dqfd case is not handled

def set_eval_mode(eval_mode, ckpt_base=CKPT_BASE):
    # what does eval mode mean?
    # we finished training a model, and we want to eval it in the environment, without modifying it
    # This depends on FLAGS.method_name with stage tag

    if eval_mode:
        # this load the pretrained model, and disable training process and break after we eval once.
        FLAGS.inenv_eval = True
        assert "__" in FLAGS.method_name
        FLAGS.ckpt_path = ckpt_base + "/" + FLAGS.method_name
        # this trigger in env eval immediately
        FLAGS.eval_freq = 1

        # previously, inenv_eval is only used with stage1 training, so below is not needed
        # but to make it work with even broader context, we make sure that it changes the following things
        FLAGS.collect_Q_experience = False
        FLAGS.torcs_demo = True
    else:
        FLAGS.inenv_eval = False

def set_method_name(stage, expid):
    # each experiment is uniquely defined by the original method name
    # however, it has two stages, and stored in two different folders
    # this function complete the folder name by appending the stage name

    # the stage is always the last and seperated with the double _ marker: __
    FLAGS.method_name = expid + "__" + stage

def get_method_name_without_stage():
    whole = FLAGS.method_name
    assert "__" in whole
    x = whole.split("__")
    assert(len(x) == 2)
    return x[0]

def torcs_human_sal():
    # corresponde to torcs_human_all_V_grounding_no_weighting_demo_stage_1_hxu_slow_final
    tag = inspect.stack()[0][3]

    stage = "stage1"

    set_method_name(stage, tag)
    set_unconditioned()
    set_stage(stage)
    set_demo_type("human")
    set_eval_mode(False)

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.core_num = "0"

def torcs_human_cross_entropy():
    # torcs_human_all_cross_entropy_demo_stage_1_hxu_slow_final
    tag = inspect.stack()[0][3]

    stage = "stage1"

    set_method_name(stage, tag)
    set_unconditioned()
    set_stage(stage)
    set_demo_type("human")
    set_eval_mode(False)

    FLAGS.supervise_cross_entropy_loss_weight = 1.0
    FLAGS.core_num = "0"

def torcs_human_dqfd():
    # torcs_human_all_dqfd_full_demo_stage_1_hxu_slow_final
    tag = inspect.stack()[0][3]

    stage = "stage1"

    set_method_name(stage, tag)
    set_unconditioned()
    set_stage(stage)
    set_demo_type("human")
    set_eval_mode(False)

    FLAGS.supervise_hinge_DQfD_loss_weight = 1.0
    FLAGS.hard_Q_loss_weight = 1.0
    FLAGS.l2_regularization_loss_weight = 1.0e-5
    FLAGS.core_num = "0"


# below starts the setting of Atari primitives
ENDURO_PATH = '/shared/hxu/data1/modelRL/300000enduro_perfect'

# hierachy of flags, group them into several smaller functionality
# each method takes in previous set flags, instead of use them directly, to be clear about the dependency order
def set_unconditioned_atari(env, divider_in=1):
    # diff with set_unconditioned_atari:
    # remove those FLAGS: torcs_path, custom_reward, autoback
    # set torcs_divider to 1, can_deal_with_human_demo to False

    # fix the human demonstration of torcs data loading problem, no other usage, should always set to True if we are using torcs
    FLAGS.can_deal_with_human_demo = False

    # TODO: set an appropriate divider for atari
    FLAGS.torcs_divider = divider_in
    divider = FLAGS.torcs_divider

    # in data collection stage, it should be False
    FLAGS.learning_stage = True
    FLAGS.batch_size = 32
    FLAGS.tiny_explore = 0.01

    FLAGS.env_id = env+"NoFrameskip-v4"

    # begin the divider attempt
    FLAGS.target_update_freq = 10000 / divider
    FLAGS.replay_buffer_size = int(1e6)
    FLAGS.max_timesteps = 2 * int(4e7) / divider
    FLAGS.dataset_size = int(3e5)

    # interaction purpose
    FLAGS.summary_interval = 10000 / divider

def set_stage_atari(stage,
              ckpt_base=CKPT_BASE):
    set_stage(stage, ckpt_base)
    # we are using atari, so we can always spawn two env, one for train and one for test. So we will always eval in the
    # environment. That sets eval_freq to positive number. torcs_demo=True will set the eval env to Torcs and train env
    # to None, which should not be our case.
    FLAGS.eval_freq = 10000
    FLAGS.torcs_demo = False

def set_demo_type_atari(type, env):
    if type == "human":
        FLAGS.demo_mode = 'hdf'
        raise
        # we haven't figure out which old file is for enduro human demo yet
        FLAGS.demo_file_path = parse_name(TORCS_HUMAN_DEMO_LIST)
    elif type == "machine":
        FLAGS.demo_mode = 'replay'
        if env == "Enduro":
            FLAGS.demo_file_path = ENDURO_PATH
        else:
            raise
    elif type == "no":
        FLAGS.demo_mode = 'no_demo'
        FLAGS.demo_file_path = ''
    else:
        raise
    # TODO: dqfd case is not handled

def set_eval_mode_atari(eval_mode, ckpt_base=CKPT_BASE):
    set_eval_mode(eval_mode, ckpt_base=CKPT_BASE)
    # we want the evaluation happen in a separate environment since we are using atari
    FLAGS.torcs_demo = False


def enduro_perfectdemo_cross_entropy():
    # what's different from huazhe's implementation
    # FLAGS.can_deal_with_human_demo = True
    # FLAGS.torcs_divider = 30
    # FLAGS.eval_frequency
    tag = inspect.stack()[0][3]

    stage = "stage1"

    set_method_name(stage, tag)
    set_unconditioned_atari("Enduro")
    set_stage_atari(stage)
    set_demo_type_atari("machine", "Enduro")
    set_eval_mode_atari(False)

    FLAGS.supervise_cross_entropy_loss_weight = 1.0
    FLAGS.core_num = "0"


def enduro_perfectdemo_sal():
    tag = inspect.stack()[0][3]

    stage = "stage1"

    set_method_name(stage, tag)
    set_unconditioned_atari("Enduro")
    set_stage_atari(stage)
    set_demo_type_atari("machine", "Enduro")
    set_eval_mode_atari(False)

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.core_num = "0"

def enduro_perfectdemo_sal_divider5():
    tag = inspect.stack()[0][3]

    stage = "stage1"

    set_method_name(stage, tag)
    set_unconditioned_atari("Enduro", divider_in=5)
    set_stage_atari(stage)
    set_demo_type_atari("machine", "Enduro")
    set_eval_mode_atari(False)

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.core_num = "0"

def enduro_perfectdemo_sal_divider5_vizbellmanerror():
    # use the soft Q loss to compute the bellman error
    tag = inspect.stack()[0][3]

    stage = "stage1"

    set_method_name(stage, tag)
    set_unconditioned_atari("Enduro", divider_in=5)
    set_stage_atari(stage)
    set_demo_type_atari("machine", "Enduro")
    set_eval_mode_atari(False)

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.core_num = "0"

    FLAGS.soft_Q_loss_weight = 0.00001

def enduro_perfectdemo_sal_maxclip():
    # use the soft Q loss to compute the bellman error
    tag = inspect.stack()[0][3]

    stage = "stage1"

    set_method_name(stage, tag)
    set_unconditioned_atari("Enduro", divider_in=5)
    set_stage_atari(stage)
    set_demo_type_atari("machine", "Enduro")
    set_eval_mode_atari(False)

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.core_num = "0"

    FLAGS.soft_Q_loss_weight = 0.00001
    FLAGS.exp_policy_grad_weighting_maxclip = 1.0


def enduro_perfectdemo_sal_minclip():
    # use the soft Q loss to compute the bellman error
    tag = inspect.stack()[0][3]

    stage = "stage1"

    set_method_name(stage, tag)
    set_unconditioned_atari("Enduro", divider_in=5)
    set_stage_atari(stage)
    set_demo_type_atari("machine", "Enduro")
    set_eval_mode_atari(False)

    FLAGS.exp_value_critic_weighting = 1.0
    FLAGS.exp_policy_grad_weighting = 1.0
    FLAGS.core_num = "0"

    FLAGS.soft_Q_loss_weight = 0.00001
    FLAGS.exp_policy_grad_weighting_minclip = 1.0
