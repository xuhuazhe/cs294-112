import inspect, sys, os, datetime
import tensorflow as tf
FLAGS = tf.app.flags.FLAGS
from dqn_utils import *

# resources list
TORCS_PATH = '/data/yang/code/rlTORCS'
CKPT_BASE = "/home/boyuan/Projects/cs294-112/hw3/link_data"
TORCS_HUMAN_DEMO_LIST = '/shared/hxu/data1/modelRL/hxu_slow_all.txt'
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

    FLAGS.torcs_divider = 1
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

def dqn_alien():
    tag = inspect.stack()[0][3]
    stage = "from_scratch"
    set_method_name(stage, tag)
    set_unconditioned()
    FLAGS.env_id = "AlienNoFrameskip-v4"
    set_stage(stage)
    set_eval_mode(False)
    FLAGS.supervise_cross_entropy_loss_weight = 0.0
    FLAGS.exp_value_critic_weighting = 0.0
    FLAGS.exp_policy_grad_weighting = 0.0
    FLAGS.tiny_explore = 0.01
    FLAGS.batch_size = 32
    FLAGS.method_name = "dqn_alien"
    FLAGS.core_num = "0"
    #Q learning specific
    FLAGS.eval_freq = -1
    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000
    FLAGS.hard_Q_loss_weight = 1.0

def dqn_hero():
    tag = inspect.stack()[0][3]
    stage = "from_scratch"
    set_method_name(stage, tag)
    set_unconditioned()
    FLAGS.env_id = "HeroNoFrameskip-v4"
    set_stage(stage)
    set_eval_mode(False)
    FLAGS.supervise_cross_entropy_loss_weight = 0.0
    FLAGS.exp_value_critic_weighting = 0.0
    FLAGS.exp_policy_grad_weighting = 0.0
    FLAGS.tiny_explore = 0.01
    FLAGS.batch_size = 32
    FLAGS.method_name = "dqn_hero"
    FLAGS.core_num = "0"
    #Q learning specific
    FLAGS.eval_freq = -1
    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000
    FLAGS.hard_Q_loss_weight = 1.0

def dqn_boxing():
    tag = inspect.stack()[0][3]
    stage = "from_scratch"
    set_method_name(stage, tag)
    set_unconditioned()
    FLAGS.env_id = "BoxingNoFrameskip-v4"
    set_stage(stage)
    set_eval_mode(False)
    FLAGS.supervise_cross_entropy_loss_weight = 0.0
    FLAGS.exp_value_critic_weighting = 0.0
    FLAGS.exp_policy_grad_weighting = 0.0
    FLAGS.tiny_explore = 0.01
    FLAGS.batch_size = 32
    FLAGS.method_name = "dqn_boxing"
    FLAGS.core_num = "1"
    #Q learning specific
    FLAGS.eval_freq = -1
    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000
    FLAGS.hard_Q_loss_weight = 1.0

def dqn_asterix():
    tag = inspect.stack()[0][3]
    stage = "from_scratch"
    set_method_name(stage, tag)
    set_unconditioned()
    FLAGS.env_id = "AsterixNoFrameskip-v4"
    set_stage(stage)
    set_eval_mode(False)
    FLAGS.supervise_cross_entropy_loss_weight = 0.0
    FLAGS.exp_value_critic_weighting = 0.0
    FLAGS.exp_policy_grad_weighting = 0.0
    FLAGS.tiny_explore = 0.01
    FLAGS.batch_size = 32
    FLAGS.method_name = "dqn_asterix"
    FLAGS.core_num = "1"
    #Q learning specific
    FLAGS.eval_freq = -1
    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000
    FLAGS.hard_Q_loss_weight = 1.0

def dqn_jamesbond():
    tag = inspect.stack()[0][3]
    stage = "from_scratch"
    set_method_name(stage, tag)
    set_unconditioned()
    FLAGS.env_id = "JamesbondNoFrameskip-v4"
    set_stage(stage)
    set_eval_mode(False)
    FLAGS.supervise_cross_entropy_loss_weight = 0.0
    FLAGS.exp_value_critic_weighting = 0.0
    FLAGS.exp_policy_grad_weighting = 0.0
    FLAGS.tiny_explore = 0.01
    FLAGS.batch_size = 32
    FLAGS.method_name = "dqn_jamesbond"
    FLAGS.core_num = "2"
    #Q learning specific
    FLAGS.eval_freq = -1
    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000
    FLAGS.hard_Q_loss_weight = 1.0

def dqn_freeway():
    tag = inspect.stack()[0][3]
    stage = "from_scratch"
    set_method_name(stage, tag)
    set_unconditioned()
    FLAGS.env_id = "FreewayNoFrameskip-v4"
    set_stage(stage)
    set_eval_mode(False)
    FLAGS.supervise_cross_entropy_loss_weight = 0.0
    FLAGS.exp_value_critic_weighting = 0.0
    FLAGS.exp_policy_grad_weighting = 0.0
    FLAGS.tiny_explore = 0.01
    FLAGS.batch_size = 32
    FLAGS.method_name = "dqn_freeway"
    FLAGS.core_num = "2"
    #Q learning specific
    FLAGS.eval_freq = -1
    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000
    FLAGS.hard_Q_loss_weight = 1.0

def dqn_icehockey():
    tag = inspect.stack()[0][3]
    stage = "from_scratch"
    set_method_name(stage, tag)
    set_unconditioned()
    FLAGS.env_id = "IceHockeyNoFrameskip-v4"
    set_stage(stage)
    set_eval_mode(False)
    FLAGS.supervise_cross_entropy_loss_weight = 0.0
    FLAGS.exp_value_critic_weighting = 0.0
    FLAGS.exp_policy_grad_weighting = 0.0
    FLAGS.tiny_explore = 0.01
    FLAGS.batch_size = 32
    FLAGS.method_name = "dqn_icehockey"
    FLAGS.core_num = "3"
    #Q learning specific
    FLAGS.eval_freq = -1
    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000
    FLAGS.hard_Q_loss_weight = 1.0

def dqn_privateeye():
    tag = inspect.stack()[0][3]
    stage = "from_scratch"
    set_method_name(stage, tag)
    set_unconditioned()
    FLAGS.env_id = "PrivateEyeNoFrameskip-v4"
    set_stage(stage)
    set_eval_mode(False)
    FLAGS.supervise_cross_entropy_loss_weight = 0.0
    FLAGS.exp_value_critic_weighting = 0.0
    FLAGS.exp_policy_grad_weighting = 0.0
    FLAGS.tiny_explore = 0.01
    FLAGS.batch_size = 32
    FLAGS.method_name = "dqn_privateeye"
    FLAGS.core_num = "3"
    #Q learning specific
    FLAGS.eval_freq = -1
    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000
    FLAGS.hard_Q_loss_weight = 1.0

def dqn_atlantis():
    tag = inspect.stack()[0][3]
    stage = "from_scratch"
    set_method_name(stage, tag)
    set_unconditioned()
    FLAGS.env_id = "AtlantisNoFrameskip-v4"
    set_stage(stage)
    set_eval_mode(False)
    FLAGS.supervise_cross_entropy_loss_weight = 0.0
    FLAGS.exp_value_critic_weighting = 0.0
    FLAGS.exp_policy_grad_weighting = 0.0
    FLAGS.tiny_explore = 0.01
    FLAGS.batch_size = 32
    FLAGS.method_name = "dqn_atlantis"
    FLAGS.core_num = "4"

    #Q learning specific
    FLAGS.eval_freq = -1
    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000
    FLAGS.hard_Q_loss_weight = 1.0

def dqn_enduro():
    tag = inspect.stack()[0][3]
    stage = "from_scratch"
    set_method_name(stage, tag)
    set_unconditioned()
    FLAGS.env_id = "EnduroNoFrameskip-v4"
    set_stage(stage)
    set_eval_mode(False)
    FLAGS.supervise_cross_entropy_loss_weight = 0.0
    FLAGS.exp_value_critic_weighting = 0.0
    FLAGS.exp_policy_grad_weighting = 0.0
    FLAGS.tiny_explore = 0.01
    FLAGS.batch_size = 32
    FLAGS.method_name = "dqn_enduro"
    FLAGS.core_num = "4"

    #Q learning specific
    FLAGS.eval_freq = -1
    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000
    FLAGS.hard_Q_loss_weight = 1.0

def dqn_asterix_collect():
    tag = inspect.stack()[0][3]
    stage = "stage2"
    set_method_name(stage, tag)
    set_unconditioned()
    FLAGS.env_id = "AsterixNoFrameskip-v4"
    set_stage(stage)
    set_eval_mode(True)

    FLAGS.supervise_cross_entropy_loss_weight = 0.0
    FLAGS.exp_value_critic_weighting = 0.0
    FLAGS.exp_policy_grad_weighting = 0.0
    FLAGS.tiny_explore = 0.01
    FLAGS.batch_size = 32
    FLAGS.method_name = "dqn_asterix_collect"
    FLAGS.core_num = "0"

    #Q learning specific
    FLAGS.eval_freq = -1
    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000
    FLAGS.hard_Q_loss_weight = 1.0

    # collect specific
    FLAGS.learning_stage = False
    FLAGS.exploration_schedule = ConstantSchedule(0.01)
    FLAGS.ckpt_path = os.path.join(CKPT_BASE, 'dqn_asterix')
    FLAGS.bad_type = "random"
    FLAGS.lf_bad_data = True
    FLAGS.final_bad_portion = 30 #50 # 70
    FLAGS.demo_name = 'dqn_asterix_collect' + '_' + str(FLAGS.final_bad_portion)

def dqn_atlantis_collect():
    tag = inspect.stack()[0][3]
    stage = "stage2"
    set_method_name(stage, tag)
    set_unconditioned()
    FLAGS.env_id = "AtlantisNoFrameskip-v4"
    set_stage(stage)
    set_eval_mode(True)

    FLAGS.supervise_cross_entropy_loss_weight = 0.0
    FLAGS.exp_value_critic_weighting = 0.0
    FLAGS.exp_policy_grad_weighting = 0.0
    FLAGS.tiny_explore = 0.01
    FLAGS.batch_size = 32
    FLAGS.method_name = "dqn_atlantis_collect"
    FLAGS.core_num = "0"

    #Q learning specific
    FLAGS.eval_freq = -1
    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000
    FLAGS.hard_Q_loss_weight = 1.0

    # collect specific
    FLAGS.learning_stage = False
    FLAGS.exploration_schedule = ConstantSchedule(0.01)
    FLAGS.ckpt_path = os.path.join(CKPT_BASE, 'dqn_atlantis')
    FLAGS.bad_type = "random"
    FLAGS.lf_bad_data = True
    FLAGS.final_bad_portion = 30 #50 # 70
    FLAGS.demo_name = 'dqn_atlantis_collect' + '_' + str(FLAGS.final_bad_portion)

def dqn_boxing_collect():
    tag = inspect.stack()[0][3]
    stage = "stage2"
    set_method_name(stage, tag)
    set_unconditioned()
    FLAGS.env_id = "BoxingNoFrameskip-v4"
    set_stage(stage)
    set_eval_mode(True)

    FLAGS.supervise_cross_entropy_loss_weight = 0.0
    FLAGS.exp_value_critic_weighting = 0.0
    FLAGS.exp_policy_grad_weighting = 0.0
    FLAGS.tiny_explore = 0.01
    FLAGS.batch_size = 32
    FLAGS.method_name = "dqn_boxing_collect"
    FLAGS.core_num = "1"

    #Q learning specific
    FLAGS.eval_freq = -1
    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000
    FLAGS.hard_Q_loss_weight = 1.0

    # collect specific
    FLAGS.learning_stage = False
    FLAGS.exploration_schedule = ConstantSchedule(0.01)
    FLAGS.ckpt_path = os.path.join(CKPT_BASE, 'dqn_boxing')
    FLAGS.bad_type = "random"
    FLAGS.lf_bad_data = True
    FLAGS.final_bad_portion = 30 #50 # 70
    FLAGS.demo_name = 'dqn_boxing_collect' + '_' + str(FLAGS.final_bad_portion)

def dqn_enduro_collect():
    tag = inspect.stack()[0][3]
    stage = "stage2"
    set_method_name(stage, tag)
    set_unconditioned()
    FLAGS.env_id = "EnduroNoFrameskip-v4"
    set_stage(stage)
    set_eval_mode(True)

    FLAGS.supervise_cross_entropy_loss_weight = 0.0
    FLAGS.exp_value_critic_weighting = 0.0
    FLAGS.exp_policy_grad_weighting = 0.0
    FLAGS.tiny_explore = 0.01
    FLAGS.batch_size = 32
    FLAGS.method_name = "dqn_enduro_collect"
    FLAGS.core_num = "1"

    #Q learning specific
    FLAGS.eval_freq = -1
    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000
    FLAGS.hard_Q_loss_weight = 1.0

    # collect specific
    FLAGS.learning_stage = False
    FLAGS.exploration_schedule = ConstantSchedule(0.01)
    FLAGS.ckpt_path = os.path.join(CKPT_BASE, 'dqn_enduro')
    FLAGS.bad_type = "random"
    FLAGS.lf_bad_data = True
    FLAGS.final_bad_portion = 30 #50 # 70
    FLAGS.demo_name = 'dqn_enduro_collect' + '_' + str(FLAGS.final_bad_portion)

def dqn_freeway_collect():
    tag = inspect.stack()[0][3]
    stage = "stage2"
    set_method_name(stage, tag)
    set_unconditioned()
    FLAGS.env_id = "FreewayNoFrameskip-v4"
    set_stage(stage)
    set_eval_mode(True)

    FLAGS.supervise_cross_entropy_loss_weight = 0.0
    FLAGS.exp_value_critic_weighting = 0.0
    FLAGS.exp_policy_grad_weighting = 0.0
    FLAGS.tiny_explore = 0.01
    FLAGS.batch_size = 32
    FLAGS.method_name = "dqn_freeway_collect"
    FLAGS.core_num = "2"

    #Q learning specific
    FLAGS.eval_freq = -1
    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000
    FLAGS.hard_Q_loss_weight = 1.0

    # collect specific
    FLAGS.learning_stage = False
    FLAGS.exploration_schedule = ConstantSchedule(0.01)
    FLAGS.ckpt_path = os.path.join(CKPT_BASE, 'dqn_freeway')
    FLAGS.bad_type = "random"
    FLAGS.lf_bad_data = True
    FLAGS.final_bad_portion = 30 #50 # 70
    FLAGS.demo_name = 'dqn_freeway_collect' + '_' + str(FLAGS.final_bad_portion)

def dqn_alien_collect():
    tag = inspect.stack()[0][3]
    stage = "stage2"
    set_method_name(stage, tag)
    set_unconditioned()
    FLAGS.env_id = "AlienNoFrameskip-v4"
    set_stage(stage)
    set_eval_mode(True)

    FLAGS.supervise_cross_entropy_loss_weight = 0.0
    FLAGS.exp_value_critic_weighting = 0.0
    FLAGS.exp_policy_grad_weighting = 0.0
    FLAGS.tiny_explore = 0.01
    FLAGS.batch_size = 32
    FLAGS.method_name = "dqn_alien_collect"
    FLAGS.core_num = "2"

    #Q learning specific
    FLAGS.eval_freq = -1
    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000
    FLAGS.hard_Q_loss_weight = 1.0

    # collect specific
    FLAGS.learning_stage = False
    FLAGS.exploration_schedule = ConstantSchedule(0.01)
    FLAGS.ckpt_path = os.path.join(CKPT_BASE, 'dqn_alien')
    FLAGS.bad_type = "random"
    FLAGS.lf_bad_data = True
    FLAGS.final_bad_portion = 30 #50 # 70
    FLAGS.demo_name = 'dqn_alien_collect' + '_' + str(FLAGS.final_bad_portion)

def dqn_hero_collect():
    tag = inspect.stack()[0][3]
    stage = "stage2"
    set_method_name(stage, tag)
    set_unconditioned()
    FLAGS.env_id = "HeroNoFrameskip-v4"
    set_stage(stage)
    set_eval_mode(True)

    FLAGS.supervise_cross_entropy_loss_weight = 0.0
    FLAGS.exp_value_critic_weighting = 0.0
    FLAGS.exp_policy_grad_weighting = 0.0
    FLAGS.tiny_explore = 0.01
    FLAGS.batch_size = 32
    FLAGS.method_name = "dqn_hero_collect"
    FLAGS.core_num = "2"

    #Q learning specific
    FLAGS.eval_freq = -1
    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000
    FLAGS.hard_Q_loss_weight = 1.0

    # collect specific
    FLAGS.learning_stage = False
    FLAGS.exploration_schedule = ConstantSchedule(0.01)
    FLAGS.ckpt_path = os.path.join(CKPT_BASE, 'dqn_hero')
    FLAGS.bad_type = "random"
    FLAGS.lf_bad_data = True
    FLAGS.final_bad_portion = 30 #50 # 70
    FLAGS.demo_name = 'dqn_hero_collect' + '_' + str(FLAGS.final_bad_portion)

def dqn_icehockey_collect():
    tag = inspect.stack()[0][3]
    stage = "stage2"
    set_method_name(stage, tag)
    set_unconditioned()
    FLAGS.env_id = "IceHockeyNoFrameskip-v4"
    set_stage(stage)
    set_eval_mode(True)

    FLAGS.supervise_cross_entropy_loss_weight = 0.0
    FLAGS.exp_value_critic_weighting = 0.0
    FLAGS.exp_policy_grad_weighting = 0.0
    FLAGS.tiny_explore = 0.01
    FLAGS.batch_size = 32
    FLAGS.method_name = "dqn_icehockey_collect"
    FLAGS.core_num = "3"

    #Q learning specific
    FLAGS.eval_freq = -1
    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000
    FLAGS.hard_Q_loss_weight = 1.0

    # collect specific
    FLAGS.learning_stage = False
    FLAGS.exploration_schedule = ConstantSchedule(0.01)
    FLAGS.ckpt_path = os.path.join(CKPT_BASE, 'dqn_icehockey')
    FLAGS.bad_type = "random"
    FLAGS.lf_bad_data = True
    FLAGS.final_bad_portion = 30 #50 # 70
    FLAGS.demo_name = 'dqn_icehockey_collect' + '_' + str(FLAGS.final_bad_portion)

def dqn_jamesbond_collect():
    tag = inspect.stack()[0][3]
    stage = "stage2"
    set_method_name(stage, tag)
    set_unconditioned()
    FLAGS.env_id = "JamesbondNoFrameskip-v4"
    set_stage(stage)
    set_eval_mode(True)

    FLAGS.supervise_cross_entropy_loss_weight = 0.0
    FLAGS.exp_value_critic_weighting = 0.0
    FLAGS.exp_policy_grad_weighting = 0.0
    FLAGS.tiny_explore = 0.01
    FLAGS.batch_size = 32
    FLAGS.method_name = "dqn_jamesbond_collect"
    FLAGS.core_num = "3"

    #Q learning specific
    FLAGS.eval_freq = -1
    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000
    FLAGS.hard_Q_loss_weight = 1.0

    # collect specific
    FLAGS.learning_stage = False
    FLAGS.exploration_schedule = ConstantSchedule(0.01)
    FLAGS.ckpt_path = os.path.join(CKPT_BASE, 'dqn_jamesbond')
    FLAGS.bad_type = "random"
    FLAGS.lf_bad_data = True
    FLAGS.final_bad_portion = 30 #50 # 70
    FLAGS.demo_name = 'dqn_jamesbond_collect' + '_' + str(FLAGS.final_bad_portion)

def dqn_privateeye_collect():
    tag = inspect.stack()[0][3]
    stage = "stage2"
    set_method_name(stage, tag)
    set_unconditioned()
    FLAGS.env_id = "PrivateEyeNoFrameskip-v4"
    set_stage(stage)
    set_eval_mode(True)

    FLAGS.supervise_cross_entropy_loss_weight = 0.0
    FLAGS.exp_value_critic_weighting = 0.0
    FLAGS.exp_policy_grad_weighting = 0.0
    FLAGS.tiny_explore = 0.01
    FLAGS.batch_size = 32
    FLAGS.method_name = "dqn_privateeye_collect"
    FLAGS.core_num = "4"

    #Q learning specific
    FLAGS.eval_freq = -1
    FLAGS.demo_mode = "no_demo"
    FLAGS.collect_Q_experience = True
    FLAGS.learning_starts = 50000
    FLAGS.hard_Q_loss_weight = 1.0

    # collect specific
    FLAGS.learning_stage = False
    FLAGS.exploration_schedule = ConstantSchedule(0.01)
    FLAGS.ckpt_path = os.path.join(CKPT_BASE, 'dqn_privateeye')
    FLAGS.bad_type = "random"
    FLAGS.lf_bad_data = True
    FLAGS.final_bad_portion = 30 #50 # 70
    FLAGS.demo_name = 'dqn_privateeye_collect' + '_' + str(FLAGS.final_bad_portion)
