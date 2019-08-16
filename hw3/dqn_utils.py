"""This file includes a collection of utility functions that are useful for
implementing DQN."""
import gym
import tensorflow as tf
import numpy as np
import random
import pickle
from atari_wrappers import *
from collections import deque
import h5py
import scipy

FLAGS = tf.app.flags.FLAGS

def sample_n_unique(sampling_f, n):
    """Helper function. Given a function `sampling_f` that returns
    comparable objects, sample n such unique objects.
    """
    res = []
    while len(res) < n:
        candidate = sampling_f()
        if candidate not in res:
            res.append(candidate)
    return res

class Schedule(object):
    def value(self, t):
        """Value of the schedule at time t"""
        raise NotImplementedError()

def linear_interpolation(l, r, alpha):
    return l + alpha * (r - l)

class PiecewiseSchedule(object):
    def __init__(self, endpoints, interpolation=linear_interpolation, outside_value=None):
        """Piecewise schedule.
        endpoints: [(int, int)]
            list of pairs `(time, value)` meanining that schedule should output
            `value` when `t==time`. All the values for time must be sorted in
            an increasing order. When t is between two times, e.g. `(time_a, value_a)`
            and `(time_b, value_b)`, such that `time_a <= t < time_b` then value outputs
            `interpolation(value_a, value_b, alpha)` where alpha is a fraction of
            time passed between `time_a` and `time_b` for time `t`.
        interpolation: lambda float, float, float: float
            a function that takes value to the left and to the right of t according
            to the `endpoints`. Alpha is the fraction of distance from left endpoint to
            right endpoint that t has covered. See linear_interpolation for example.
        outside_value: float
            if the value is requested outside of all the intervals sepecified in
            `endpoints` this value is returned. If None then AssertionError is
            raised when outside value is requested.
        """
        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        self._interpolation = interpolation
        self._outside_value = outside_value
        self._endpoints      = endpoints

    def value(self, t):
        """See Schedule.value"""
        for (l_t, l), (r_t, r) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if l_t <= t and t < r_t:
                alpha = float(t - l_t) / (r_t - l_t)
                return self._interpolation(l, r, alpha)

        # t does not belong to any of the pieces, so doom.
        assert self._outside_value is not None
        return self._outside_value

class ConstantSchedule(object):
    def __init__(self, value):
        """Value remains constant over time.
        Parameters
        ----------
        value: float
            Constant value of the schedule
        """
        self._v = value

    def value(self, t):
        """See Schedule.value"""
        return self._v

class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p            = final_p
        self.initial_p          = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction  = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)

def minimize_and_clip(optimizer, objective, var_list, clip_val=10):
    """Minimized `objective` using `optimizer` w.r.t. variables in
    `var_list` while ensure the norm of the gradients for each
    variable is clipped to `clip_val`
    """
    gradients = optimizer.compute_gradients(objective, var_list=var_list)
    for i, (grad, var) in enumerate(gradients):
        if grad is not None:
            print("optimize variable ", var.op.name)
            gradients[i] = (tf.clip_by_norm(grad, clip_val), var)
            tf.contrib.deprecated.histogram_summary("gradients/"+gradients[i][0].op.name, gradients[i][0])

    return optimizer.apply_gradients(gradients)

def initialize_interdependent_variables(session, vars_list, feed_dict):
    """Initialize a list of variables one at a time, which is useful if
    initialization of some variables depends on initialization of the others.
    """
    vars_left = vars_list
    while len(vars_left) > 0:
        new_vars_left = []
        for v in vars_left:
            try:
                # If using an older version of TensorFlow, uncomment the line
                # below and comment out the line after it.
                #session.run(tf.initialize_variables([v]), feed_dict)
                session.run(tf.variables_initializer([v]), feed_dict)
            except tf.errors.FailedPreconditionError:
                new_vars_left.append(v)
        if len(new_vars_left) >= len(vars_left):
            # This can happend if the variables all depend on each other, or more likely if there's
            # another variable outside of the list, that still needs to be initialized. This could be
            # detected here, but life's finite.
            raise Exception("Cycle in variable dependencies, or extenrnal precondition unsatisfied.")
        else:
            vars_left = new_vars_left

def get_wrapper_by_name(env, classname):
    currentenv = env
    while True:
        if classname in currentenv.__class__.__name__:
            return currentenv
        elif isinstance(env, gym.Wrapper):
            currentenv = currentenv.env
        else:
            raise ValueError("Couldn't find wrapper named %s"%classname)

class ReplayBuffer(object):
    def __init__(self, size, frame_history_len, num_actions):
        """This is a memory efficient implementation of the replay buffer.

        The sepecific memory optimizations use here are:
            - only store each frame once rather than k times
              even if every observation normally consists of k last frames
            - store frames as np.uint8 (actually it is most time-performance
              to cast them back to float32 on GPU to minimize memory transfer
              time)
            - store frame_t and frame_(t+1) in the same buffer.

        For the tipical use case in Atari Deep RL buffer with 1M frames the total
        memory footprint of this buffer is 10^6 * 84 * 84 bytes ~= 7 gigabytes

        Warning! Assumes that returning frame of zeros at the beginning
        of the episode, when there is less frames than `frame_history_len`,
        is acceptable.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        frame_history_len: int
            Number of memories to be retried for each observation.
        """
        self.size = size
        self.frame_history_len = frame_history_len
        self.num_actions = num_actions

        self.next_idx      = 0
        self.num_in_buffer = 0

        self.obs      = None
        self.action   = None
        self.reward   = None
        self.done     = None

    def can_sample(self, batch_size):
        """Returns true if `batch_size` different transitions can be sampled from the buffer."""
        return batch_size + 1 <= self.num_in_buffer

    def _encode_sample(self, idxes):
        obs_batch = np.concatenate([self._encode_observation(idx)[None] for idx in idxes], 0)
        act_batch      = self.action[idxes]
        rew_batch      = self.reward[idxes]
        next_obs_batch = np.concatenate([self._encode_observation(idx + 1)[None] for idx in idxes], 0)
        done_mask      = np.array([1.0 if self.done[idx] else 0.0 for idx in idxes], dtype=np.float32)

        return obs_batch, act_batch, rew_batch, next_obs_batch, done_mask

    def sample(self, batch_size, name='rl'):
        """Sample `batch_size` different transitions.

        i-th sample transition is the following:

        when observing `obs_batch[i]`, action `act_batch[i]` was taken,
        after which reward `rew_batch[i]` was received and subsequent
        observation  next_obs_batch[i] was observed, unless the epsiode
        was done which is represented by `done_mask[i]` which is equal
        to 1 if episode has ended as a result of that action.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            Array of shape
            (batch_size, img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8
        act_batch: np.array
            Array of shape (batch_size,) and dtype np.int32
        rew_batch: np.array
            Array of shape (batch_size,) and dtype np.float32
        next_obs_batch: np.array
            Array of shape
            (batch_size, img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8
        done_mask: np.array
            Array of shape (batch_size,) and dtype np.float32
        """
        assert self.can_sample(batch_size)

        #TODO: I am not fully understand what is the meaning of the name
        num_good_in_buffer = int(self.num_in_buffer*(1-FLAGS.bad_portion))
        if name == 'rl':
            idxes = sample_n_unique(lambda: random.randint(0, self.num_in_buffer - 2), batch_size)
        elif name == 'supervised':
            idxes = sample_n_unique(lambda: random.randint(0, num_good_in_buffer - 2), batch_size)
        elif name == 'dqfd':
            is_bad = np.random.rand(1) <= FLAGS.bad_portion
            if is_bad:
                idxes = sample_n_unique(lambda: random.randint(num_good_in_buffer - 1, self.num_in_buffer - 2),
                                        batch_size)
            else:
                idxes = sample_n_unique(lambda: random.randint(0, num_good_in_buffer - 2),
                                        batch_size)
        return self._encode_sample(idxes)


    def encode_recent_observation(self):
        """Return the most recent `frame_history_len` frames.

        Returns
        -------
        observation: np.array
            Array of shape (img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8, where observation[:, :, i*img_c:(i+1)*img_c]
            encodes frame at time `t - frame_history_len + i`
        """
        assert self.num_in_buffer > 0
        return self._encode_observation((self.next_idx - 1) % self.size)

    def _encode_observation(self, idx):
        end_idx   = idx + 1 # make noninclusive
        start_idx = end_idx - self.frame_history_len
        # this checks if we are using low-dimensional observations, such as RAM
        # state, in which case we just directly return the latest RAM.
        if len(self.obs.shape) == 2:
            return self.obs[end_idx-1]
        # if there weren't enough frames ever in the buffer for context
        if start_idx < 0 and self.num_in_buffer != self.size:
            start_idx = 0
        for idx in range(start_idx, end_idx - 1):
            if self.done[idx % self.size]:
                start_idx = idx + 1
        missing_context = self.frame_history_len - (end_idx - start_idx)
        # if zero padding is needed for missing context
        # or we are on the boundry of the buffer
        if start_idx < 0 or missing_context > 0:
            frames = [np.zeros_like(self.obs[0]) for _ in range(missing_context)]
            for idx in range(start_idx, end_idx):
                frames.append(self.obs[idx % self.size])
            return np.concatenate(frames, 2)
            # this branch returns a H*W*(NC) object
        else:
            # this optimization has potential to saves about 30% compute time \o/
            img_h, img_w = self.obs.shape[1], self.obs.shape[2]
            # from NHWC to HWNC, to H*W*NC
            return self.obs[start_idx:end_idx].transpose(1, 2, 0, 3).reshape(img_h, img_w, -1)

    def store_frame(self, frame):
        """Store a single frame in the buffer at the next available index, overwriting
        old frames if necessary.

        Parameters
        ----------
        frame: np.array
            Array of shape (img_h, img_w, img_c) and dtype np.uint8
            the frame to be stored

        Returns
        -------
        idx: int
            Index at which the frame is stored. To be used for `store_effect` later.
        """
        if self.obs is None:
            self.obs         = np.empty([self.size] + list(frame.shape), dtype=np.uint8)
            self.action      = np.empty([self.size],                     dtype=np.int32)
            self.reward      = np.empty([self.size],                     dtype=np.float32)
            self.done        = np.empty([self.size],                     dtype=np.bool)
            self.info        = np.empty([self.size],
                                         dtype={'names':['speed','angle','trackPos', 'trackWidth','damage', 'next_damage', 'stuck'],
                                                'formats':['f8','f8', 'f8', 'f8', 'i1', 'i1', 'b1']})

        self.obs[self.next_idx] = frame

        ret = self.next_idx
        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)

        return ret

    def store_effect(self, idx, action, reward, done, info=None):
        """Store effects of action taken after obeserving frame stored
        at index idx. The reason `store_frame` and `store_effect` is broken
        up into two functions is so that once can call `encode_recent_observation`
        in between.

        Paramters
        ---------
        idx: int
            Index in buffer of recently observed frame (returned by `store_frame`).
        action: int
            Action that was performed upon observing this frame.
        reward: float
            Reward that was received when the actions was performed.
        done: bool
            True if episode was finished after performing that action.
        """
        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx]   = done

        if info is not None:
            self.info[idx] = (info['speed'],info['angle'],info['trackPos'],
                              info['trackWidth'],info['damage'], info['next_damage'], info['is_stuck'])
        else:
            self.info = None

def get_hdf_demo(filename, replay_buffer, sync=True, num_actions=9):
    print("Warning: the num_actions is set as ", num_actions, ". modify if needed")
    # sync=True is the updated version. Only set it to false when we use older data

    print('Get keys of HDF! Please Wait... Demonstration is huge.')
    action = "A"
    reward = "R"
    obs = "S"
    terminal = "terminal"

    # parse filename into a list, it could be a comma(,) seperated filename list
    filename = filename.split(",")
    filename = [x.strip() for x in filename if x.strip() != ""]

    for fi in filename:
        f1 = h5py.File(fi, 'r')
        _action = list(f1[action])
        _reward = list(f1[reward])
        _obs = np.array(f1[obs])
        _terminal = list(f1[terminal])

        assert (len(_action) == len(_reward))
        assert (len(_action) == len(_obs))
        assert (len(_action) == len(_terminal))
        print(len(_obs), '*' * 30)

        _obs_buffer = deque(maxlen=2)
        for i in range(len(_obs)):
            if i % 5000 == 0:
                print('%d are loaded' % i)

            # TODO: we might only need the sync=True and Flags.can_deal_with_human_demo=True case, but not sure yet
            if sync:
                if FLAGS.can_deal_with_human_demo:
                    if i < len(_obs)-1:
                        idx = replay_buffer.store_frame(TorcsProcessFrame84.aframe(_obs[i], 120, 160, 'resize'))
                        replay_buffer.store_effect(idx, _action[i+1], _reward[i+1], _terminal[i+1])
                else:
                    idx = replay_buffer.store_frame(_obs[i][:, :, 3][..., np.newaxis])
                    replay_buffer.store_effect(idx, _action[i], np.sign(_reward[i]), _terminal[i])
            else:
                _obs_buffer.append(_obs[i])
                if i % 4 == 3:
                    max_frame = np.max(np.stack(_obs_buffer), axis=0)
                    max_frame = process_frame84(max_frame)
                    idx = replay_buffer.store_frame(max_frame)
                    replay_buffer.store_effect(idx, _action[i], np.sign(_reward[i]), _terminal[i])

    print('Loaded! Almost there! Replay Size is %d' % replay_buffer.num_in_buffer)
    return replay_buffer


def load_replay_pickle(pickle_dir, step_num):
    print('loading replay buffer...')
    if FLAGS.bad_portion > 0:
        p = (1-FLAGS.bad_portion)
        with open(pickle_dir, 'r') as f:
            replay_buffer = pickle.load(f)
        with open(FLAGS.bad_dir, 'r') as f:
            replay_buffer_bad = pickle.load(f)
        size = replay_buffer.size
        good_size = int(size*p)
        bad_size = size-int(size*p)

        for attr in ['obs', 'action', 'reward', 'done']:
            mixed = np.concatenate((getattr(replay_buffer, attr)[0:good_size],
                                    getattr(replay_buffer_bad, attr)[0:bad_size]))
            setattr(replay_buffer, attr, mixed)
    else:
        with open(pickle_dir, 'rb') as f:
            replay_buffer = pickle.load(f, encoding='latin1')

        for attr in ['obs', 'action', 'reward', 'done']:
            truncated = getattr(replay_buffer, attr)[0:step_num]
            setattr(replay_buffer, attr, truncated)

        replay_buffer.size = step_num
        replay_buffer.num_in_buffer = step_num
    assert(step_num <= 300000)
    print('loaded! truncate at %d' % step_num)
    return replay_buffer


# TODO: I didn't refactor this function
def eval_valset(q, obs_t_ph, val_set_file, session, gamma, frame_history_length=4):
    print("enter the eval_valset function")
    def parse_valset(filename):
        action = "A"
        reward = "R"
        obs = "S"
        terminal = "terminal"
        lives = "lives"

        # parse filename into a list, it could be a comma(,) seperated filename list
        filename = filename.split(",")
        filename = [x.strip() for x in filename if x.strip() != ""]

        obs_list = []
        action_list = []
        reward_list = []
        terminal_list = []

        for fi in filename:
            f1 = h5py.File(fi, 'r')
            _action = list(f1[action])
            _reward = list(f1[reward])

            #_obs = list(f1[obs])
            _obs = np.array(f1[obs])
            _obs_out = []

            _terminal = list(f1[terminal])
            assert (len(_action) == len(_reward))
            assert (len(_action) == len(_obs))
            assert (len(_action) == len(_terminal))
            print(len(_obs), '*' * 30)

            for i in range(len(_obs)):
                if FLAGS.can_deal_with_human_demo:
                    if i < len(_obs) - 1:
                        _obs_out.append(TorcsProcessFrame84.aframe(_obs[i], 120, 160, 'resize'))

            obs_list = obs_list + _obs_out[0:-1]
            action_list = action_list + _action[1:]
            reward_list = reward_list + _reward[1:]
            terminal_list = terminal_list + _terminal[1:]
        #import pdb; pdb.set_trace()
        return obs_list, reward_list, action_list, terminal_list

    valset_obs, valset_reward, valset_action, valset_terminal = parse_valset(val_set_file)
    frame_counter = 0
    avg_bellman = 0

    viz_list=[]

    print("before looping")
    for i in range(3, len(valset_obs)-1):
        print("in evaluation, timestep", i)

        input_obs = np.asarray(valset_obs[i+1-frame_history_length:i+1]).reshape(1,  84, 84, frame_history_length)
        next_obs  = np.asarray(valset_obs[i+2-frame_history_length:i+2]).reshape(1,  84, 84, frame_history_length)


        reward = valset_reward[i]
        action = valset_action[i]
        feed_input_obs = np.reshape(input_obs, list(input_obs.shape))
        feed_next_obs  = np.reshape(next_obs, list(next_obs.shape))

        q_values = session.run(q, feed_dict={obs_t_ph: feed_input_obs})
        # TODO: Yang, save those runed actions into the file and analyze it.
        q_values = np.squeeze(q_values)
        viz_list.append((q_values, feed_input_obs, action, reward))
        # end of saving visualization results
        q_act    = q_values[action]

        q_next = session.run(q, feed_dict={obs_t_ph: feed_next_obs})
        q_next_max = np.argmax(q_next)
        q_look_ahead = reward + gamma * q_next_max
        if not valset_terminal[i]:
            frame_counter += 1
            avg_bellman += q_act - q_look_ahead

    if True:
        print("saving visualization output for validation")
        with open("q_values_dump.pkl", "w") as f:
            pickle.dump(viz_list, f)

    return avg_bellman/frame_counter


def inspect_q_values(q, obs_t_ph,
                             session,
                             replay_buffer):
    # check the q values on the replay buffer.
    num_samples = 100
    viz_list = []
    for i in range(num_samples):
        print("evaluating samples at ", i)
        package = replay_buffer.sample(1, 'rl')
        obs_t_batch, act_t_batch, rew_t_batch, obs_tp1_batch, done_mask = \
            package

        q_values = session.run(q, feed_dict={obs_t_ph: obs_t_batch})
        # TODO: Yang, save those runed actions into the file and analyze it.
        q_values = np.squeeze(q_values)
        viz_list.append((q_values, act_t_batch))

    print("in oneval replay: saving visualization output for validation")
    with open("q_values_dump.pkl", "w") as f:
        pickle.dump(viz_list, f)


# evaluate a policy in a testing environment
def eval_policy(env, q, obs_t_ph,
                session,
                eps, frame_history_len, num_actions, img_c):
    reward_calc = 0
    input_obs = env.reset()
    frame_counter = 0
    damage_counter = 0
    damage_inds = []
    while True:
        frame_counter += 1
        is_greedy = np.random.rand(1) >= eps
        if is_greedy and frame_counter >= frame_history_len:
            # import pdb; pdb.set_trace()
            feed_input_obs = np.reshape(input_obs,[1]+list(input_obs.shape))
            q_values = session.run(q, feed_dict={obs_t_ph: feed_input_obs})
            action = np.argmax(np.squeeze(q_values))
        else:
            action = np.random.choice(num_actions)

        obs, reward, done, info = env.step(action)
        if "torcs" in FLAGS.env_id and info != {}:
            damage = int(info['damage'])
            next_damage = int(info['next_damage'])
            if next_damage - damage > 1:
                damage_counter += 1
                damage_inds.append(1)
            else:
                damage_inds.append(0)
        input_obs = np.concatenate((input_obs, obs), 2)
        assert(len(env.observation_space.shape) == 3)
        if input_obs.shape[2] > frame_history_len*img_c:
            input_obs = input_obs[:, :, -frame_history_len*img_c:]
        reward_calc += reward
        if done:
            break
        if frame_counter >= 30*60*(60/4):
            # 30mins * 60seconds * 15Hz
            print("emulator reach 30 mins maximum length")
            break
    return reward_calc, frame_counter, damage_counter, damage_inds

def eps_scheduler(t, good_step, m_bad, m_good):
    if m_bad > 0:
        if t < good_step:
            should_save = True
            return 0, should_save
        elif (t-good_step) % (m_good+m_bad) < m_good:
            should_save = False
            return 0, should_save
        elif (t-good_step) % (m_good+m_bad) >= m_good:
            should_save = True
            return 1, should_save
    else:
        return 0.1, True

def _activation_summary(x):
  print(x)
  tensor_name = x.op.name
  tf.contrib.deprecated.histogram_summary(tensor_name + '/activations', x)
  tf.contrib.deprecated.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def activation_summaries(endpoints):
  if isinstance(endpoints, dict):
    end_values = endpoints.values()
  elif isinstance(endpoints, list):
    # throw away the tuple's first entry which is the name
    end_values = [t if isinstance(t, tf.Tensor) else t[1] for t in endpoints]
  else:
    print(endpoints, "unknown endpoint type")

  with tf.name_scope('summaries'):
    print("-"*40 + "\nAll tensors that will be summarized:")
    for act in end_values:
      _activation_summary(act)
