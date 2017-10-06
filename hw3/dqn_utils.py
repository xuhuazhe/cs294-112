"""This file includes a collection of utility functions that are useful for
implementing DQN."""
import gym
import tensorflow as tf
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
from atari_wrappers import *
from collections import deque
import h5py
import scipy

FLAGS = tf.app.flags.FLAGS

def huber_loss(x, delta=1.0):
    # https://en.wikipedia.org/wiki/Huber_loss
    return tf.select(
        tf.abs(x) < delta,
        tf.square(x) * 0.5,
        delta * (tf.abs(x) - 0.5 * delta)
    )

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

def compute_exponential_averages(variables, decay):
    """Given a list of tensorflow scalar variables
    create ops corresponding to their exponential
    averages
    Parameters
    ----------
    variables: [tf.Tensor]
        List of scalar tensors.
    Returns
    -------
    averages: [tf.Tensor]
        List of scalar tensors corresponding to averages
        of al the `variables` (in order)
    apply_op: tf.runnable
        Op to be run to update the averages with current value
        of variables.
    """
    averager = tf.train.ExponentialMovingAverage(decay=decay)
    apply_op = averager.apply(variables)
    return [averager.average(v) for v in variables], apply_op

def minimize_and_clip(optimizer, objective, var_list, dueling_list=None, clip_val=10, multiplier=1/np.sqrt(2)):
    """Minimized `objective` using `optimizer` w.r.t. variables in
    `var_list` while ensure the norm of the gradients for each
    variable is clipped to `clip_val`
    """
    gradients = optimizer.compute_gradients(objective, var_list=var_list)
    for i, (grad, var) in enumerate(gradients):
        if grad is not None:
            print(var.op.name, "what is var for me!!!!!!!!!!!!")
            #break
            if 'q_func/convnet/last_conv' in var.op.name:
                print('*'*30, 'multiply!')
                grad = multiplier * grad
            if FLAGS.optimize_V_only and not("value_only" in var.op.name):
                print('zero out gradient for', var.op.name)
                grad = 0.0 * grad
            gradients[i] = (tf.clip_by_norm(grad, clip_val), var)
            tf.histogram_summary("gradients/"+gradients[i][0].op.name, gradients[i][0])
    #if  FLAGS.dueling:
    #    gradients_duel = optimizer.compute_gradients(objective, var_list=dueling_list)
    #    for i, (grad, var) in enumerate(gradients_duel):
    #        if grad is not None:
    #            gradients_duel[i] = (multiplier * grad, var)
    #            tf.histogram_summary("gradients_scaled/"+gradients_duel[i][0].op.name, gradients_duel[i][0])
    #    gradients = gradients + gradients_duel
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
                session.run(tf.initialize_variables([v]), feed_dict)
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

    def fix_action_dist(self):
        print("Warning: this function should not be called! You should regenerate the dataset")
        self.num_actions = 9
        self.action_dist = np.ones([self.size, self.num_actions], dtype=np.float32) / self.num_actions

    def can_sample(self, batch_size):
        """Returns true if `batch_size` different transitions can be sampled from the buffer."""
        if FLAGS.inenv_eval:
            return True
        else:
            return batch_size + 1 <= self.num_in_buffer

    def _encode_observation_batch(self, idxes, plus):
        obs = [self._encode_observation(idx+plus, True) for idx in idxes]

        img_h, img_w, img_c = self.obs.shape[1], self.obs.shape[2], self.obs.shape[3]
        framelen=self.frame_history_len

        # fitler out the tuples
        indexes = []
        batches = []

        def add_a_batch():
            # form this batch
            batch = self.obs[indexes].reshape(-1, framelen, img_h, img_w, img_c)
            batch = batch.transpose(0, 2, 3, 1, 4)
            print(batch.shape)
            batch = batch.reshape(-1, img_h, img_w, framelen*img_c)
            batches.append(batch)

        for i, tuple in enumerate(obs):
            if isinstance(tuple, type((1, 1))):
                indexes += range(tuple[0], tuple[1])
            else:
                # size batch*4**H*W*C
                if len(indexes) > 0:
                    add_a_batch()
                    indexes = []
                batches.append(obs[i].reshape(1, img_h, img_w, framelen*img_c))

        if len(indexes) > 0:
            add_a_batch()

        return np.concatenate(batches, 0)


    def _encode_sample(self, idxes):
        obs_batch = np.concatenate([self._encode_observation(idx)[None] for idx in idxes], 0)
        #obs_batch = self._encode_observation_batch(idxes, 0)
        act_batch      = self.action[idxes]
        rew_batch      = self.reward[idxes]
        next_obs_batch = np.concatenate([self._encode_observation(idx + 1)[None] for idx in idxes], 0)
        #next_obs_batch = self._encode_observation_batch(idxes, 1)
        done_mask      = np.array([1.0 if self.done[idx] else 0.0 for idx in idxes], dtype=np.float32)
        action_dist_batch = self.action_dist[idxes, :]

        return obs_batch, act_batch, rew_batch, next_obs_batch, done_mask, action_dist_batch

    def sample_onpolicy(self, number_path, total_path):
        assert(number_path <= total_path)
        counter = 0
        for i in range(self.num_in_buffer):
            obs, act, rew, next_obs, done, action_dist = _encode_sample(i)
        return [self._encode_sample(idxes), need_hinge]

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
        need_hinge = 1
        assert self.can_sample(batch_size)
        if name == 'rl':
            idxes = sample_n_unique(lambda: random.randint(0, self.num_in_buffer - 2), batch_size)
        elif name == 'supervised':
            idxes = sample_n_unique(lambda: random.randint(0, int(self.num_in_buffer*(1-FLAGS.bad_portion)) - 2), batch_size)
        elif name == 'dqfd':
            p = FLAGS.bad_portion
            is_bad = np.random.rand(1) <= p
            if is_bad:
                idxes = sample_n_unique(
                    lambda: random.randint(int(self.num_in_buffer * (1 - FLAGS.bad_portion)) - 1,
                                           self.num_in_buffer - 2), batch_size)
                need_hinge = 0
            else:
                idxes = sample_n_unique(
                    lambda: random.randint(0, int(self.num_in_buffer * (1 - FLAGS.bad_portion)) - 2), batch_size)
        return [self._encode_sample(idxes), need_hinge]


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

    def _encode_observation(self, idx, returnIndex=False):
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
            if returnIndex:
                return (start_idx, end_idx)
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
            self.action_dist = np.empty([self.size, self.num_actions],   dtype=np.float32)
            self.info        = np.empty([self.size],
                                         dtype={'names':['speed','angle','trackPos', 'trackWidth','damage', 'next_damage', 'stuck'],
                                                'formats':['f8','f8', 'f8', 'f8', 'i1', 'i1', 'b1']})

        self.obs[self.next_idx] = frame

        ret = self.next_idx
        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)

        return ret

    def store_effect(self, idx, action, reward, done, action_dist, info=None):
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
        self.action_dist[idx, :] = action_dist
        if info is not None:
            self.info[idx] = (info['speed'],info['angle'],info['trackPos'],
                              info['trackWidth'],info['damage'], info['next_damage'], info['is_stuck'])
        else:
            self.info = None




def get_hdf_demo(filename, replay_buffer, sync=True, num_actions=9):
    print("Warning: the num_actions is set as ", num_actions, ". modify if needed")

    # sync=True is the updated version. Only set it to false when we use older data
    # List all groups
    # print("Keys: %s" % f.keys())

    print('Get keys of HDF! Please Wait... Demonstration is huge.')
    action = "A"
    reward = "R"
    obs = "S"
    terminal = "terminal"
    lives = "lives"

    # parse filename into a list, it could be a comma(,) seperated filename list
    filename = filename.split(",")
    filename = [x.strip() for x in filename if x.strip() != ""]

    delta_action_dist = np.zeros((num_actions), dtype=np.float32)

    for fi in filename:
        f1 = h5py.File(fi, 'r')
        _action = list(f1[action])
        _reward = list(f1[reward])
        _obs = list(f1[obs])
        _terminal = list(f1[terminal])
        assert (len(_action) == len(_reward))
        assert (len(_action) == len(_obs))
        assert (len(_action) == len(_terminal))
        print(len(_obs), '*' * 30)

        _obs_buffer = deque(maxlen=2)
        for i in range(len(_obs)):
            if i % 5000 == 0:
                print('%d are loaded' % i)

            delta_action_dist[_action[i]]=1.0

            if sync:
                if FLAGS.human_torcs:
                    idx = replay_buffer.store_frame(TorcsProcessFrame84.aframe(_obs[i], 120, 160, 'resize'))
                    replay_buffer.store_effect(idx, _action[i], _reward[i], _terminal[i], delta_action_dist)
                else:
                    idx = replay_buffer.store_frame(_obs[i][:, :, 3][..., np.newaxis])
                    replay_buffer.store_effect(idx, _action[i], np.sign(_reward[i]), _terminal[i], delta_action_dist)
            else:
                _obs_buffer.append(_obs[i])
                if i % 4 == 3:
                    max_frame = np.max(np.stack(_obs_buffer), axis=0)
                    max_frame = process_frame84(max_frame)
                    idx = replay_buffer.store_frame(max_frame)
                    replay_buffer.store_effect(idx, _action[i], np.sign(_reward[i]), _terminal[i], delta_action_dist)
            delta_action_dist[_action[i]] = 0.0

    print('Loaded! Almost there! Replay Size is %d' % replay_buffer.size)
    return replay_buffer


def load_replay_pickle(pickle_dir, step_num, bad_dir=''):
    bad_dir = FLAGS.bad_dir
    print('loading replay buffer...')
    if FLAGS.bad_portion > 0:
        p = (1-FLAGS.bad_portion)
        with open(pickle_dir, 'r') as f:
            replay_buffer = pickle.load(f)
        with open(bad_dir, 'r') as f:
            replay_buffer_bad = pickle.load(f)
        size = replay_buffer.size
        replay_buffer.obs = np.concatenate((replay_buffer.obs[0:int(size*p)],
                            replay_buffer_bad.obs[0:size-int(size*p)]))
        replay_buffer.action = np.concatenate((replay_buffer.action[0:int(size*p)],
                            replay_buffer_bad.action[0:size - int(size * p)]))
        replay_buffer.reward = np.concatenate((replay_buffer.reward[0:int(size*p)],
                            replay_buffer_bad.reward[0:size - int(size * p)]))
        replay_buffer.done = np.concatenate((replay_buffer.done[0:int(size*p)],
                            replay_buffer_bad.done[0:size - int(size * p)]))
    else:
        with open(pickle_dir, 'r') as f:
            replay_buffer = pickle.load(f)
        replay_buffer.obs = replay_buffer.obs[0:step_num]
        replay_buffer.action = replay_buffer.action[0:step_num]
        replay_buffer.reward = replay_buffer.reward[0:step_num]
        replay_buffer.done = replay_buffer.done[0:step_num]
        replay_buffer.size = step_num
        replay_buffer.num_in_buffer = step_num
    assert(step_num <= 300000)
    print('loaded! truncate at %d' % step_num)
    return replay_buffer


def eval_policy(env, q, obs_t_ph,
                session,
                eps, frame_history_len, num_actions, img_c):
    # TODO: we could probably use Replay buffer to get rid of frame_history_len issue
    reward_calc = 0

    input_obs = env.reset()
    frame_counter = 0
    damage_counter = 0
    while True:
        frame_counter += 1
        is_greedy = np.random.rand(1) >= eps
        if is_greedy and frame_counter >= frame_history_len:
            feed_input_obs = np.reshape(input_obs,[1]+list(input_obs.shape))
            q_values = session.run(q, feed_dict={obs_t_ph: feed_input_obs})
            action = np.argmax(np.squeeze(q_values))
        else:
            action = np.random.choice(num_actions)

        obs, reward, done, info = env.step(action)
        if info != {}:
            #print(info)
            damage = int(info['damage'])
            next_damage = int(info['next_damage'])
            if next_damage - damage > 1:
                damage_counter += 1
        input_obs = np.concatenate((input_obs, obs), 2)
        assert(len(env.observation_space.shape) == 3)
        if input_obs.shape[2] > frame_history_len*img_c:
            input_obs = input_obs[:, :, -frame_history_len*img_c:]
        reward_calc += reward
        if done:
            break
        if frame_counter >= 30*60*(60/4):
        #if frame_counter >= 5 * 60 * (60 / 4):
            # 30mins * 60seconds * 15Hz
            print("emulator reach 5 mins maximum length")
            break
    return reward_calc, frame_counter, damage_counter

def eps_scheduler(t, good_step, m_bad, m_good):
    if m_bad > 0:
        if t< good_step:
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
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


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

def select_exploration(name, exploration, t):
    if name == "normal":
        eps = exploration.value(t)
        return eps
    elif name == "tiny":
        eps = FLAGS.tiny_explore
        return eps
    else:
        raise ValueError("explore_method invalid %s" % FLAGS.explore_value_method)

def pathlength(path):
    return len(path["reward"])

def discount(x, gamma):
    """
    Compute discounted sum of future values
    out[i] = in[i] + gamma * in[i+1] + gamma^2 * in[i+2] + ...
    """
    return scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]

class NnValueFunction(object):
    coeffs = None

    def __init__(self, session):
        self.net = None
        self.session = session

    def create_net(self, shape):
        self.x = tf.placeholder(tf.float32, shape=[None, shape], name="x")
        self.y = tf.placeholder(tf.float32, shape=[None], name="y")
        hidden1 = tf.nn.relu(dense(self.x, 32, 'value-net-hidden1', 1.0))
        hidden2 = tf.nn.relu(dense(hidden1, 16, 'value-net-hidden2', 1.0))
        self.net = dense(hidden2, 1, 'value-net-out', 1.0)
        self.net = tf.reshape(self.net, (-1,))
        l2 = (self.net - self.y) * (self.net - self.y)
        self.train = tf.train.AdamOptimizer().minimize(l2)
        self.session.run(tf.initialize_all_variables())

    def preproc(self, X):
        return np.concatenate([np.ones([X.shape[0], 1]), X, np.square(X)/2.0], axis=1)

    def fit(self, X, y):
        featmat = self.preproc(X)
        if self.net is None:
            self.create_net(featmat.shape[1])
        for _ in range(40):
            self.session.run(self.train, {self.x: featmat, self.y: y})

    def predict(self, X):
        if self.net is None:
            return np.zeros(X.shape[0])
        else:
            ret = self.session.run(self.net, {self.x: self.preproc(X)})
            return np.reshape(ret, (ret.shape[0],))

'''_________discrete_env__________'''
'''
from gym import Env, spaces
from gym.utils import seeding

def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()


class DiscreteEnv(Env):

    """
    Has the following members
    - nS: number of states
    - nA: number of actions
    - P: transitions (*)
    - isd: initial state distribution (**)
    (*) dictionary dict of dicts of lists, where
      P[s][a] == [(probability, nextstate, reward, done), ...]
    (**) list or array of length nS
    """
    def __init__(self, nS, nA, P, isd):
        self.P = P
        self.isd = isd
        self.lastaction=None # for rendering
        self.nS = nS
        self.nA = nA

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        self._seed()
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.s = categorical_sample(self.isd, self.np_random)
        self.lastaction=None
        return self.s

    def _step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d= transitions[i]
        self.s = s
        self.lastaction=a
        return (s, r, d, {"prob" : p})
import numpy as np
import sys
from six import StringIO, b

from gym import utils
import discrete_env

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {
    "4x4": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG"
    ],
}

class FrozenLakeEnv(discrete_env.DiscreteEnv):
    """
    However, the ice is slippery, so you won't always move in the direction you intend.
    The surface is described using a grid like the following
        SFFF
        FHFH
        FFFH
        HFFG
    S : starting point, safe
    F : frozen surface, safe
    H : hole, fall to your doom
    G : goal, where the frisbee is located
    The episode ends when you reach the goal or fall in a hole.
    You receive a reward of 1 if you reach the goal, and zero otherwise.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None, map_name="4x4",is_slippery=True):
        if desc is None and map_name is None:
            raise ValueError('Must provide either desc or map_name')
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc,dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape

        nA = 4
        nS = nrow * ncol

        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        P = {s : {a : [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row*ncol + col
        def inc(row, col, a):
            if a==0: # left
                col = max(col-1,0)
            elif a==1: # down
                row = min(row+1,nrow-1)
            elif a==2: # right
                col = min(col+1,ncol-1)
            elif a==3: # up
                row = max(row-1,0)
            return (row, col)

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = P[s][a]
                    letter = desc[row, col]
                    if letter in b'GH':
                        li.append((1.0, s, 0, True))
                    else:
                        if is_slippery:
                            for b in [(a-1)%4, a, (a+1)%4]:
                                newrow, newcol = inc(row, col, b)
                                newstate = to_s(newrow, newcol)
                                newletter = desc[newrow, newcol]
                                done = bytes(newletter) in b'GH'
                                rew = float(newletter == b'G')
                                li.append((0.8 if b==a else 0.1, newstate, rew, done))
                        else:
                            newrow, newcol = inc(row, col, a)
                            newstate = to_s(newrow, newcol)
                            newletter = desc[newrow, newcol]
                            done = bytes(newletter) in b'GH'
                            rew = float(newletter == b'G')
                            li.append((1.0, newstate, rew, done))

        super(FrozenLakeEnv, self).__init__(nS, nA, P, isd)

    def _render(self, mode='human', close=False):
        if close:
            return
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Left","Down","Right","Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        return outfile '''