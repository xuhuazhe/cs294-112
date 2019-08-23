import cv2
import numpy as np
from collections import deque
import gym
from gym import spaces


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env=None, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        super(NoopResetEnv, self).__init__(env)
        self.noop_max = noop_max
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset()
        noops = np.random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, _, _ = self.env.step(0)
        return obs

class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        """Take action on reset for environments that are fixed until firing."""
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self):
        self.env.reset()
        obs, _, _, _ = self.env.step(1)
        obs, _, _, _ = self.env.step(2)
        return obs

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env=None):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        super(EpisodicLifeEnv, self).__init__(env)
        self.lives = 0
        # True if all lives were used
        self.was_real_done  = True
        # This variable is not used, only assigned
        self.was_real_reset = False

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert somtimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset()
            self.was_real_reset = True
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
            self.was_real_reset = False
        self.lives = self.env.unwrapped.ale.lives()
        return obs

# Warning: Max pool is done across temporal domain, only on recent 2 frames
class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=2)
        self._skip       = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break

        max_frame = np.max(np.stack(self._obs_buffer), axis=0)

        return max_frame, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs

def _process_frame84(frame):
    img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
    img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
    resized_screen = cv2.resize(img, (84, 110),  interpolation=cv2.INTER_LINEAR)
    x_t = resized_screen[18:102, :]
    x_t = np.reshape(x_t, [84, 84, 1])
    return x_t.astype(np.uint8)

def process_frame84(frame):
    return _process_frame84(frame)

class ProcessFrame84(gym.Wrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1))

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return _process_frame84(obs), reward, done, info

    def reset(self):
        return _process_frame84(self.env.reset())

class TorcsProcessFrame84(gym.Wrapper):
    @staticmethod
    def aframe(frame, image_height, image_width, resize_or_crop):
        img = np.reshape(frame, [image_height, image_width, 3]).astype(np.float32)
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        if resize_or_crop == "crop":
            resized_screen = cv2.resize(img, (112, 84), interpolation=cv2.INTER_LINEAR)
            x_t = resized_screen[:, 14:98]
        elif resize_or_crop == "resize":
            x_t = cv2.resize(img, (84, 84), interpolation=cv2.INTER_LINEAR)
        else:
            raise ValueError("invalid resize_or_crop parameter")

        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)

    def __init__(self, env=None, resize_or_crop="resize"):
        self.image_width = 160
        self.image_height = 120
        self.resize_or_crop = resize_or_crop

        super(TorcsProcessFrame84, self).__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1))

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.aframe(obs, self.image_height, self.image_width, self.resize_or_crop), \
               reward, done, info

    def reset(self):
        return self.aframe(self.env.reset(), self.image_height, self.image_width,
                           self.resize_or_crop)

class ClippedRewardsWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # assert np.sign(reward) == reward
        return obs, np.sign(reward), done, info

def wrap_deepmind_ram(env):
    env = EpisodicLifeEnv(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    # env = ClippedRewardsWrapper(env)
    return env

# In summary, the following wrapper has been applied to the Env
# 1. Turn many lives to a single life
# 2. reset for random number of NOOP
# 3. Get one frame for every {1,2,3,4} frames, and take max(3,4), accumulate reward
# 4. FIRE to start if needed
# 5. convert to Gray image and crop a part.
# 6. clip reward to -1, 0, +1
def wrap_deepmind(env):
    print(env.spec.id, '=================')
    assert 'NoFrameskip' in env.spec.id
    env = EpisodicLifeEnv(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ProcessFrame84(env)
    # env = ClippedRewardsWrapper(env)
    return env

# note that the difference between torcs and original atari game
# 1. no random number of NOOP reset
# 2. frame skip has implemented in Torcs, but no max between frames applied
# 3. !!! convert to gray and bilinear resize to 84*84
# 4. !!! NO reward clipping, learning rate scale might be different
def wrap_torcs(env):
    env = TorcsProcessFrame84(env, resize_or_crop="resize")
    return env



class FrozenLakeStateToFrame(gym.Wrapper):
    @staticmethod
    def aframe(state):
        # should have a state space of range(0-8*5), size width=1*height=1*channel=1
        x_t = np.reshape(state, [1, 1, 1])
        return x_t.astype(np.uint8)

    def __init__(self, env=None):
        self.image_width = 1
        self.image_height = 1

        super(FrozenLakeStateToFrame, self).__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(1, 1, 1))

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.aframe(obs), reward, done, info

    def reset(self):
        return self.aframe(self.env.reset())

# wrapping of Frozen Lake
# 1. No randomness
# 2. no frameskip, becaues not a video game
# 3. no image, but need to convert observation to a image type
# 4. no reward clipping, because the reward is already in 1.
def wrap_frozen_lake(env):
    env = FrozenLakeStateToFrame(env)
    return env
