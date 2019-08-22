
import gym
import numpy as np

from collections import deque
from gym import spaces
from baselines.common.atari_wrappers import LazyFrames
import cv2



class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, num_players=2,width=84, height=84, grayscale=True, dict_space_key=None):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.

        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
        observation should be warped.
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = [new_space for _ in range(num_players)]
        else:
            raise Exception("Not Implemented")
            # original_space = self.observation_space.spaces[self._key]
            # self.observation_space.spaces[self._key] = new_space
        assert original_space[0].dtype == np.uint8 and len(original_space[0].shape) == 3

    def observation(self, obs_list):
        target_obs_list=[]
        for obs in obs_list:
            if self._key is None:
                frame = obs
            # else:
            #     frame = obs[self._key]

            if self._grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = cv2.resize(
                frame, (self._width, self._height), interpolation=cv2.INTER_AREA
            )
            if self._grayscale:
                frame = np.expand_dims(frame, -1)

            if self._key is None:
                obs = frame
            target_obs_list.append(obs)

        return target_obs_list

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, num_players=2,skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        # self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip       = skip
        self._num_players=num_players

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = [0.0 for _ in range(self._num_players)]
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            # if i == self._skip - 2: self._obs_buffer[0] = obs
            # if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward=list(map(lambda x,y:x+y,reward,total_reward))
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        # max_frame = self._obs_buffer.max(axis=0)

        return obs, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)




class FrameStack(gym.Wrapper):
    def __init__(self, env, k,num_players=2):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.num_players=num_players
        self.frames = deque([], maxlen=k)
        shp = env.observation_space[0].shape
        self.observation_space = [spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space[i].dtype)for i in range(num_players)]

    def reset(self):
        obs_list = self.env.reset()
        for _ in range(self.k):
            self.frames.append(obs_list)
        return self._get_ob()

    def step(self, action):
        obs_list, reward, done, info = self.env.step(action)
        self.frames.append(obs_list)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        all_player_obs=[]
        for player_obs in zip(*self.frames):
            all_player_obs.append(np.concatenate(player_obs,axis=-1))
        return all_player_obs
