
from baselines.common.vec_env import VecEnvWrapper
import numpy as np

class VecNormalize(VecEnvWrapper):
    """
    A vectorized wrapper that normalizes the observations
    and returns from an environment.
    """

    def __init__(self, venv, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8, use_tf=False):
        VecEnvWrapper.__init__(self, venv)
        if use_tf:
            from baseline.common.running_mean_std import TfRunningMeanStd
            self.ob_rms = [TfRunningMeanStd(shape=self.observation_space[i].shape,scope='ob_rms')for i in range(self.num_agent)] if ob else None
            self.ret_rms = [TfRunningMeanStd(shape=(), scope='ret_rms')for _ in range(self.num_agent)] if ret else None
        else:
            from baseline.common.running_mean_std import RunningMeanStd
            self.ob_rms = [RunningMeanStd(shape=self.observation_space[i].shape)for i in range(self.num_agent)] if ob else None
            self.ret_rms = [RunningMeanStd(shape=())for _ in range(self.num_agent)] if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = [np.zeros(self.num_envs)for _ in range(self.num_agent)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_agent = venv.num_agent

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.ret = [self.ret[i] * self.gamma + rews[i] for i in range(self.num_agent)]
        obs = self._obfilt(obs)
        if self.ret_rms:
            for i,rms in enumerate(self.ret_rms):
                rms.update(self.ret[i])
                rews[i] = np.clip(rews[i] / np.sqrt(rms.var + self.epsilon), -self.cliprew, self.cliprew)
                self.ret[i][news[i]] = 0.
        return obs, rews, news, infos

    def _obfilt(self, obs):
        if self.ob_rms:
            for i in range(self.num_agent):
                self.ob_rms[i].update(obs[i])
                obs[i] = np.clip((obs[i] - self.ob_rms[i].mean) / np.sqrt(self.ob_rms[i].var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def reset(self):
        self.ret = [np.zeros(self.num_envs)for _ in range(self.num_agent)]
        obs = self.venv.reset()
        return self._obfilt(obs)