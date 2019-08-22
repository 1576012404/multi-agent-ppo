from baselines.common.vec_env import VecEnvWrapper
import numpy as np
import time
from collections import deque

class VecMonitor(VecEnvWrapper):
    def __init__(self, venv, filename=None, keep_buf=0, info_keywords=()):
        VecEnvWrapper.__init__(self, venv)
        self.eprets = None
        self.eplens = None
        self.epcount = 0
        self.tstart = time.time()
        self.info_keywords = info_keywords
        self.keep_buf = keep_buf
        self.num_agent=venv.num_agent
        if self.keep_buf:
            self.epret_buf = deque([], maxlen=keep_buf)
            self.eplen_buf = deque([], maxlen=keep_buf)

    def reset(self):
        obs = self.venv.reset()
        self.eprets = [np.zeros(self.num_envs, 'f')for _ in range(self.num_agent)]
        self.eplens = [np.zeros(self.num_envs, 'i')for _ in range(self.num_agent)]
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        newinfos = [list(info[:])for info in infos]
        for i in range(self.num_agent):
            self.eprets[i] += rews[i]
            self.eplens[i] += 1
            for j in range(self.num_envs):
                if dones[i][j]:
                    info = infos[i][j].copy()
                    ret = self.eprets[i][j]
                    eplen = self.eplens[i][j]
                    epinfo = {'r': ret, 'l': eplen, 't': round(time.time() - self.tstart, 6)}
                    for k in self.info_keywords:
                        epinfo[k] = info[k]
                    info['episode%s'%i] = epinfo
                    if self.keep_buf:
                        self.epret_buf.append(ret)
                        self.eplen_buf.append(eplen)
                    self.epcount += 1
                    self.eprets[i][j] = 0
                    self.eplens[i][j] = 0
                    newinfos[i][j] = info
        return obs, rews, dones, newinfos