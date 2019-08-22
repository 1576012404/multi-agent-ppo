import multiprocessing as mp

import numpy as np
from baselines.common.vec_env import SubprocVecEnv




class SubprocVecEnv(SubprocVecEnv):
    """
    VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
    Recommended to use when num_envs > 1 and step() can be a bottleneck.
    """
    def __init__(self, env_fns,num_agent, spaces=None, context='spawn'):
        """
        Arguments:

        env_fns: iterable of callables -  functions that create environments to run in subprocesses. Need to be cloud-pickleable
        """
        self.num_agent=num_agent
        super(SubprocVecEnv,self).__init__(env_fns=env_fns,spaces=spaces,context=context)


    def step_async(self, actions):
        self._assert_not_closed()
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        dones=[[done,done]for done in dones]
        return _flatten_each_obs(obs), [np.stack(i)for i in zip(*rews)], [np.stack(i)for i in zip(*dones)],list(zip(*infos))

    def reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', None))
        return _flatten_each_obs([remote.recv() for remote in self.remotes])

    def close_extras(self):
        self.closed = True
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def get_images(self):
        self._assert_not_closed()
        for pipe in self.remotes:
            pipe.send(('render', None))
        imgs = [pipe.recv() for pipe in self.remotes]
        return imgs


def _flatten_each_obs(obs):
    assert isinstance(obs, (list, tuple))
    assert len(obs) > 0
    return [np.stack(same_agent_obs)for same_agent_obs in zip(*obs)]#n anget  #each agent nenv*env_shape