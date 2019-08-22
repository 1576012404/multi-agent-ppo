from gym import Env as gymEnv
from gym import spaces
import retro
import numpy as np
import copy



class Env(gymEnv):

    def __init__(self,iIndex,num_agent=2):
        super(Env,self).__init__()
        self._env = retro.make(game='Pong-Atari2600', players=2)
        self._index=iIndex
        self.num_agent=num_agent
        observation_space = spaces.Box(-255, 255, (210, 160, 3),dtype=np.uint8)
        action_space=spaces.Discrete(3)
        # action_space = spaces.MultiBinary(8)
        self.observation_space=[observation_space for _ in range(num_agent)]
        self.action_space=[action_space for _ in range(num_agent)]
        self.render_obs=None

    def reset(self):
        obs=self._env.reset()
        obs2=copy.copy(obs)
        self.render_obs=obs

        return [obs,obs2]

    def step(self,actions):
        two_player_action=np.zeros((16,),dtype=np.int8)


        if np.random.randint(2)<1:
            two_player_action[15] = 1
        else:
            two_player_action[14] = 1
        for i,action in enumerate(actions):
            if action>0:
                two_player_action[2*i+3+action]=1

        # action=np.concatenate(action,axis=0)
        # action = action.astype(np.int8)
        obs,rewards,done,info=self._env.step(two_player_action)
        self.render_obs=obs


        return [obs,copy.copy(obs)],rewards,done,[info,copy.copy(info)]

    def render(self, mode='array'):
        if mode=="human":
            self._env.render()
        else:
            return self.render_obs




