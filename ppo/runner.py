import numpy as np


class Runner():
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """

    def __init__(self, *, env, model_list, nsteps, gamma, lam):
        self.env = env
        self.model_list = model_list
        self.nenv = nenv = env.num_envs if hasattr(env, 'num_envs') else 1
        self.n_agent = len(self.model_list)
        self.obs_list = env.reset()
        self.nsteps = nsteps
        self.states_list = [model.initial_state for model in model_list]
        print("obs_list", len(self.obs_list), self.obs_list[0].shape)
        self.dones = [[False for _ in range(nenv)] for _ in range(self.n_agent)]
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma

    def run(self):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [[] for _ in range(self.n_agent)], \
                                                                             [[] for _ in range(self.n_agent)], \
                                                                             [[] for _ in range(self.n_agent)], \
                                                                             [[] for _ in range(self.n_agent)], \
                                                                             [[] for _ in range( self.n_agent)], \
                                                                             [[] for _ in range(self.n_agent)]
        mb_states = self.states_list
        epinfos = [[] for _ in range(self.n_agent)]
        # For n in range number of steps
        for _ in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            all_agent_action = []
            states_list = []
            for i in range(self.n_agent):
                actions, values, states, neglogpacs = self.model_list[i].step(self.obs_list[i], S=self.states_list[i],
                                                                              M=self.dones[i])
                mb_obs[i].append(self.obs_list[i].copy())
                mb_actions[i].append(actions)
                all_agent_action.append(actions)
                mb_values[i].append(values)
                mb_neglogpacs[i].append(neglogpacs)
                mb_dones[i].append(self.dones[i])
                states_list.append(states)

            self.states_list = states_list
            all_env_action = list(zip(*all_agent_action))

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            self.obs_list, rewards, self.dones, infos = self.env.step(all_env_action)
            for i in range(self.n_agent):
                for info in infos[i]:
                    maybeepinfo = info.get('episode%s' % i)
                    if maybeepinfo: epinfos[i].append(maybeepinfo)
                mb_rewards[i].append(rewards[i])
        # batch of steps to batch of rollouts
        mb_obs = [np.asarray(obs, dtype=self.obs_list[0].dtype) for obs in mb_obs]
        mb_rewards = [np.asarray(rewards, dtype=np.float32) for rewards in mb_rewards]
        mb_actions = [np.asarray(actions, dtype=np.float32) for actions in mb_actions]
        mb_values = [np.asarray(values, dtype=np.float32) for values in mb_values]
        mb_neglogpacs = [np.asarray(neglogpacs, dtype=np.float32) for neglogpacs in mb_neglogpacs]
        mb_dones = [np.asarray(dones, dtype=np.bool) for dones in mb_dones]
        last_values = []

        for i in range(self.n_agent):
            last_values.append(self.model_list[i].value(self.obs_list[i], S=self.states_list[i], M=self.dones[i]))
        # discount/bootstrap off value fn

        mb_advs = [np.zeros_like(mb_rewards[0]) for _ in range(self.n_agent)]
        # print("adv",mb_advs[0].shape,mb_rewards[0].shape)
        lastgaelam = [0 for _ in range(self.n_agent)]
        mb_returns = []
        for i in range(self.n_agent):
            for t in reversed(range(self.nsteps)):
                if t == self.nsteps - 1:
                    nextnonterminal = 1.0 - self.dones[i]
                    nextvalues = last_values[i]
                else:
                    nextnonterminal = 1.0 - mb_dones[i][t + 1]
                    nextvalues = mb_values[i][t + 1]

                delta = mb_rewards[i][t] + self.gamma * nextvalues * nextnonterminal - mb_values[i][t]
                mb_advs[i][t] = lastgaelam[i] = delta + self.gamma * self.lam * nextnonterminal * lastgaelam[i]
            # print("mb_values", mb_advs[0].shape,mb_values[0].shape)
            single_returns = mb_advs[i] + mb_values[i]

            mb_returns.append(single_returns)
        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
                mb_states, epinfos)


# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr_list):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr_list[0].shape
    return [arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:]) for arr in arr_list]
