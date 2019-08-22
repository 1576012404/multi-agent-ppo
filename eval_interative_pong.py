from multi_env_tools.vec_normalize import VecNormalize
from multi_env_tools.subproc_vec_env import SubprocVecEnv
from multi_env_tools.vec_monitor import VecMonitor
import datetime
from baselines import logger
from interative_ppo import ppo
from pong_env import Env
from multi_env_tools.env_wrappers import MaxAndSkipEnv,FrameStack,WarpFrame
import numpy as np

def Eval():
    iNum_Player=num_agent=2

    def EnvFunc(iIndex):
        def _EnvFunc_():
            oEnv=Env(iIndex)
            oEnv = WarpFrame(oEnv)
            oEnv = MaxAndSkipEnv(oEnv, skip=4)
            oEnv=FrameStack(oEnv,4)
            return oEnv
        return _EnvFunc_

    learning_rate = 3e-4
    clip_range = 0.2
    n_timesteps = int(0)
    hyperparmas = {'nsteps': 1024, 'noptepochs': 10, 'nminibatches': 32, 'lr': learning_rate, 'cliprange': clip_range,
                   'vf_coef': 0.5, 'ent_coef': 0.00}

    num_env = 32
    env = SubprocVecEnv([EnvFunc(i) for i in range(num_env)], num_agent=iNum_Player)
    env = VecMonitor(env)
    # env = VecNormalize(env)

    act_list = ppo.learn(
        network="cnn",
        env=env,
        total_timesteps=n_timesteps,
        log_interval=1,
        save_interval=1,
        # load_path=["baselineLog/ppobaseliens-2019-06-05-11-26-10-382460/checkpoints/0-00300",
        #           "baselineLog/ppobaseliens-2019-06-05-11-26-10-382460/checkpoints/1-00300"]
        **hyperparmas,

        # value_network="copy"
    )
    obs=env.reset()
    bDone=False
    iFrame=0
    iReward=np.zeros((2,))
    while not bDone:
        all_agent_action=[]
        for i in range(num_agent):
            actions=act_list[i].step(obs[i])[0]
            all_agent_action.append(actions)
        all_env_action=list(zip(*all_agent_action))

        obs,reward,done,info=env.step(all_env_action)

        iReward+=np.array([reward[0][0],reward[1][0]])
        iFrame+=1
        env.render("human")
        if done[0]:
            print("done",done)
            obs=env.reset()



if __name__=="__main__":
    Eval()