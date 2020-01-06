from multi_env_tools.vec_normalize import VecNormalize
from multi_env_tools.subproc_vec_env import SubprocVecEnv
from multi_env_tools.vec_monitor import VecMonitor
import datetime
from baselines import logger
from interative_ppo import ppo
from pong_env import Env
from multi_env_tools.env_wrappers import MaxAndSkipEnv,FrameStack,WarpFrame


def Train():
    logdir="BaselinesLog/ppo/"+datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    logger.configure(logdir,["tensorboard","stdout"])
    iNum_Player=2

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
    n_timesteps = int(3e8)
    hyperparmas = {'nsteps': 256, 'noptepochs': 8, 'nminibatches': 16, 'lr': learning_rate, 'cliprange': clip_range,
                   'vf_coef': 0.5, 'ent_coef': 0.00}

    num_env = 64
    env = SubprocVecEnv([EnvFunc(i) for i in range(num_env)], num_agent=iNum_Player)
    env = VecMonitor(env)
    # env = VecNormalize(env)

    act = ppo.learn(
        network="cnn",
        env=env,
        total_timesteps=n_timesteps,
        log_interval=1,
        save_interval=1,
        # load_path="baselineLog/ppobaseliens-2019-06-05-11-26-10-382460/checkpoints/00300",
        **hyperparmas,

        # value_network="copy"
    )



if __name__=="__main__":
    Train()
