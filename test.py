import retro
from baselines.common.atari_wrappers import MaxAndSkipEnv,FrameStack,WarpFrame
import time
import numpy as np
from pong_env import Env

def main():
    env = Env(0)
    # env = WarpFrame(env)
    # env = MaxAndSkipEnv(env, skip=4)
    # env=FrameStack(env,4)
    obs = env.reset()
    print("obs",env.action_space)
    while True:
        # action_space will by MultiBinary(16) now instead of MultiBinary(8)
        # the bottom half of the actions will be for player 1 and the top half for player 2
        action_space_list=env.action_space
        action=[action_space.sample() for action_space in action_space_list]

        # print("action", action)

        obs, rew, done, info = env.step(action)
        time.sleep(0.02)


        # print("action", obs.shape,rew,done,info)

        # rew will be a list of [player_1_rew, player_2_rew]
        # done and info will remain the same
        env.render()
        if done:
            print("done",done)
            obs = env.reset()
    env.close()


if __name__ == "__main__":
    main()

