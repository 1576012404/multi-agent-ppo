import os
import time
import numpy as np
import os.path as osp
from baselines import logger
from collections import deque
from baselines.common import explained_variance, set_global_seeds
from multi_env_tools.policies import build_policy
import tensorflow as tf

try:
    from mpi4py import MPI
except ImportError:
    MPI = None
from interative_ppo.runner import Runner


def constfn(val):
    def f(_):
        return val
    return f

def learn(*, network, env, total_timesteps, eval_env = None, seed=None, nsteps=2048, ent_coef=0.0, lr=3e-4,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=0, load_path=None, model_fn=None, update_fn=None, init_fn=None, mpi_rank_weight=1, comm=None, **network_kwargs):
    '''
    Learn policy using PPO algorithm (https://arxiv.org/abs/1707.06347)

    Parameters:
    ----------

    network:                          policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baseline.common/models.py for full list)
                                      specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
                                      tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                                      neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                                      See common/models.py/lstm for more details on using recurrent nets in policies

    env: baseline.common.vec_env.VecEnv     environment. Needs to be vectorized for parallel environment simulation.
                                      The environments produced by gym.make can be wrapped using baseline.common.vec_env.DummyVecEnv class.


    nsteps: int                       number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                                      nenv is number of environment copies simulated in parallel)

    total_timesteps: int              number of timesteps (i.e. number of actions taken in the environment)

    ent_coef: float                   policy entropy coefficient in the optimization objective

    lr: float or function             learning rate, constant or a schedule function [0,1] -> R+ where 1 is beginning of the
                                      training and 0 is the end of the training.

    vf_coef: float                    value function loss coefficient in the optimization objective

    max_grad_norm: float or None      gradient norm clipping coefficient

    gamma: float                      discounting factor

    lam: float                        advantage estimation discounting factor (lambda in the paper)

    log_interval: int                 number of timesteps between logging events

    nminibatches: int                 number of training minibatches per update. For recurrent policies,
                                      should be smaller or equal than number of environments run in parallel.

    noptepochs: int                   number of training epochs per update

    cliprange: float or function      clipping range, constant or schedule function [0,1] -> R+ where 1 is beginning of the training
                                      and 0 is the end of the training

    save_interval: int                number of timesteps between saving events

    load_path: str                    path to load the model from

    **network_kwargs:                 keyword arguments to the policy / network builder. See baseline.common/policies.py/build_policy and arguments to a particular type of network
                                      For instance, 'mlp' network architecture has arguments num_hidden and num_layers.



    '''

    set_global_seeds(seed)

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)
    # Get the nb of env
    nenvs = env.num_envs

    # Get state_space and action_space
    ob_space_list = env.observation_space
    ac_space_list = env.action_space
    num_agent =len(ob_space_list)

    # Calculate the batch_size
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches
    is_mpi_root = (MPI is None or MPI.COMM_WORLD.Get_rank() == 0)

    # Instantiate the model object (that creates act_model and train_model)
    if model_fn is None:
        from interative_ppo.model import Model
        model_fn = Model

    from interative_ppo.act_model import ActModel

    model_list=[]
    sample_model_list=[]

    for i,i_obs_space in enumerate(ob_space_list):
        i_action_space=ac_space_list[i]
        print("network",network,network_kwargs)
        policy = build_policy(i_obs_space,i_action_space, network, **network_kwargs)

        model = model_fn(policy=policy, ob_space=i_obs_space, ac_space=i_action_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                        nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                        max_grad_norm=max_grad_norm, comm=comm, mpi_rank_weight=mpi_rank_weight,model_index=i)
        model_list.append(model)

    g=tf.Graph()
    with g.as_default():
        sample_sess=tf.Session(graph=g)
        for i ,i_obs_space in enumerate(ob_space_list):
            i_action_space=ac_space_list[i]
            policy=build_policy(i_obs_space,i_action_space,network,**network_kwargs)
            model=ActModel(policy=policy,ob_space=i_obs_space,ac_space=i_action_space,nbatch_act=nenvs,sample_sess=sample_sess,model_index=i)
            sample_model_list.append(model)

    print("after_load")

    if load_path is not None:
        for i,i_load_path in enumerate(load_path):
            model_list[i].load(i_load_path)
    # Instantiate the runner object
    runner = Runner(env=env, model_list=model_list,sample_model_list=sample_model_list, nsteps=nsteps, gamma=gamma, lam=lam)
    if eval_env is not None:
        eval_runner = Runner(env = eval_env, model_list = model_list, nsteps = nsteps, gamma = gamma, lam= lam)

    epinfobuf = [deque(maxlen=100)for _ in range(num_agent)]
    if eval_env is not None:
        eval_epinfobuf = deque(maxlen=100)

    if init_fn is not None:
        init_fn()

    # Start total timer

    tfirststart = time.perf_counter()

    nupdates = total_timesteps // nbatch

    interative_interval=3
    interative_Away_Now=50


    for update in range(1, nupdates+1):
        assert nbatch % nminibatches == 0
        # Start timer
        tstart = time.perf_counter()
        frac = 1.0 - (update - 1.0) / nupdates
        # Calculate the learning rate
        lrnow = lr(frac)
        # Calculate the cliprange
        cliprangenow = cliprange(frac)

        # if update % log_interval == 0 and is_mpi_root: logger.info('Stepping environment...')

        # Get minibatch
        if update>interative_interval and update%interative_interval==0:
            for iLoad in range(num_agent):
                # iStartSample=max(interative_interval,(update-interative_Away_Now)//3*3)
                # iSample=np.random.choice(range(iStartSample,update,3))
                iSample=1
                sample_path=osp.join(logger.get_dir(),"checkpoints","%s-%.5i"%(iLoad,iSample))
                oSample_Model=sample_model_list[iLoad]
                oSample_Model.load(sample_path)


        mblossvals = [[] for _ in range(num_agent)]
        for iTrainAgent in range(num_agent):
            obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run(iTrainAgent) #pylint: disable=E0632
            if eval_env is not None:
                eval_obs, eval_returns, eval_masks, eval_actions, eval_values, eval_neglogpacs, eval_states, eval_epinfos = eval_runner.run() #pylint: disable=E0632

            # if update % log_interval == 0 and is_mpi_root: logger.info('Done.')

            epinfobuf[iTrainAgent].extend(epinfos[iTrainAgent])
            if eval_env is not None:
                eval_epinfobuf.extend(eval_epinfos)

        # Here what we're going to do is for each minibatch calculate the loss and append it.

            if states[0] is None: # nonrecurrent version
                # Index of each element of batch_size
                # Create the indices array
                inds = np.arange(nbatch)
                for _ in range(noptepochs):
                    # Randomize the indexes
                    np.random.shuffle(inds)
                    # 0 to batch_size with batch_train_size step
                    for start in range(0, nbatch, nbatch_train):
                        end = start + nbatch_train
                        mbinds = inds[start:end]
                        slices = (arr[mbinds] for arr in (obs[iTrainAgent], returns[iTrainAgent], masks[iTrainAgent], actions[iTrainAgent], values[iTrainAgent], neglogpacs[iTrainAgent]))
                        mblossvals[iTrainAgent].append(model_list[iTrainAgent].train(lrnow, cliprangenow, *slices))
            else: # recurrent version  TODO
                assert nenvs % nminibatches == 0
                envsperbatch = nenvs // nminibatches
                envinds = np.arange(nenvs)
                flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
                for _ in range(noptepochs):
                    np.random.shuffle(envinds)
                    for start in range(0, nenvs, envsperbatch):
                        end = start + envsperbatch
                        mbenvinds = envinds[start:end]
                        mbflatinds = flatinds[mbenvinds].ravel()
                        slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                        mbstates = states[mbenvinds]
                        mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))

        # Feedforward --> get losses --> update
        lossvals = [np.mean(mblossvals[i], axis=0)for i in range(num_agent)]
        # End timer
        tnow = time.perf_counter()
        # Calculate the fps (frame per second)
        fps = int(nbatch / (tnow - tstart))

        if update_fn is not None:
            update_fn(update)

        if update % log_interval == 0 or update == 1:
            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            ev = explained_variance(values[0], returns[0])
            logger.logkv("misc/serial_timesteps", update*nsteps)
            logger.logkv("misc/nupdates", update)
            logger.logkv("misc/total_timesteps", update*nbatch)
            logger.logkv("fps", fps)
            logger.logkv("misc/explained_variance", float(ev))
            for i in range(num_agent):
                logger.logkv('eprewmean%s'%i, safemean([epinfo['r'] for epinfo in epinfobuf[i]]))
                logger.logkv('eplenmean%s'%i, safemean([epinfo['l'] for epinfo in epinfobuf[i]]))
            if eval_env is not None:
                logger.logkv('eval_eprewmean', safemean([epinfo['r'] for epinfo in eval_epinfobuf]) )
                logger.logkv('eval_eplenmean', safemean([epinfo['l'] for epinfo in eval_epinfobuf]) )
            logger.logkv('misc/time_elapsed', tnow - tfirststart)
            for i in range(num_agent):
                for (lossval, lossname) in zip(lossvals[i], model_list[i].loss_names):
                    logger.logkv('loss%s/'%i + lossname, lossval)

            logger.dumpkvs()
        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir() and is_mpi_root:
            checkdir = osp.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            for iModel,model in enumerate(model_list):
                savepath = osp.join(checkdir, '%s-%.5i'%(iModel,update))
                print('Saving to', savepath)
                model.save(savepath)

    return model_list
# Avoid division error when calculate the mean (in our case if epinfo is empty returns np.nan, not return an error)
def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)



