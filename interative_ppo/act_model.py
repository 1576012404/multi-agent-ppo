import tensorflow as tf
import functools


from multi_env_tools.tf_utils import save_trainable_variables,load_trainable_variables,initialize_sample
from baselines.common.tf_util import initialize

try:
    from baselines.common.mpi_adam_optimizer import MpiAdamOptimizer
    from mpi4py import MPI
    from baselines.common.mpi_util import sync_from_root
except ImportError:
    MPI = None

class ActModel(object):
    """
    We use this object to :
    __init__:
    - Creates the step_model
    - Creates the train_model

    train():
    - Make the training part (feedforward and retropropagation of gradients)

    save/load():
    - Save load the model
    """
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act,sample_sess,model_index=0):
        self.sess  = sample_sess
        self.model_index=model_index



        with tf.variable_scope('ppo2_model%s'%model_index):
            # CREATE OUR TWO MODELS
            # act_model that is used for sampling
            act_model = policy(nbatch_act, 1, sample_sess)

        self.act_model=act_model
        self.step=act_model.step
        self.initial_state=act_model.initial_state
        scope="ppo2_model%s"%model_index
        self.save=functools.partial(save_trainable_variables,variables=tf.trainable_variables(scope=scope),sess=sample_sess)
        self.load=functools.partial(load_trainable_variables,variables=tf.trainable_variables(scope=scope),sess=sample_sess)
        initialize_sample(sample_sess)




