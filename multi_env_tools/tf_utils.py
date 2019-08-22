
import joblib
from baselines.common.tf_util import get_session
import tensorflow as tf
import os

def save_trainable_variables(save_path,variables=None,scope=None,sess=None):
    sess=sess or get_session()
    variables=variables or tf.trainable_variables(scope)
    ps=sess.run(variables)
    save_dict={v.name:value for v,value in zip(variables,ps)}
    dirname = os.path.dirname(save_path)
    if any(dirname):
        os.makedirs(dirname, exist_ok=True)
    joblib.dump(save_dict, save_path)


def load_trainable_variables(load_path, variables=None,scope=None, sess=None):
    import joblib
    sess = sess or get_session()
    variables = variables or tf.trainable_variables(scope)

    loaded_params = joblib.load(os.path.expanduser(load_path))
    restores = []
    if isinstance(loaded_params, list):
        assert len(loaded_params) == len(variables), 'number of variables loaded mismatches len(variables)'
        for d, v in zip(loaded_params, variables):
            restores.append(v.assign(d))
    else:
        for v in variables:
            restores.append(v.assign(loaded_params[v.name]))

    sess.run(restores)

ALREADY_INITIALIZED_SAMPLE=set()
def initialize_sample(sess):
    new_variables=set(tf.global_variables())-ALREADY_INITIALIZED_SAMPLE
    sess.run(tf.variables_initializer(new_variables))
    ALREADY_INITIALIZED_SAMPLE.update(new_variables)