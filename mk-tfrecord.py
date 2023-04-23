import pickle
import numpy as np
import tensorflow as tf
import os

def convert_pkl_tfrecords(open_file ='./data/hopper-medium-replay-v2.pkl'):
    with open(open_file, 'rb') as f:
        data = pickle.load(f)
    #
    obs_act_size = data[0]['observations'].shape[1] + data[0]['actions'].shape[1]
    obs_act_data = np.array([], ).reshape(0, obs_act_size)

    for num, trajectory in enumerate(data):
        for num in range(len(trajectory["actions"])-31):
            obs = trajectory["observations"][num:num+32]
            act = trajectory["actions"][num:num+32]
            obs_act = np.concatenate([obs, act],axis=1)
            obs_act_data = np.concatenate([obs_act_data, obs_act], axis = 0) # if ys.size else

        dataset = tf.data.Dataset.from_tensor_slices(obs_act_data)
    tf.data.Dataset.save(dataset, './data/hopper-medium-replay-v2')#os.path.dirname(open_file))
    print("Saved to {}".format(os.path.dirname(open_file)))

def load_tfrecords(task_name='data/hopper-medium-replay-v2.tfrecords'):
    dataset = tf.data.experimental.load(task_name)
    return dataset

if __name__ == '__main__':
    convert_pkl_tfrecords()