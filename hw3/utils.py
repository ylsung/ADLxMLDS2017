import logging
import os
import pickle

def create_logger(save_path, file_type, keep_train):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    file_name = os.path.join(save_path, file_type + '_log.txt')
    cs = logging.StreamHandler()
    cs.setLevel(logging.DEBUG)
    logger.addHandler(cs)

    if keep_train:
        fh = logging.FileHandler(file_name)
    else:
        fh = logging.FileHandler(file_name, mode='w')
    fh.setLevel(logging.INFO)

    logger.addHandler(fh)

    return logger

def save_pickle(save_path, name, obj):
    file_name = os.path.join(save_path, '%s.pickle' % name)
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(load_path, name):
    file_name = os.path.join(load_path, '%s.pickle' % name)
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data