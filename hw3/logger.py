import logging
import os

def create_logger(save_path, file_type):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    file_name = os.path.join(save_path, file_type + '_log.txt')
    cs = logging.StreamHandler()
    cs.setLevel(logging.DEBUG)
    logger.addHandler(cs)

    fh = logging.FileHandler(file_name, mode='w')
    fh.setLevel(logging.INFO)

    logger.addHandler(fh)

    return logger