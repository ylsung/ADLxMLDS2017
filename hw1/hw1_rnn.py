import argparse
import os
from train import train
from data_loader import DataLoader, Transformer
import numpy as np
from sklearn.model_selection import KFold

def parser():
    parser = argparse.ArgumentParser(description='hw1: rnn model')
    parser.add_argument('--todo', choices=['valid', 'test'], default='valid', help='valid or test')
    parser.add_argument('--data_directory', default='data/', help='path for all data')
    parser.add_argument('--model', default='rnn', help='use which model to train')
    parser.add_argument('--cuda', action='store_true', help='whether to use cuda device')
    parser.add_argument('--_lambda', type=float, default=0.1, help='parameter for l2 regularization')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training')
    parser.add_argument('--epoch', type=int, default=100, help='max epoch to run')
    parser.add_argument('--early_stop', type=int, default=3, help='the epochs to stop training while the validation error not decrease')
    parser.add_argument('--lr', type=float, default=0.03, help='learning rate')
    parser.add_argument('--save', action='store_true', help='save the parameter')
    parser.add_argument('--load_directory', default='', help='load the parameter')
    parser.add_argument('--write_file', help='write to the directory')
    return parser.parse_args()

def make_data(data_root):
    train_fbank_path = os.path.join(data_root, 'fbank/train.ark')
    train_mfcc_path = os.path.join(data_root, 'mfcc/train.ark')

    label_path = os.path.join(data_root, 'label/train.lab')

    test_fbank_path = os.path.join(data_root, 'fbank/test.ark')
    test_mfcc_path = os.path.join(data_root, 'mfcc/test.ark')

    _48_39_path = os.path.join(data_root, 'phones/48_39.map')
    num2char_path = os.path.join(data_root, '48phone_char.map')

    transformer = Transformer(_48_39_path, num2char_path)

    train_fbank_data_loader = DataLoader(train_fbank_path, 'data')
    # train_mfcc_data_loader = DataLoader(train_mfcc_path, 'data')

    train_label_data_loader = DataLoader(label_path, 'label', transformer)

    test_fbank_data_loader = DataLoader(test_fbank_path, 'data')
    # test_mfcc_data_loader = DataLoader(test_mfcc_path, 'data')

    train_fbank_id, train_fbank_feature, train_fbank_framelength = train_fbank_data_loader.load_data()
    # train_mfcc_id, train_mfcc_feature, train_mfcc_framelength = train_mfcc_data_loader.load_data()

    test_fbank_id, test_fbank_feature, test_fbank_framelength = test_fbank_data_loader.load_data()
    # test_mfcc_id, test_mfcc_feature, test_mfcc_framelength = test_mfcc_data_loader.load_data()

    label_instance_id, train_label = train_label_data_loader.load_label()

    # mapping label and fbank
    index = np.argsort(label_instance_id)
    label_instance_id = label_instance_id[index]
    train_label = train_label[index]

    assert np.sum(np.char.equal(label_instance_id, train_fbank_id)) == label_instance_id.shape[0], 'the label and data not aligned'

    return (train_fbank_id, train_fbank_feature, train_label, train_fbank_framelength), transformer

    # check they are the same
def k_fold_fn(train_tuple):
    # train_tuple's content 
    # [0] : id
    # [1] : feature
    # [2] : label
    # [3] : framelength

    kf = KFold(n_splits=5)
    train_list = []

    valid_list = []

    for train_index, valid_index in kf.split(train_tuple[0]):
        train_id = train_tuple[0][train_index]
        train_feature = train_tuple[1][train_index]
        train_label = train_tuple[2][train_index]
        train_framelength = train_tuple[3][train_index]

        train_list.append((train_id, train_feature, train_label, train_framelength))

        valid_id = train_tuple[0][valid_index]
        valid_feature = train_tuple[1][valid_index]
        valid_label = train_tuple[2][valid_index]
        valid_framelength = train_tuple[3][valid_index]

        valid_list.append((valid_id, valid_feature, valid_label, valid_framelength))

    return train_list, valid_list


def main():
    args = parser()
    print(args)

    # load fbank/mfcc data and label
    train_tuple, transformer = make_data(args.data_directory)
    # train, valid tuple format: (id, feature, label, framelength)
    train_tuple_list, valid_tuple_list = k_fold_fn(train_tuple)
    if args.todo == 'valid':
        train(args, transformer, train_tuple_list, valid_tuple_list)

if __name__ == '__main__':
    main()

