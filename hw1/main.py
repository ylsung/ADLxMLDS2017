import argparse
import os
from train import train, test
from data_loader import DataLoader, Transformer
import numpy as np
from sklearn.model_selection import KFold

def parser():
    parser = argparse.ArgumentParser(description='hw1: rnn model')
    parser.add_argument('--todo', choices=['valid', 'test'], default='valid', help='valid or test')
    parser.add_argument('--data_directory', default='data/', help='path for all data')
    parser.add_argument('--model', default='rnn', help='use which model to train')
    parser.add_argument('--cuda', action='store_true', help='whether to use cuda device')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--_lambda', type=float, default=0.1, help='parameter for l2 regularization')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training')
    parser.add_argument('--epoch', type=int, default=100, help='max epoch to run')
    parser.add_argument('--early_stop', type=int, default=3, help='the epochs to stop training while the validation error not decrease')
    parser.add_argument('--lr', type=float, default=0.03, help='learning rate')
    parser.add_argument('--save', action='store_true', help='save the parameter')
    parser.add_argument('--save_folder', default='test')
    parser.add_argument('--load_folder', default='', help='load the parameter')
    parser.add_argument('--write_file', default='prediction.csv', help='write to the directory')
    return parser.parse_args()

def make_data(data_root, dtype):
    if dtype == 'valid':
        train_fbank_path = os.path.join(data_root, 'fbank/train.ark')
        train_mfcc_path = os.path.join(data_root, 'mfcc/train.ark')

        label_path = os.path.join(data_root, 'label/train.lab')

        _48_39_path = os.path.join(data_root, 'phones/48_39.map')
        num2char_path = os.path.join(data_root, '48phone_char.map')

        transformer = Transformer(_48_39_path, num2char_path)

        train_fbank_data_loader = DataLoader(train_fbank_path, 'data')
        train_mfcc_data_loader = DataLoader(train_mfcc_path, 'data')

        train_label_data_loader = DataLoader(label_path, 'label', transformer)

        train_fbank_id, train_fbank_feature, train_fbank_framelength = train_fbank_data_loader.load_data()
        train_mfcc_id, train_mfcc_feature, train_mfcc_framelength = train_mfcc_data_loader.load_data()

        label_instance_id, train_label = train_label_data_loader.load_label()

        # mapping label and fbank
        index = np.argsort(label_instance_id)
        label_instance_id = label_instance_id[index]
        train_label = train_label[index]

        assert np.sum(np.char.equal(label_instance_id, train_fbank_id)) == label_instance_id.shape[0], 'the label and data not aligned'
        assert np.sum(np.char.equal(label_instance_id, train_mfcc_id)) == label_instance_id.shape[0], 'the label and data not aligned'

        # concatenate fbank and mfcc

        train_feature = np.concatenate((train_fbank_feature, train_mfcc_feature), axis=2)

        # train_feature = train_fbank_feature
        return [train_fbank_id, train_feature, train_label, train_fbank_framelength], transformer


    elif dtype == 'test':
        test_fbank_path = os.path.join(data_root, 'fbank/test.ark')
        test_mfcc_path = os.path.join(data_root, 'mfcc/test.ark')

        _48_39_path = os.path.join(data_root, 'phones/48_39.map')
        num2char_path = os.path.join(data_root, '48phone_char.map')

        transformer = Transformer(_48_39_path, num2char_path)

        test_fbank_data_loader = DataLoader(test_fbank_path, 'data')
        test_mfcc_data_loader = DataLoader(test_mfcc_path, 'data')
        test_fbank_id, test_fbank_feature, test_fbank_framelength = test_fbank_data_loader.load_data()
        test_mfcc_id, test_mfcc_feature, test_mfcc_framelength = test_mfcc_data_loader.load_data()

        assert np.sum(np.char.equal(test_fbank_id, test_mfcc_id)) == test_mfcc_id.shape[0], 'the label and data not aligned'

        # concatenate fbank and mfcc

        test_feature = np.concatenate((test_fbank_feature, test_mfcc_feature), axis=2)
        # test_feature = test_fbank_feature

        return [test_fbank_id, test_feature, test_fbank_framelength], transformer

    # check they are the same
def k_fold_fn(train_tuple):
    # train_tuple's content 
    # [0] : id
    # [1] : feature
    # [2] : label
    # [3] : framelength

    kf = KFold(n_splits=fold)
    train_list = []

    valid_list = []

    for train_index, valid_index in kf.split(train_tuple[0]):
        train_id = train_tuple[0][train_index]
        train_feature = train_tuple[1][train_index]
        train_label = train_tuple[2][train_index]
        train_framelength = train_tuple[3][train_index]

        train_list.append([train_id, train_feature, train_label, train_framelength])

        valid_id = train_tuple[0][valid_index]
        valid_feature = train_tuple[1][valid_index]
        valid_label = train_tuple[2][valid_index]
        valid_framelength = train_tuple[3][valid_index]

        valid_list.append([valid_id, valid_feature, valid_label, valid_framelength])

    return train_list, valid_list

def k_fold_station_fn(train_tuple):
    all_index = list(range(train_tuple[0].shape[0]))
    split_num = train_tuple[0].shape[0] // fold

    i = 0
    train_list = []

    valid_list = []
    while i < fold:
        if i == fold - 1:
            valid_index = all_index[i * split_num:]
        else:
            valid_index = all_index[i * split_num: (i + 1) * split_num]
        train_index = list(set(all_index) - set(valid_index))

        train_id = train_tuple[0][train_index]
        train_feature = train_tuple[1][train_index]
        train_label = train_tuple[2][train_index]
        train_framelength = train_tuple[3][train_index]

        train_list.append([train_id, train_feature, train_label, train_framelength])

        valid_id = train_tuple[0][valid_index]
        valid_feature = train_tuple[1][valid_index]
        valid_label = train_tuple[2][valid_index]
        valid_framelength = train_tuple[3][valid_index]

        valid_list.append([valid_id, valid_feature, valid_label, valid_framelength])

        i += 1
    return train_list, valid_list



def normalize(train_data, train_framelength, valid_data=[], valid_framelength=[]):
    total_size = train_framelength.sum()
    mean = (train_data).sum(axis=0).sum(axis=0) / float(total_size)
    E_sum_square = (train_data ** 2 / float(total_size)).sum(axis=0).sum(axis=0)
    std = (E_sum_square - mean ** 2) ** 0.5

    for i in range(train_data.shape[0]):
        train_data[i][:train_framelength[i]] = (train_data[i][:train_framelength[i]] - mean) / std
    if len(valid_data) != 0:
        for i in range(valid_data.shape[0]):
            valid_data[i][:valid_framelength[i]] = (valid_data[i][:valid_framelength[i]] - mean) / std
    return train_data, valid_data, mean, std

def normalize_2(train_data, train_framelength, valid_data=[], valid_framelength=[]):
    total_size = train_framelength.sum()
    # mean = (train_data).sum(axis=0).sum(axis=0) / float(total_size)
    mask = framelength2mask(train_data, train_framelength)
    train_data[~mask] = -1000
    _max = train_data.max(axis=0).max(axis=0)
    train_data[~mask] = 1000
    _min = train_data.min(axis=0).min(axis=0)
    train_data[~mask] = 0.0
    _middle = (_max + _min) / 2.0
    _range = (_max - _min) / 2.0 + 1e-12

    for i in range(train_data.shape[0]):
        train_data[i][:train_framelength[i]] = (train_data[i][:train_framelength[i]] - _middle) / (_range)
    if len(valid_data) != 0:
        for i in range(valid_data.shape[0]):
            valid_data[i][:valid_framelength[i]] = (valid_data[i][:valid_framelength[i]] - _middle) / (_range)
    return train_data, valid_data, _middle, _range

def framelength2mask(data, framelength):
    mask = np.zeros(shape=data.shape, dtype=np.bool)
    for i in range(len(framelength)):
        mask[i, :framelength[i], :] = 1.0
    return mask

def combine_around_data(data):
    before = np.zeros(shape=data.shape, dtype=np.float32)
    after = np.zeros(shape=data.shape, dtype=np.float32)

    before[:, 1:, :] = data[:, :-1, :]
    after[:, :-1, :] = data[:, 1:, :]

    return np.concatenate((before, data, after), axis=2)
def transition_matrix(label, framelength):
    transition_matrix = np.zeros(shape=(48, 48), dtype=np.float32)
    transition_matrix.fill(1e-4)
    for i in range(label.shape[0]):
        for j in range(framelength[i] - 1):
            transition_matrix[label[i][j]][label[i][j + 1]] += 1
    _sum = transition_matrix.sum(1).reshape(-1, 1)

    transition_matrix = transition_matrix / _sum
    return transition_matrix
def main():
    args = parser()
    print(args)
    if not os.path.exists('model'):
        os.mkdir('model')
    if not os.path.exists(os.path.join('model', args.save_folder)):
        os.mkdir(os.path.join('model', args.save_folder))

    if args.todo == 'valid':
        # load fbank/mfcc data and label
        train_tuple, transformer = make_data(args.data_directory, 'valid')
        print(train_tuple[1].shape)
        print(train_tuple[2].shape)
        print(train_tuple[3].shape)
        # train, valid tuple format: (id, feature, label, framelength)
        # train_tuple is a list
        transition_matrix_list = []
        if fold != 1:
            train_list, valid_list = k_fold_station_fn(train_tuple)
            for i in range(fold):

                train_list[i][1], valid_list[i][1], mean, std =\
                    normalize_2(train_list[i][1], train_list[i][3], valid_list[i][1], valid_list[i][3])
                # train_list[i][1] = combine_around_data(train_list[i][1])
                # valid_list[i][1] = combine_around_data(valid_list[i][1])

                # transition_matrix_list.append(transition_matrix(train_list[i][2], train_list[i][3]))
                if args.save:
                    mean_name = os.path.join('model', args.save_folder, 'mean_%d' % i)
                    std_name = os.path.join('model', args.save_folder, 'std_%d' % i)
                    np.save(mean_name, mean)
                    np.save(std_name, std)
        else:
            valid_list = []
            train_list = [train_tuple]
            train_list[0][1], _ = normalize(train_list[0][1], train_list[0][3])
            print(train_list[0][1].shape)
        train(args, transformer, train_list, valid_list, transition_matrix_list)
    elif args.todo == 'test':
        test_tuple, transformer = make_data(args.data_directory, 'test')
        test_list = []
        for i in range(fold):
            mean_name = os.path.join('model', args.load_folder, 'mean_%d.npy' % i)
            std_name = os.path.join('model', args.load_folder, 'std_%d.npy' % i)
            mean = np.load(mean_name)
            std = np.load(std_name)
            test_data = (test_tuple[1] - mean) / std
            test_list.append([test_tuple[0], test_data, test_tuple[2]])
        test(args, transformer, test_list)

if __name__ == '__main__':
    fold = 5
    main()

