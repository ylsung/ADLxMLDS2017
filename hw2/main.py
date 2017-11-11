import numpy as np
import os
import argparse
import json
import re
import pickle

def parser():
    parser = argparse.ArgumentParser(description='seq2seq')

    parser.add_argument('--todo', choices=['train', 'valid', 'test', 'special'], default='valid')
    parser.add_argument('--model', default='seq2seq')
    parser.add_argument('--data', default='data')
    parser.add_argument('--save', default='')
    parser.add_argument('--load', default='')
    parser.add_argument('--test_out', default='output/test_out.txt')
    parser.add_argument('--peer_out', default='output/peer_out.txt')

    return parser.parse_args()

def load_data(directory, file_type, word2num={}, num2word={}, max_length=0):
    if file_type == 'train':
        feat_directory = os.path.join(directory, 'training_data/feat')
        label_name = os.path.join(directory, 'training_label.json')
    elif file_type == 'test':
        feat_directory = os.path.join(directory, 'testing_data/feat')
        label_name = os.path.join(directory, 'testing_label.json')
    files = os.listdir(feat_directory)
    with open(label_name, 'r') as f:
        label_json = json.load(f)

    if file_type == 'train':
        word2num = {}
        num2word = {}
        word2num['BOS'] = 0
        word2num['EOS'] = 1
        word2num['UNK'] = 2
        num2word[0] = 'BOS'
        num2word[1] = 'EOS'
        num2word[2] = 'UNK'
        max_length = 0
        index = len(word2num)
        label_dict = {}
        max_caption = 0
        split_str = '; |, |\*|\n| |'
        for label_piece in label_json:
            _id = label_piece['id'].split('.')[0]
            label_list = label_piece['caption']
            max_caption = max(max_caption, len(label_list))
            for caption in label_list:

                word_list = caption.rstrip('.').split()
                max_length = max(max_length, len(word_list) + 2)

                for word in word_list:
                    if word not in word2num:
                        num2word[index] = word
                        word2num[word] = index
                        index += 1
            # max_length = max(max_length, len(max(label_list, key=len)))

            label_dict[_id] = label_list
    elif file_type == 'test':

        label_dict = {}
        max_caption = 0
        split_str = '; |, |\*|\n| |'
        for label_piece in label_json:
            _id = label_piece['id'].split('.')[0]
            label_list = label_piece['caption']
            max_caption = max(max_caption, len(label_list))

            label_dict[_id] = label_list


    data_list = []
    caption_list = []
    caption_length_list = []
    str_length_list = []
    name_list = []
    for file in files:
        file_name = file.split('.')[0]
        name_list.append(file_name)
        caption_piece = label_dict[file_name]
        caption_length = len(caption_piece)
        caption_array = np.zeros(shape=(max_caption, max_length), dtype=int)
        length_array = np.zeros(shape=(max_caption, ), dtype=int)
        caption_length_list.append(caption_length)
        i = 0
        for caption in caption_piece:
            word_list = caption.rstrip('.').split()
            j = 1
            caption_array[i][0] = word2num['BOS']
            for word in word_list:
                if word2num.get(word) == None:
                    caption_array[i][j] = word2num['UNK']
                else:
                    caption_array[i][j] = word2num[word]
                j += 1
            caption_array[i][j] = word2num['EOS']
            length_array[i] = len(word_list) + 2
            i += 1
        caption_list.append(caption_array)
        str_length_list.append(length_array)

        data_list.append(np.load(os.path.join(feat_directory, file)))
    return np.stack(data_list), np.stack(caption_list), np.stack(str_length_list), np.stack(caption_length_list),\
     np.stack(name_list), word2num, num2word, max_length

# train_data, label_list, length_list, word2num, num2word = load_data('MLDS_hw2_data', 'train')
def load_test_data(directory):
    feat_directory = os.path.join(directory, 'testing_data/feat')
    name_directory = os.path.join(directory, 'testing_id.txt')

    name_array = np.loadtxt(name_directory, dtype=str)
    data_list = []
    for name_avi in name_array:
        name_avi += '.npy'
        data_list.append(np.load(os.path.join(feat_directory, name_avi)))
    return np.stack(data_list), name_array
def load_special_data(directory):
    feat_directory = os.path.join(directory, 'testing_data/feat')
    name_directory = os.path.join(directory, 'special_id.txt')

    name_array = np.loadtxt(name_directory, dtype=str)
    data_list = []
    for name_avi in name_array:
        name_avi += '.npy'
        data_list.append(np.load(os.path.join(feat_directory, name_avi)))
    return np.stack(data_list), name_array
def mean_std(data):
    # return mean and std of each feature 
    E_sum = data.sum(0).sum(0) / (data.shape[0] * data.shape[1])
    E_square_sum = (data ** 2).sum(0).sum(0) / (data.shape[0] * data.shape[1])

    return E_sum, (E_square_sum - E_sum ** 2) ** 0.5
def normalize(data, mean, std):
    return (data - mean) / (std + 1e-12)
def main():
    args = parser()
    print(args)
    if args.model == 'seq2seq':
        from train_seq2seq import train, test
    elif args.model == 's2vt':
        from train_s2vt import train, test

    if args.todo == 'train':
        train_data, label_array, str_length_array, caption_length_array, name_array, word2num, num2word, max_length = \
         load_data(args.data, 'train')
        _mean, _std = mean_std(train_data)
        train_data = normalize(train_data, _mean, _std)
        save_path = os.path.join('model', args.save)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        train_data = train_data.astype(np.float32)
        print('train word len: ', len(word2num))
        # load testing data
        test_data, test_label_array, test_str_length_array, test_caption_length_array, test_name_array, \
          word2num, num2word, test_max_length = load_data(args.data, 'test', word2num, num2word, max_length)
        test_data = normalize(test_data, _mean, _std)

        test_data = test_data.astype(np.float32)
        # print(train_data.dtype)
        print('data shape: ', test_data.shape)
        print('label shape: ', test_label_array.shape)
        print('str length shape: ', test_str_length_array.shape)
        print('caption length shape', test_caption_length_array.shape)
        print('name array shape', test_name_array.shape)
        print('train + test word len: ', len(word2num))

        valid_dict = {
            'valid_data': test_data,
            'valid_label_array': test_label_array,
            'valid_str_length_array': test_str_length_array,
            'valid_caption_length_array': test_caption_length_array,
            'valid_name_array': test_name_array,
        }

        if args.save != '':
            info_dict = {
                'mean': _mean, 
                'std': _std,
                'word2num': word2num,
                'num2word': num2word,
                'max_length': max_length,
            }
            info_path = os.path.join('model', args.save, 'info.pickle')
            with open(info_path, 'wb') as f:
                pickle.dump(info_dict, f)
        train(args, train_data, label_array, str_length_array, caption_length_array, name_array, word2num, num2word, valid_dict, max_length)
    elif args.todo == 'test':
        info_path = os.path.join('model', args.load, 'info.pickle')
        with open(info_path, 'rb') as f:
            info = pickle.load(f)
            _mean = info['mean']
            _std = info['std']
            word2num = info['word2num']
            num2word = info['num2word']
            max_length = info['max_length']
        test_data, test_name = load_test_data(args.data)

        test_data = normalize(test_data, _mean, _std)
        test_data = test_data.astype(np.float32)

        test(args, test_data, test_name, word2num, num2word, max_length)
    elif args.todo == 'special':
        info_path = os.path.join('model', args.load, 'info.pickle')
        with open(info_path, 'rb') as f:
            info = pickle.load(f)
            _mean = info['mean']
            _std = info['std']
            word2num = info['word2num']
            num2word = info['num2word']
            max_length = info['max_length']
        special_data, special_name = load_special_data(args.data)
        special_data = normalize(special_data, _mean, _std)
        special_data = special_data.astype(np.float32)

        test(args, special_data, special_name, word2num, num2word, max_length)

if __name__ == '__main__':
    main()