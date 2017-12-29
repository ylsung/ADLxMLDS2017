import re
import numpy as np
import logging
import os
import random
# from PIL import Image
import skimage.io
import scipy
import scipy.ndimage
# import matplotlib
# matplotlib.use('agg')
# import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
import torchvision as tv
import torch
from torchvision import transforms

def list2dict(_list):
    style2id = {}
    id2style = {}

    for i, color in enumerate(_list):
        style2id[color] = i
        id2style[i] = color

    return style2id, id2style

hair_color_list = ['orange hair', 'white hair', 'aqua hair', 'gray hair',
'green hair', 'red hair', 'purple hair', 'pink hair', 'blue hair', 'black hair', 
'brown hair', 'blonde hair']

eyes_color_list = ['gray eyes', 'black eyes', 'orange eyes',
'pink eyes', 'yellow eyes', 'aqua eyes', 'purple eyes', 'green eyes', 'brown eyes', 
'red eyes', 'blue eyes']

hair_style2id, hair_id2style = list2dict(hair_color_list)
eyes_style2id, eyes_id2style = list2dict(eyes_color_list)

class SimpleDataset(Dataset):

    def __init__(self, img_path, tags_dict, mask_dict, image_size, degree):
        
        self.img_list = os.listdir(img_path)
        self.tags_dict = tags_dict
        self.mask_dict = mask_dict
        self.img_path = img_path
        self.image_size = image_size
        self.degree = degree

    def __getitem__(self, index):
        img_name = self.img_list[index]
        path = os.path.join(self.img_path, img_name)
        img = skimage.io.imread(path)
        # img = Image.open(path)
        # img = img.convert('RGB')

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Scale((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.Lambda(lambda data: np.asarray(data)),
            transforms.Lambda(lambda data: scipy.ndimage.rotate(data, random.uniform(-self.degree, self.degree), 
                reshape=False, mode='reflect')),
            transforms.Lambda(lambda data: data.reshape(self.image_size, self.image_size, 3)),
            transforms.ToTensor(),
            ])

        img = transform(img)
        # if self.transform is not None:
        #     img = self.transform(img)
        # else:
        #     img = tv.transforms.ToTensor()(img)
        label = torch.from_numpy(self.tags_dict[img_name])
        mask = torch.from_numpy(self.mask_dict[img_name])
        return img, label, mask

    def __len__(self):
        return len(self.img_list)

def np2Var(array, volatile=False, requires_grad=False):
    tensor = torch.from_numpy(array)
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return torch.autograd.Variable(tensor, volatile=volatile, 
        requires_grad=requires_grad)

def tensor2Var(tensor, volatile=False, requires_grad=False):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return torch.autograd.Variable(tensor, volatile=volatile, 
        requires_grad=requires_grad)

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

# def plot(x, y, x_name, y_name, file_name):
#     plt.clf()
#     plt.plot(x, y)
#     plt.xlabel(x_name)
#     plt.ylabel(y_name)
#     plt.savefig(file_name, dpi=96)

def load_tags(tags_name, style_dict):
    _id_list = []
    _tags_list = []
    _mask_list = []
    split_str = '\t|:|'
    i = 0.0
    no_hair = 0.0
    no_eyes = 0.0
    both_no = 0.0
    with open(tags_name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            split_line = line.rstrip().split(',', 1)
            _id = '%s.jpg' % split_line[0]
            _id_list.append(_id)

            split_line = re.split(split_str, split_line[1])
        
            # print(split_line)
            i += 1
            has_hair = 0
            has_eyes = 0
            style_array = np.zeros((len(style_dict), ), dtype=np.float32)
            mask_array = np.zeros((len(style_dict), ), dtype=np.float32)
            for split in split_line:
                if style_dict.get(split) != None:

                    if split[-4:] == 'hair' and not has_hair:
                        has_hair = 1
                        style_array[style_dict[split]] = 1.0
                        mask_array[:len(hair_color_list)] = 1.0
                    elif split[-4:] == 'eyes' and not has_eyes:
                        has_eyes = 1
                        style_array[style_dict[split]] = 1.0
                        mask_array[-len(eyes_color_list):] = 1.0

            if not has_hair:
                no_hair += 1.0
                # style_array[style_dict['unk hair']] = 1.0
                # print(split_line)
            if not has_eyes:
                no_eyes += 1.0
                # style_array[style_dict['unk eyes']] = 1.0
            if not has_hair and not has_eyes:
                both_no += 1.0
            _mask_list.append(mask_array)
            _tags_list.append(style_array)
    print(i)
    print('no hair: ', 100.0 * no_hair / i)
    print('no eyes: ', 100.0 * no_eyes / i)
    print('both no: ', 100.0 * both_no / i)
    tags_dict = {key: value for key, value in zip(_id_list, _tags_list)}
    mask_dict = {key: value for key, value in zip(_id_list, _mask_list)}
    return tags_dict, mask_dict

def load_te_tags(tags_name, style_dict):
    _id_list = []
    _tags_list = []
    split_str = ' '

    with open(tags_name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            split_line = line.rstrip().split(',', 1)
            _id = split_line[0]
            _id_list.append(_id)

            split_line = re.split(split_str, split_line[1])
        
            # print(split_line)
            style_array = np.zeros((len(style_dict), ), dtype=np.float32)
            has_hair = 0
            has_eyes = 0
            for i, split in enumerate(split_line):
                if split == 'hair' or split == 'eyes':
                    try:
                        style = split_line[i - 1] + ' ' + split
                        if split == 'hair' and not has_hair:
                            has_hair = 1
                            style_array[style_dict[style]] = 1.0
                        if split == 'eyes' and not has_eyes:
                            has_eyes = 1
                            style_array[style_dict[style]] = 1.0
                    except:
                        pass

            _tags_list.append(style_array)

    return np.stack(_id_list), np.stack(_tags_list)

def array_back_style(array, id2style):
    style_list = []

    for i in range(array.shape[0]):
        style = ''
        for j in range(array.shape[1]):
            if array[i, j] == 1.0:
                style += id2style[j]
                style += ' '
        if style == '':
            style = 'none'
        style = style.rstrip()
        style_list.append(style)
    return style_list


# hair_color_list = ['unk hair', 'orange hair', 'white hair', 'aqua hair', 'gray hair',
# 'green hair', 'red hair', 'purple hair', 'pink hair', 'blue hair', 'black hair', 
# 'brown hair', 'blonde hair']

# eyes_color_list = ['unk eyes', 'gray eyes', 'black eyes', 'orange eyes',
# 'pink eyes', 'yellow eyes', 'aqua eyes', 'purple eyes', 'green eyes', 'brown eyes', 
# 'red eyes', 'blue eyes']


def load_style():
    style_list = hair_color_list + eyes_color_list

    return style_list

def create_fake_tags(real_hair, real_eyes):
    hair_candidatas = random.sample(hair_color_list, 2)

    if hair_style2id[hair_candidatas[0]] != real_hair:
        fake_hair = hair_style2id[hair_candidatas[0]]
    else:
        fake_hair = hair_style2id[hair_candidatas[1]]

    eye_candidates = random.sample(eyes_color_list, 2)

    real_eyes -= len(hair_color_list)
    if eyes_style2id[eye_candidates[0]] != real_eyes:
        fake_eyes = eyes_style2id[eye_candidates[0]] + len(hair_color_list)
    else:
        fake_eyes = eyes_style2id[eye_candidates[1]] + len(hair_color_list)

    return fake_hair, fake_eyes

if __name__ == '__main__':
    # hair_color_list = ['unk hair', 'orange hair', 'white hair', 'aqua hair', 'gray hair',
    # 'green hair', 'red hair', 'purple hair', 'pink hair', 'blue hair', 'black hair', 
    # 'brown hair', 'blonde hair']

    # eyes_color_list = ['unk eyes', 'gray eyes', 'black eyes', 'orange eyes',
    # 'pink eyes', 'yellow eyes', 'aqua eyes', 'purple eyes', 'green eyes', 'brown eyes', 
    # 'red eyes', 'blue eyes']

    style_list = hair_color_list + eyes_color_list

    style2id, id2style = list2dict(style_list)
    # print(style_dict)

    tags_dict = load_tags('data/tags_clean.csv', style2id)

    # print(_tags_list[:3])

    # print(create_fake_tags(10, 14))


