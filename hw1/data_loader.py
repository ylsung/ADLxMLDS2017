import numpy as np

class DataLoader(object):
    def __init__(self, file_name, dtype, Transformer=None):
        super(DataLoader, self).__init__()
        self.file_name = file_name
        if dtype == 'data':
            with open(self.file_name, 'r') as f:
                lines = f.readlines()
                instance_id_list = []
                frame_length_list = []
                feature_list = []
                pre_instance_id = ''
                i = -1
                feature_num = 0
                for line in lines:
                    line_list = line.rstrip('\n').split(' ')
                    # get the speaker id list: [person id, sentence id, frame id]
                    speak_id_list = line_list.pop(0).split('_')
                    feature_num = len(line_list)
                    instance_id = speak_id_list[0] + '_' + speak_id_list[1]
                    line_list = list(map(float, line_list))
                    if pre_instance_id != instance_id:
                        instance_id_list.append(instance_id)
                        frame_length_list.append(1)
                        i += 1
                        feature_list.append([])
                        feature_list[i].append(line_list)
                        pre_instance_id = instance_id
                    else:
                        frame_length_list[i] += 1
                        feature_list[i].append(line_list)
                max_length = np.max(frame_length_list)
                feature_list_array = np.zeros(shape=(len(instance_id_list), max_length, feature_num))
                for i in range(len(feature_list)):
                    feature_list_array[i, :frame_length_list[i], :] = np.stack(feature_list[i])
            self.instance_id_array = np.char.asarray(instance_id_list)
            self.feature_array = feature_list_array.astype(np.float32)
            self.frame_length_array = np.asarray(frame_length_list).astype(np.int)
        elif dtype == 'label':
            with open(self.file_name, 'r') as f:
                lines = f.readlines()
                instance_id_list = []
                label_list = []
                pre_instance_id = ''
                i = -1
                for line in lines:
                    line_list = line.rstrip('\n').split(',')
                    speak_id_list = line_list.pop(0).split('_')
                    instance_id = speak_id_list[0] + '_' + speak_id_list[1]
                    label = Transformer.transform2num(line_list[0])
                    if pre_instance_id != instance_id:
                        i += 1
                        instance_id_list.append(instance_id)
                        label_list.append([])
                        label_list[i].append(label)
                        pre_instance_id = instance_id
                    else:
                        label_list[i].append(label)
            self.label_instance_id_array = np.char.asarray(instance_id_list)
            max_length = 0
            for labels in label_list:
                if len(labels) > max_length:
                    max_length = len(labels)
            self.label_array = np.ndarray(shape=(len(label_list), max_length), dtype=np.int)
            self.label_array.fill(0)
            for i in range(len(label_list)):
                self.label_array[i, :len(label_list[i])] = np.stack(label_list[i])

    def load_data(self):
        return self.instance_id_array, self.feature_array, self.frame_length_array
    def load_label(self):
        return self.label_instance_id_array, self.label_array

class Transformer(object):
    def __init__(self, _48_to_39_file, char_to_num_char_file):
        super(Transformer, self).__init__()
        _48_39_char = {}
        phone2num = {}
        num2char = {}
        with open(_48_to_39_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line_list = line.rstrip('\n').split('\t')
                _48_39_char[line_list[0]] = line_list[1] 

        with open(char_to_num_char_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line_list = line.rstrip('\n').split('\t')
                phone2num[line_list[0]] = line_list[1]
                num2char[line_list[1]] = line_list[2]
            for char in _48_39_char:
                transform_char = _48_39_char[char]
                num_transform_char = phone2num[transform_char]
                num_char = phone2num[char]
                num2char[num_char] = num2char[num_transform_char]
        self._48_39_char = _48_39_char
        self.phone2num = phone2num
        self.num2char = num2char
    def transform2num(self, char):
        return self.phone2num[char]
    def transform2char(self, number):
        return self.num2char[number]



if __name__ == '__main__':
    # d = DataLoader('./data/fbank/test.ark', dtype='data')
    # instance_id_array, feature_array, frame_length_array = d.get()
    t = Transformer('data/phones/48_39.map', 'data/48phone_char.map')
    d = DataLoader('./data/fbank/train.ark', dtype='data')
    l = DataLoader('data/label/train.lab', dtype='label', Transformer=t)
    a, b, c = d.load_data()
    e, f = l.load_label()
