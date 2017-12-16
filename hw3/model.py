import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np


class Model(nn.Module):
    def init(self, ):
        super(Model, self).__init__()

    def load(self, model_path):
        self.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    def save(self, model_path):
        torch.save(self.state_dict(), model_path) 

class DQN(Model):
    def __init__(self, image_s=84, dim=16, kernel_list=[8, 4], stride_list=[4, 2], o_s=4):
        super(DQN, self).__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(4, dim, kernel_size=kernel_list[0], stride=stride_list[0]),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim*2, kernel_size=kernel_list[1], stride=stride_list[1]),
            nn.BatchNorm2d(dim*2),
            nn.ReLU(True),
            nn.Conv2d(dim*2, dim*2, kernel_size=kernel_list[2], stride=stride_list[2]),
            nn.BatchNorm2d(dim*2),
            nn.ReLU(True),
            )

        dim *= 2
        t_image_s = image_s
        for i in range(len(kernel_list)):
            t_image_s = math.ceil((t_image_s - (kernel_list[i] - 1)) / stride_list[i])

        linear_size = t_image_s ** 2 * dim
        linear_dim = 512
        self.output = nn.Sequential(
            nn.Linear(linear_size, linear_dim),
            nn.ReLU(True),
            nn.BatchNorm1d(linear_dim),
            nn.Linear(linear_dim, o_s)
            )
        # self.output = nn.Sequential(
        #     nn.Linear(linear_size, o_s)
        #     )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data, a=1)
                m.bias.data.fill_(0.0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight.data, a=1)
                m.bias.data.fill_(0.0)   
        # print(linear_size)

    def forward(self, inputs):
        conv_feature = self.feature(inputs)
        flat_feature = conv_feature.view(conv_feature.size(0), -1)
        # print(flat_feature.size())

        return self.output(flat_feature)
class PG(Model):
    def __init__(self, image_s=84, dim=16, kernel_list=[8, 4], stride_list=[4, 2], o_s=4):
        super(PG, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(1, dim, kernel_size=kernel_list[0], stride=stride_list[0]),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim*2, kernel_size=kernel_list[1], stride=stride_list[1]),
            nn.BatchNorm2d(dim*2),
            nn.ReLU(True),
            # nn.Conv2d(dim*2, dim*2, kernel_size=kernel_list[2], stride=stride_list[2]),
            # nn.BatchNorm2d(dim*2),
            # nn.ReLU(True),
            )

        dim *= 2
        t_image_s = image_s
        for i in range(len(kernel_list)):
            t_image_s = math.ceil((t_image_s - (kernel_list[i] - 1)) / stride_list[i])

        linear_size = t_image_s ** 2 * dim
        linear_dim = 128
        # linear_size = image_s ** 2
        # linear_dim = 256
        self.output = nn.Sequential(
            nn.Linear(linear_size, linear_dim),
            nn.ReLU(True),
            nn.BatchNorm1d(linear_dim),
            nn.Linear(linear_dim, o_s),
            nn.Softmax(),
            )
        # self.output = nn.Sequential(
        #     nn.Linear(linear_size, o_s)
        #     )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data, a=1)
                m.bias.data.fill_(0.0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight.data, a=1)
                m.bias.data.fill_(0.0)
    def forward(self, inputs):
        conv_feature = self.feature(inputs)
        flat_feature = conv_feature.view(conv_feature.size(0), -1)
        # print(flat_feature.size()) 
        # flat_feature = inputs.view(inputs.size(0), -1)
        return self.output(flat_feature)  

class duelDQN(Model):
    def __init__(self, image_s=84, dim=16, kernel_list=[8, 4], stride_list=[4, 2], o_s=4):
        super(duelDQN, self).__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(4, dim, kernel_size=kernel_list[0], stride=stride_list[0]),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim*2, kernel_size=kernel_list[1], stride=stride_list[1]),
            nn.BatchNorm2d(dim*2),
            nn.ReLU(True),
            nn.Conv2d(dim*2, dim*2, kernel_size=kernel_list[2], stride=stride_list[2]),
            nn.BatchNorm2d(dim*2),
            nn.ReLU(True),
            )

        dim *= 2
        t_image_s = image_s
        for i in range(len(kernel_list)):
            t_image_s = math.ceil((t_image_s - (kernel_list[i] - 1)) / stride_list[i])

        linear_size = t_image_s ** 2 * dim
        linear_dim = 512
        self.output_A = nn.Sequential(
            nn.Linear(linear_size, linear_dim),
            nn.ReLU(True),
            nn.BatchNorm1d(linear_dim),
            nn.Linear(linear_dim, o_s)
            )
        self.output_V = nn.Sequential(
            nn.Linear(linear_size, linear_dim),
            nn.ReLU(True),
            nn.BatchNorm1d(linear_dim),
            nn.Linear(linear_dim, 1)
            )
        # self.output = nn.Sequential(
        #     nn.Linear(linear_size, o_s)
        #     )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data, a=1)
                m.bias.data.fill_(0.0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight.data, a=1)
                m.bias.data.fill_(0.0)   
        # print(linear_size)

    def forward(self, inputs):
        conv_feature = self.feature(inputs)
        flat_feature = conv_feature.view(conv_feature.size(0), -1)
        # print(flat_feature.size())

        return self.output_V(flat_feature) + self.output_A(flat_feature)


class A2C(Model):
    def __init__(self, image_s=84, dim=16, kernel_list=[8, 4], stride_list=[4, 2], o_s=4):
        super(A2C, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(1, dim, kernel_size=kernel_list[0], stride=stride_list[0]),
            # nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim*2, kernel_size=kernel_list[1], stride=stride_list[1]),
            # nn.BatchNorm2d(dim*2),
            nn.ReLU(True),
            # nn.Conv2d(dim*2, dim*2, kernel_size=kernel_list[2], stride=stride_list[2]),
            # nn.BatchNorm2d(dim*2),
            # nn.ReLU(True),
            )

        dim *= 2
        t_image_s = image_s
        for i in range(len(kernel_list)):
            t_image_s = math.ceil((t_image_s - (kernel_list[i] - 1)) / stride_list[i])

        linear_size = t_image_s ** 2 * dim
        linear_dim = 128

        self.output_pi = nn.Sequential(
            nn.Linear(linear_size, linear_dim),
            nn.ReLU(True),
            # nn.BatchNorm1d(linear_dim),
            nn.Linear(linear_dim, o_s),
            nn.Softmax(),
            )
        self.output_V = nn.Sequential(
            nn.Linear(linear_size, linear_dim),
            nn.ReLU(True),
            # nn.BatchNorm1d(linear_dim),
            nn.Linear(linear_dim, 1),
            )
        # self.output = nn.Sequential(
        #     nn.Linear(linear_size, o_s)
        #     )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data, a=1)
                m.bias.data.fill_(0.0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight.data, a=1)
                m.bias.data.fill_(0.0)
    def forward(self, inputs):
        conv_feature = self.feature(inputs)
        flat_feature = conv_feature.view(conv_feature.size(0), -1)
        # print(flat_feature.size()) 
        # flat_feature = inputs.view(inputs.size(0), -1)
        return self.output_pi(flat_feature), self.output_V(flat_feature)  

class lstmA2C(Model):
    def __init__(self, image_s=84, dim=16, kernel_list=[8, 4], stride_list=[4, 2], o_s=4):
        super(lstmA2C, self).__init__()

        self.conv1 = nn.Conv2d(1, dim, kernel_size=kernel_list[0], stride=stride_list[0])
        self.conv2 = nn.Conv2d(dim, dim*2, kernel_size=kernel_list[1], stride=stride_list[1])
        dim *= 2
        t_image_s = image_s
        for i in range(len(kernel_list)):
            t_image_s = math.ceil((t_image_s - (kernel_list[i] - 1)) / stride_list[i])

        linear_size = t_image_s ** 2 * dim
        
        hidden_size = 256
        self.hidden_size = 256

        self.num_layers=1
        self.direction=2

        self.lstm = nn.LSTMCell(linear_size, hidden_size)

        self.output_pi = nn.Linear(hidden_size, o_s)
        self.output_V = nn.Linear(hidden_size, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data, a=1)
                m.bias.data.fill_(0.0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight.data, a=1)
                m.bias.data.fill_(0.0)
    def forward(self, inputs, h, c):
        conv_feature = F.elu(self.conv1(inputs))
        conv_feature = F.elu(self.conv2(conv_feature))
        flat_feature = conv_feature.view(conv_feature.size(0), -1)
        # flat_feature = flat_feature.unsqueeze(1)

        h, c = self.lstm(flat_feature, (h, c))


        # output = output.view(output.size(0), -1)

        return self.output_pi(h), self.output_V(h), h, c
    def init_hidden(self, batch_size):
        h = torch.zeros(batch_size, self.hidden_size)
        c = torch.zeros(batch_size, self.hidden_size)

        return Variable(h), Variable(c)

def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class ActorCritic(Model):
    def __init__(self, num_inputs, action_space):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.lstm = nn.LSTMCell(32 * 5 * 5, 256)

        num_outputs = action_space
        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear = nn.Linear(256, num_outputs)

        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))

        x = x.view(-1, 32 * 5 * 5)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        return self.critic_linear(x), self.actor_linear(x), (hx, cx)

