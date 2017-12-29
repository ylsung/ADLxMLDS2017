from agent_dir.agent import Agent
import scipy.misc
import numpy as np
from collections import namedtuple

import os
from collections import deque
from utils import create_logger, save_pickle, load_pickle
from model import PG

import torch
from torch.autograd import Variable
import random

ImageSize = 80

## os.environ["CUDA_VISIBLE_DEVICES"]="6"
class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
    
        super(Agent_PG,self).__init__(env)
        self.args = args
        self.memory = []
        self.reward_queue = deque(maxlen=30)
        self.discount_reward_queue = deque(maxlen=30)
        if args.save_path != '':
            split_path = os.path.split(args.save_path)
            if not os.path.exists(split_path[0]):
                os.mkdir(split_path[0])
            if not os.path.exists(args.save_path):
                os.mkdir(args.save_path)

        self.model = PG(
            image_s=ImageSize,
            dim=16,
            kernel_list=[8, 4],
            stride_list=[4, 2],
            o_s=6,
            )

        args.load_path = 'ADLxMLDS2017_hw3_model/pg_batch1_rms1e-4'

        args.model_id = '-1'

        self.model_path = os.path.join(args.load_path, 'model' + args.model_id + '.pth')
        if args.test_pg:
            self.logger = create_logger(args.save_path, 'test', False)
            self.model.load(self.model_path)

            self.logger.info('load model %s' % self.model_path)
            #you can load your model here
            print('loading trained model')
            self.logger.info(self.model)
        elif args.train_pg:
            self.logger = create_logger(args.save_path, 'train', args.keep_training)
            if args.keep_training:
                self.model.load(self.model_path)
                self.reward_queue = load_pickle(args.load_path, 'reward%d' % args.start_game)
            else:
                print(self.args)
                self.logger.info(self.model)
        if torch.cuda.is_available():
            self.model.cuda()

        ##################
        # YOUR CODE HERE #
        ##################


    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        self.first = 1
        pass


    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        train_pg(self, )

        pass


    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        self.model.eval()

        if self.first:
            now_state = prepro(observation) - prepro(observation)
            self.pre_state = observation
            self.first = 0
            
        else:
            now_state = prepro(observation) - prepro(self.pre_state)
            self.pre_state = observation

        self.model.eval()


        state_v = np2var(np.expand_dims(now_state, axis=0), volatile=True)
        pred = self.model(state_v)
        # _, action = torch.topk(pred, 1, dim=1)
        prob = pred.cpu().data.numpy().reshape((-1, ))

        action = np.random.choice(a=6, p=prob)
        self.model.train()
        return action

    def random_action(self):
        return self.env.get_random_action()

def np2var(array, volatile=False, requires_grad=False):
    tensor = torch.from_numpy(array)

    if torch.cuda.is_available():
        tensor = tensor.cuda()

    return Variable(tensor, volatile=volatile, requires_grad=requires_grad)
def phi(image_input):
    image_input = image_input[34: 194, 15:145, :]
    image_output = image_input.astype(np.float32) / 255.0
    return image_output

def prepro(o,image_size=[ImageSize,ImageSize]):
    """
    Call this function to preprocess RGB image to grayscale image if necessary
    This preprocessing code is from
        https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/async_agent.py
    
    Input: 
    RGB image: np.array
        RGB screen of game, shape: (210, 160, 3)
    Default return: np.array 
        Grayscale image, shape: (80, 80, 1)
    
    """
    y = 0.2126 * o[:, :, 0] + 0.7152 * o[:, :, 1] + 0.0722 * o[:, :, 2]
    y = y.astype(np.uint8)
    resized = scipy.misc.imresize(y, image_size)
    return np.expand_dims(resized.astype(np.float32),axis=0) / 255.0

def exploration_rate(args, now_game):
    eps_start = args.start_exploration
    eps_end = args.end_exploration
    eps_decay = args.decay_exploration

    return eps_end + (eps_start - eps_end) * np.exp(-1.0 * float(now_game) / eps_decay)

Transition = namedtuple('Transition',
            ('state', 'action', 'reward'))
def train_pg(agent):
    optimizer = torch.optim.RMSprop(agent.model.parameters(), lr=agent.args.lr)
    # optimizer = torch.optim.Adam(agent.model.parameters(), lr=agent.args.lr)
    best_record = 0.0
    state_list = []
    next_state_list = []
    action_list = []
    reward_list = []
    for game in range(agent.args.games):
        state = agent.env.reset()
        agent.init_game_setting()
        done = False
        episode_reward = 0.0

        #playing one game
        loss = 0.0

        # explore = exploration_rate(agent.args, 
        #     game + agent.args.start_game - agent.args.learning_start)

        while(not done):

            action = agent.make_action(state, test=False)


            next_state, reward, done, info = agent.env.step(action)
            
            if done:
                next_state_list.insert(0, state_list[0])
                state_list.insert(0, state_list[0])
            else:
                state_list.append(prepro(state))
                next_state_list.append(prepro(next_state))
            action_list.append(action)
            reward_list.append(reward)
            
            episode_reward += reward
            state = next_state

        agent.reward_queue.append(episode_reward)


        loss = run_iteration(agent, optimizer, state_list, next_state_list, 
            action_list, reward_list)

        del state_list[:], next_state_list[:], action_list[:], reward_list[:]

        reward_per_game = np.mean(agent.reward_queue)
        agent.logger.info('game %d, tr avg reward: %4f, loss: %4f' % (
            game + agent.args.start_game, reward_per_game, loss))
        if episode_reward > best_record:
            best_record = episode_reward
            model_path = os.path.join(agent.args.save_path, 'model-1.pth')
            agent.model.save(model_path)

        if game % 100 == 0 and game != 0:
            model_path = os.path.join(agent.args.save_path, 'model%d.pth' % \
                (game + agent.args.start_game))
            agent.model.save(model_path)

            save_pickle(agent.args.save_path, 'reward%d'% (game + agent.args.start_game), 
                agent.reward_queue)

def convert_memory2numpy(memory, time):
    state_array = np.zeros(shape=(len(memory), 1, ImageSize, ImageSize), dtype=np.float32)
    mask = np.zeros(shape=(len(memory), 1), dtype=np.float32)
    reward_array = np.zeros(shape=(len(memory), 1), dtype=np.float32)
    action_array = np.zeros(shape=(len(memory), 1), dtype=int)
    for i, piece in enumerate(memory):
        try:
            if time == 0:
                state_array[i] = prepro(piece[time].state)
            else:
                state_array[i] = prepro(piece[time].state - piece[time - 1].state)
            mask[i] = 1.0
            reward_array[i] = piece[time].reward
            action_array[i] = piece[time].action
        except:
            pass
    return state_array, mask, reward_array, action_array

# def run_iteration(agent, optimizer):
#     R = 0.0
#     sum_prob = 0.0
#     # print(len(max(agent.memory, key=len)))
#     for time in range(len(max(agent.memory, key=len)) -1, -1, -1):
#         # print(time)
#         state_array, mask, reward_array, action_array = \
#             convert_memory2numpy(agent.memory, time)
#         # if time != 0:
#         #     state_array = trajectory[time].state - trajectory[time - 1].state
#         # else:
#         #     state_array = trajectory[time].state
#         # state_array = prepro(state_array)
#         # reward_array = np.zeros(shape=(1, 1), dtype=np.float32)
#         # action_array = np.zeros(shape=(1, 1), dtype=int)
#         # reward_array[0] = trajectory[time].reward
#         # action_array[0] = trajectory[time].action


#         state_v = np2var(state_array)

#         mask_v = np2var(mask)
#         reward_v = np2var(reward_array)
#         action_v = np2var(action_array)
#         log_prob_all = agent.model(state_v)
#         # print(log_prob_all)

#         log_prob = log_prob_all.gather(1, action_v)

#         R = R * mask_v * agent.args.gamma + reward_v
#         sum_prob += log_prob * mask_v
#     length = np.zeros(shape=(len(agent.memory), 1), dtype=np.float32)
#     for i in range(len(agent.memory)):
#         length[i] = len(agent.memory[i])
#     length_v = np2var(length)
#     # sum_prob /= float(len(agent.memory))
#     # R = (R - torch.mean(R)) / (torch.std(R) + 1e-8)
#     R = R / length_v
#     loss = -(sum_prob * R).mean()

#     agent.model.zero_grad()
#     loss.backward()

#     optimizer.step()

#     return loss.cpu().data.numpy()[0]

def discount_reward(reward_list, gamma):
    discount_r = np.zeros_like(reward_list).astype(np.float32)
    running_add = 0.0

    for time in reversed(range(len(reward_list))):
        if reward_list[time] != 0:
            running_add = 0.0
        running_add = running_add * gamma + reward_list[time]
        discount_r[time] = running_add
    return discount_r

def run_iteration(agent, optimizer, state_list, next_state_list, action_list, reward_list):
    R = 0.0
    sum_prob = 0.0
    # print(len(max(agent.memory, key=len)))
    loss = 0.0
    
    discount_r_array = discount_reward(reward_list, agent.args.gamma).reshape(-1, 1)
    state_array = np.stack(state_list)
    next_state_array = np.stack(next_state_list)
    action_array = np.vstack(action_list)

    # normalize discount reward
    discount_r_array = (discount_r_array - discount_r_array.mean()) / (discount_r_array.std() + 1e-8)

    state_array = next_state_array - state_array


    state_v = np2var(state_array)

    # mask_v = np2var(mask)
    reward_v = np2var(discount_r_array)
    action_v = np2var(action_array)

    log_prob_all = torch.log(agent.model(state_v))

    # print(log_prob_all)

    log_prob = log_prob_all.gather(1, action_v)

    loss = -torch.sum(log_prob * reward_v)


        # R = R * mask_v * agent.args.gamma + reward_v
        # sum_prob += log_prob * mask_v
    # length = np.zeros(shape=(1, ), dtype=np.float32)

    # length[0] = len(trajectory)
    # length_v = np2var(length)
    # sum_prob /= float(len(agent.memory))
    # R = (R - torch.mean(R)) / (torch.std(R) + 1e-8)
    # R = R / length_v
    # loss = -(sum_prob * R).mean()
    # loss = loss / length_v

    agent.model.zero_grad()
    loss.backward()

    optimizer.step()

    return loss.cpu().data.numpy()[0]







