import numpy as np
import os
from agent_dir.agent import Agent
from replay_memory import ReplayMemory
from collections import deque
from utils import create_logger, save_pickle, load_pickle
from model import DQN
import random
from replay_memory import Transition

import torch
from torch.autograd import Variable
import torch.nn.functional as F


## os.environ["CUDA_VISIBLE_DEVICES"] = '6'

########### my argument #########
# batch_size
# lr
# start_exploration
# end_exploration
# decay_exploration
# gamma
# load_path
# save_path
# memory_size
# games
# start_game
# keep_training
# model_id
#################################


class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_DQN, self).__init__(env)

        self.image_s = 84

        self.args = args
        self.memory = ReplayMemory(args.memory_size)
        self.reward_queue = deque(maxlen=30)
        self.model = DQN(image_s=self.image_s,
            dim=32,
            kernel_list=[8, 4, 3],
            stride_list=[4, 2, 1],
            o_s=4)
        # target network is not training during play.
        self.target_model = DQN(image_s=self.image_s,
            dim=32,
            kernel_list=[8, 4, 3],
            stride_list=[4, 2, 1],
            o_s=4)

        if args.save_path != '':
            split_path = os.path.split(args.save_path)
            if not os.path.exists(split_path[0]):
                os.mkdir(split_path[0])
            if not os.path.exists(args.save_path):
                os.mkdir(args.save_path)

        args.load_path = 'ADLxMLDS2017_hw3_model/dqn_henorm_replay100000_target50_exp2000_lstart1000_freq4'

        args.model_id = '-1'

        self.model_path = os.path.join(args.load_path, 'model' + args.model_id + '.pth')
        if args.test_dqn:
            self.logger = create_logger(args.save_path, 'test', False)
            self.model.load(self.model_path)

            self.logger.info('load model %s' % self.model_path)
            #you can load your model here
            print('loading trained model')
            self.logger.info(self.model)
        elif args.train_dqn:
            self.logger = create_logger(args.save_path, 'train', args.keep_training)
            if args.keep_training:
                self.model.load(self.model_path)
                self.reward_queue = load_pickle(args.load_path, 'reward%d' % args.start_game)
            else:
                print(self.args)
                self.logger.info(self.model)

        if torch.cuda.is_available():
            self.model.cuda()
            self.target_model.cuda()

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
        pass


    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################

        train_dqn(self)



    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        self.model.eval()
        if test:
            if random.uniform(0, 1) < 0.01:
                self.model.train()
                return self.random_action()
            else:
                observation = np.transpose(observation, (2, 0, 1))
                observation = observation.reshape((-1, 4, self.image_s, self.image_s))
                state_v = np2var(observation, volatile=True)
                pred = self.model(state_v)
                _, action = torch.max(pred, dim=1)
                self.model.train()

                return action.cpu().data.numpy()[0]
            # return self.env.get_random_action()
        else:
            # if random.uniform(0, 1) < self.args.exploration:
            #     self.model.train()
            #     return random.randint(0, 3)

            observation = np.transpose(observation, (2, 0, 1))
            observation = observation.reshape((-1, 4, self.image_s, self.image_s))
            state_v = np2var(observation, volatile=True)
            pred = self.model(state_v)
            _, action = torch.max(pred, dim=1)
            self.model.train()

            return action.cpu().data.numpy()[0]

    def random_action(self):
        return self.env.get_random_action()


#################################
## train functions
#################################
#################################

def np2var(array, volatile=False, requires_grad=False):
    tensor = torch.from_numpy(array)

    if torch.cuda.is_available():
        tensor = tensor.cuda()

    return Variable(tensor, volatile=volatile, requires_grad=requires_grad)

def run_iter(agent, optimizer):

    # random sample from Memory
    if agent.args.batch_size > len(agent.memory):
        return 0.0
    else:
        samples = agent.memory.sample(agent.args.batch_size)

    batch = Transition(*zip(*samples))

    batch_state = np.stack(batch.state)

    temp_next_state = [s if s is not None else np.zeros(shape=(84, 84, 4), dtype=np.float32) \
        for s in batch.next_state]
    
    batch_mask = 1.0 - np.vstack(batch.done).astype(np.float32)


    batch_next_state = np.stack(temp_next_state)
    batch_action_v = np2var(np.vstack(batch.action))
    batch_reward_v = np2var(np.vstack(batch.reward).astype(np.float32))


    batch_state = np.transpose(batch_state, (0, 3, 1, 2))

    batch_state_v = np2var(batch_state)

    # compute computed Q(s)

    all_computed_q_s = agent.model(batch_state_v)
    
    # compute computed Q(s,a)

    computed_q_s = all_computed_q_s.gather(1, batch_action_v)

    # compute maximum next state Q*(next_s, a')

    batch_next_state = np.transpose(batch_next_state, (0, 3, 1, 2))

    batch_next_state_v = np2var(batch_next_state)

    all_ideal_next_q_s = agent.target_model(batch_next_state_v).detach()

    batch_mask_v = np2var(batch_mask)
    # print(batch_mask_v)


    max_next_q_s, _ = torch.max(all_ideal_next_q_s, dim=1)
    max_next_q_s = max_next_q_s.view(-1, 1)

    ideal_q_s = agent.args.gamma * max_next_q_s * batch_mask_v + batch_reward_v

    agent.model.zero_grad()

    # loss = F.smooth_l1_loss(computed_q_s, ideal_q_s)
    loss = F.mse_loss(computed_q_s, ideal_q_s)
    # loss = loss = F.l1_loss(computed_q_s, ideal_q_s)
    # loss = loss.clamp(-1, 1)
    loss.backward()

    # for p in agent.model.parameters():
    #     p.grad.data.clamp_(-1, 1)

    optimizer.step()

    return loss.cpu().data.numpy()[0]

def exploration_rate(args, now_game, method='exp'):
    eps_start = args.start_exploration
    eps_end = args.end_exploration
    eps_decay = args.decay_exploration
    if method == 'exp':

        return eps_end + (eps_start - eps_end) * np.exp(-1.0 * float(now_game) / eps_decay)
    elif method == 'tanh':
        offset = 3.0 * eps_decay
        return eps_end + (eps_start - eps_end) * \
            (1.0 - np.tanh((float(now_game) - offset) / eps_decay)) / 2.0



def train_dqn(agent, ):
    optimizer = torch.optim.RMSprop(agent.model.parameters(), lr=agent.args.lr, alpha=0.95, eps=0.01)
    best_record = 0.0
    i = 0
    update = 0
    for game in range(agent.args.games):

        state = agent.env.reset()
        agent.init_game_setting()
        done = False
        episode_reward = 0.0

        #playing one game
        loss = 0.0
        explore = exploration_rate(agent.args, 
                game + agent.args.start_game - agent.args.learning_start, 'exp')
        while(not done):
            if agent.args.learning_start > game and not agent.args.keep_training:
                action = agent.random_action()
            elif random.uniform(0, 1) < explore:
                action = agent.random_action()
            else:
                action = agent.make_action(state, test=False)

            next_state, reward, done, info = agent.env.step(action)

            agent.memory.push(state, action, reward, next_state, done)

            if i % agent.args.online_update_freq == 0:
                loss = run_iter(agent, optimizer)

            state = next_state

            episode_reward += reward

            i += 1

            # print(action)
        if game % agent.args.update_target == 0:
            agent.target_model.load_state_dict(agent.model.state_dict())

        agent.reward_queue.append(episode_reward)

        reward_per_game = np.mean(agent.reward_queue)
        agent.logger.info('game: %d, step: %d, tr avg reward: %4f, explore: %4f' % (
            game + agent.args.start_game, i, reward_per_game, explore))

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


