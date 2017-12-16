import tensorflow as tf
import numpy as np
from scipy.misc import imresize
from functools import partial
from typing import Iterable
import scipy.misc

import time

from copy import deepcopy
from agent_dir.agent import Agent
from collections import deque

from utils import create_logger
input_shape = [80, 80, 1]
output_dim = 6

# os.environ["CUDA_VISIBLE_DEVICES"]="4"
def resize_image(image, new_HW):
    """Returns a resize image
    
    Args:
        image (3-D Array): RGB Image Array of shape (H, W, C)
        new_HW (tuple, optional): New Height and Width (height, width)
    
    Returns:
        3-D Array: A resized image of shape (`height`, `width`, C)
    """
    return imresize(image, new_HW, interp='nearest')


def crop_ROI(image, height_range=(35, 193), width_range=(0, 160)):
    """Crops a region of interest (ROI)
    
    Args:
        image (3-D Array): RGB Image of shape (H, W, C)
        height_range (tuple, optional): Height range to keep (h_begin, h_end)
        width_range (tuple, optional): Width range to keep (w_begin, w_end)
    
    Returns:
        3-D array: Cropped image of shape (h_end - h_begin, w_end - w_begin, C)
    """
    h_beg, h_end = height_range
    w_beg, w_end = width_range
    return image[h_beg:h_end, w_beg:w_end, ...]

ImageSize = 80

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
    return np.expand_dims(resized.astype(np.float32),axis=2) / 255.0


def pipeline(image, new_HW):
    """Image process pipeline
    
    Args:
        image (3-D Array): 3-D array of shape (H, W, C)
        new_HW (tuple): New height and width int tuple of (height, width)
    
    Returns:
        3-D Array: Binarized image of shape (height, width, 1)
    """
    image = crop_ROI(image, (35, 193))
    image = resize_image(image, (80, 80))
    image = kill_background_grayscale(image, (144, 72, 17))
    image = np.expand_dims(image, axis=2)

    # image = (image - np.mean(image)) / (np.std(image) + 1e-8)

    return image
    # return prepro(image)
def kill_background_grayscale(image, bg):
    """Make the background 0
    Args:
        image (3-D array): Numpy array (H, W, C)
        bg (tuple): RGB code of background (R, G, B)
    Returns:
        image (2-D array): Binarized image of shape (H, W)
            The background is 0 and everything else is 1
    """
    H, W, _ = image.shape

    R = image[..., 0]
    G = image[..., 1]
    B = image[..., 2]

    cond = (R == bg[0]) & (G == bg[1]) & (B == bg[2])

    image = np.zeros((H, W))
    image[~cond] = 1

    return image

def discount_rewards(rewards, gamma):
    """Discount rewards by a `gamma`
    
    Args:
        rewards (1-D Array): Reward array of shape (N,)
        gamma (float, optional): Discount Rate
    
    Returns:
        1-D Array: Discounted Reward array of shape (N,)
    """
    discounted = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0

    for i in reversed(range(len(rewards))):
        if rewards[i] != 0:
            running_add = 0
        running_add = rewards[i] + gamma * running_add
        discounted[i] = running_add
    return discounted


def discount_multi_rewards(multi_rewards, gamma):
    """
    Args:
        multi_rewards (2-D Array): Reward array of shape (n_envs, n_timesteps)
        gamma (float, optional): Discount rate for a reward
    
    Returns:
        discounted_multi_rewards (2-D Array): Reward array of shape (n_envs, n_timesteps)
    """
    n_envs = len(multi_rewards)
    discounted = []
    for id in range(n_envs):
        discounted.append(discount_rewards(multi_rewards[id], gamma))
    return discounted







class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
    
        super(Agent_PG,self).__init__(env)
        self.args = args

        self.reward_queue = deque(maxlen=30)

        self.input_shape = input_shape
        self.output_dim = output_dim
        self.__build_network(self.input_shape, self.output_dim)
        if args.train_a3c:
            self.logger = create_logger(args.logdir, 'train', 'False')
        if args.test_a3c:
            self.logger = create_logger(args.logdir, 'test', 'False')
            saver = tf.train.Saver()
            latest_checkpoint = tf.train.latest_checkpoint(self.args.logdir)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth=True

            self.sess = tf.Session(config=config)
            if latest_checkpoint is not None:
                saver.restore(self.sess, latest_checkpoint)
                self.logger.info("Restored from {}".format(latest_checkpoint))
            self.pipeline_fn = partial(pipeline, new_HW=input_shape[:-1])

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


    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################

        train_a3c(self)



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

        s = self.pipeline_fn(observation)
        actions = self.get_actions(s, self.sess)

        return actions[0]

    def random_action(self):
        return self.env.get_random_action()

    def __build_network(self, input_shape: list, output_dim: int):
        """Create a basic network architecture """

        self.states = tf.placeholder(tf.float32, shape=[None, *input_shape], name="states")
        self.actions = tf.placeholder(tf.uint8, shape=[None], name="actions")
        action_onehots = tf.one_hot(self.actions, depth=output_dim, name="action_onehots")
        self.rewards = tf.placeholder(tf.float32, shape=[None], name="rewards")
        self.advantages = tf.placeholder(tf.float32, shape=[None], name="advantages")

        net = self.states

        with tf.variable_scope("layer1"):
            net = tf.layers.conv2d(net, filters=16, kernel_size=(8, 8), strides=(4, 4), name="conv")
            net = tf.nn.relu(net, name="relu")

        with tf.variable_scope("layer2"):
            net = tf.layers.conv2d(net, filters=32, kernel_size=(4, 4), strides=(2, 2), name="conv")
            net = tf.nn.relu(net, name="relu")

        net = tf.contrib.layers.flatten(net)

        with tf.variable_scope("fc1"):
            net = tf.layers.dense(net, units=256, name="fc")
            net = tf.nn.relu(net, name="relu")

        with tf.variable_scope("action_network"):
            action_scores = tf.layers.dense(net, units=output_dim, name="action_scores")
            self.action_probs = tf.nn.softmax(action_scores, name="action_probs")
            single_action_prob = tf.reduce_sum(self.action_probs * action_onehots, axis=1)
            log_action_prob = - tf.log(single_action_prob + 1e-7) * self.advantages
            action_loss = tf.reduce_sum(log_action_prob)

        with tf.variable_scope("entropy"):
            entropy = - tf.reduce_sum(self.action_probs * tf.log(self.action_probs + 1e-7), axis=1)
            entropy_sum = tf.reduce_sum(entropy)

        with tf.variable_scope("value_network"):
            self.values = tf.squeeze(tf.layers.dense(net, units=1, name="values"))
            value_loss = tf.reduce_sum(tf.squared_difference(self.rewards, self.values))

        with tf.variable_scope("total_loss"):
            self.loss = action_loss + value_loss * 0.5 - entropy_sum * self.args.entropy_loss_coeff

        with tf.variable_scope("train_op"):
            self.optim = tf.train.AdamOptimizer(learning_rate=self.args.lr)
            gradients = self.optim.compute_gradients(loss=self.loss)
            gradients = [(tf.clip_by_norm(grad, 50.0), var) for grad, var in gradients]
            self.train_op = self.optim.apply_gradients(gradients,
                                                       global_step=tf.train.get_or_create_global_step())

        tf.summary.histogram("Action Probs", self.action_probs)
        tf.summary.histogram("Entropy", entropy)
        tf.summary.histogram("Actions", self.actions)
        tf.summary.scalar("Loss/total", self.loss)
        tf.summary.scalar("Loss/actor", action_loss)
        tf.summary.scalar("Loss/value", value_loss)
        tf.summary.image("Screen", tf.gather(self.states[:, :, :, -1:], tf.random_uniform(shape=[3, ],
                                                                                          minval=0,
                                                                                          maxval=tf.shape(self.states)[0],
                                                                                          dtype=np.int32)))

        self.summary_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter("{}/main".format(self.args.logdir), graph=tf.get_default_graph())

    def get_actions(self, states, sess=None):
        """Get actions given states
        Args:
            states (4-D Array): States Array of shape (N, H, W, C)
        
        Returns:
            actions (1-D Array): Action Array of shape (N,)
        """
        if sess == None:
            sess = tf.get_default_session()

        feed = {
            self.states: np.reshape(states, [-1, *self.input_shape])
        }
        action_probs = sess.run(self.action_probs, feed)
        noises = np.random.uniform(size=action_probs.shape[0])[:, np.newaxis]

        return (np.cumsum(action_probs, axis=1) > noises).argmax(axis=1)

    def get_values(self, states):
        """Get values given states
        Args:
            states (4-D Array): States Array of shape (N, H, W, C)
        
        Returns:
            values (1-D Array): Values (N,)
        """
        sess = tf.get_default_session()
        feed = {
            self.states: np.reshape(states, [-1, *self.input_shape])
        }
        return sess.run(self.values, feed).reshape(-1)

    def get_actions_values(self, states):
        """Get actions and values given states
        
        Args:
            states (4-D Array): States Array of shape (N, H, W, C)
        
        Returns:
            actions (1-D Array): Action Array of shape (N,)
            values (1-D Array): Values (N,)
        """
        sess = tf.get_default_session()
        feed = {
            self.states: states,
        }

        action_probs, values = sess.run([self.action_probs, self.values], feed)
        noises = np.random.uniform(size=action_probs.shape[0])[:, np.newaxis]

        return (np.cumsum(action_probs, axis=1) > noises).argmax(axis=1), values.flatten()

    def train_iter(self, states, actions, rewards, values):
        """Update parameters by gradient descent
        
        Args:
            states (5-D Array): Image arrays of shape (n_envs, n_timesteps, H, W, C)
            actions (2-D Array): Action arrays of shape (n_envs, n_timesteps)
            rewards (2-D Array): Rewards array of shape (n_envs, n_timesteps)
            values (2-D Array): Value array of shape (n_envs, n_timesteps)
        """

        states = np.vstack([s for s in states if len(s) > 0])
        actions = np.hstack(actions)
        values = np.hstack(values)

        rewards = discount_multi_rewards(rewards, self.args.gamma)
        rewards = np.hstack(rewards)
        rewards -= np.mean(rewards)
        rewards /= np.std(rewards) + 1e-7

        advantages = rewards - values
        advantages -= np.mean(advantages)
        advantages /= np.std(advantages) + 1e-7

        sess = tf.get_default_session()
        feed = {
            self.states: states,
            self.actions: actions,
            self.rewards: rewards,
            self.advantages: advantages
        }
        _, summary_op, global_step = sess.run([self.train_op,
                                               self.summary_op,
                                               tf.train.get_global_step()],
                                              feed_dict=feed)
        self.summary_writer.add_summary(summary_op, global_step=global_step)

def run_episodes(envs, agent: Agent, t_max=50, pipeline_fn=pipeline,episo= 1):
    """Run multiple environments and update the agent
    
    Args:
        envs (Iterable[gym.Env]): A list of gym environments
        agent (Agent): Agent class
        t_max (int, optional): Number of steps before update (default: 5)
        pipeline_fn (function, optional): State preprocessing function
    
    Returns:
        1-D Array: Episode Reward array of shape (n_env,)
    """
    n_envs = len(envs)
    all_dones = False

    states_memory = [[] for _ in range(n_envs)]
    actions_memory = [[] for _ in range(n_envs)]
    rewards_memory = [[] for _ in range(n_envs)]
    values_memory = [[] for _ in range(n_envs)]

    is_env_done = [False for _ in range(n_envs)]
    episode_rewards = [0 for _ in range(n_envs)]

    observations = []
    lives_info = []

    for id, env in enumerate(envs):
        env.reset()
        s, r, done, info = env.step(1)
        s = pipeline_fn(s)
        observations.append(s)
        # print(s)
        # print(np.sum(s, axis = 0))
        # print(len(np.sum(s, axis = 0)))
        # if "Breakout" in FLAGS.env:
        #     lives_info.append(info['ale.lives'])

    while not all_dones:

        for t in range(t_max):

            actions, values = agent.get_actions_values(observations)

            for id, env in enumerate(envs):

                if not is_env_done[id]:

                    s2, r, is_env_done[id], info = env.step(actions[id])
                    s3 = observations[id]
                    s4 = s3[0:80,10:72,0]
                    s5 = np.sum(s4, axis=1)
                    reward_shaping = max(0.05 - episo*0.0025, 0.0)
                    if 3 in s5:
                        rewards_memory[id].append(r+reward_shaping)
                    else:
                        rewards_memory[id].append(r)

                    
                    episode_rewards[id] += r

                    # if "Breakout" in FLAGS.env and info['ale.lives'] < lives_info[id]:
                    #     r = -1.0
                    #     lives_info[id] = info['ale.lives']

                    states_memory[id].append(observations[id])
                    actions_memory[id].append(actions[id])
                    # rewards_memory[id].append(r)
                    values_memory[id].append(values[id])

                    observations[id] = pipeline_fn(s2)

        future_values = agent.get_values(observations)

        for id in range(n_envs):
            if not is_env_done[id] and rewards_memory[id][-1] != -1:
                rewards_memory[id][-1] += agent.args.gamma * future_values[id]

        agent.train_iter(states_memory, actions_memory, rewards_memory, values_memory)

        states_memory = [[] for _ in range(n_envs)]
        actions_memory = [[] for _ in range(n_envs)]
        rewards_memory = [[] for _ in range(n_envs)]
        values_memory = [[] for _ in range(n_envs)]

        all_dones = np.all(is_env_done)

        

    return episode_rewards

def train_a3c(agent):
    
    pipeline_fn = partial(pipeline, new_HW=input_shape[:-1])

    envs = [deepcopy(agent.env) for i in range(agent.args.n_envs)]
    # envs[0] = gym.wrappers.Monitor(envs[0], "monitors", force=True)

    summary_writers = [tf.summary.FileWriter(logdir="{}/env-{}".format(agent.args.logdir, i)) \
        for i in range(agent.args.n_envs)]
    # agent = Agent(input_shape, output_dim)

    saver = tf.train.Saver()
    latest_checkpoint = tf.train.latest_checkpoint(agent.args.logdir)

    best_reward = -21.0
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    with tf.Session(config=config) as sess:
        # try:
        if latest_checkpoint is not None:
            saver.restore(sess, latest_checkpoint)
            agent.logger.info("Restored from {}".format(latest_checkpoint))
        else:
            init = tf.global_variables_initializer()
            sess.run(init)
            agent.logger.info("Initialized weights")

        episode = 1
        while True:
            rewards = run_episodes(envs, agent, pipeline_fn=pipeline_fn, episo=episode)

            # agent.reward_queue.append(np.mean(rewards))

            agent.logger.info('episode: %d, reward: %.4f' % (episode, np.mean(rewards)))
            agent.logger.info(rewards)



            for id, r in enumerate(rewards):
                summary = tf.Summary()
                summary.value.add(tag="Episode Reward", simple_value=r)
                summary_writers[id].add_summary(summary, global_step=episode)
                summary_writers[id].flush()

            if np.mean(rewards) > best_reward:
                best_reward = np.mean(rewards)

                print('best record', best_reward)

                saver.save(sess, "{}/model.ckpt".format(agent.args.logdir), write_meta_graph=False)
                print("Saved to {}/model.ckpt".format(agent.args.logdir))

            episode += 1

        # finally:
        #     saver.save(sess, "{}/model.ckpt".format(agent.args.logdir), write_meta_graph=False)
        #     print("Saved to {}/model.ckpt".format(agent.args.logdir))

        #     # for env in envs:
        #     #     env.close()

        #     for writer in summary_writers:
        #         writer.close()

