import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt 
import sys
import re
import numpy as np
from scipy.interpolate import spline

def plot_reward2game(file_names, names, name):

    handles_list = []
    for i, file_name in enumerate(file_names):

        game = []
        reward = []
        with open(file_name, 'r') as f:
            lines = f.readlines()

            for line in lines:

                words = re.split(', | |: ', line.rstrip('\n'))
                # print(words)
                if words[0] == 'episode' or words[0] == 'game':
                    # game.append(int(words[1]))
                    # reward.append(float(words[7]))
                    if names[i] == 'a3c':
                        game.append(int(words[1]))
                        reward.append(float(words[3]))
                    else:
                        game.append(int(words[1]))
                        reward.append(float(words[5]))
        game = game[:270]
        reward = reward[:270]
        # game = np.asarray(game)
        # reward = np.asarray(reward)
        # print(reward)
        # game_new = np.linspace(game.min(),game.max(),60) #300 represents number of points to make between T.min and T.max
        # print('123')
        # reward_new = spline(game, reward, game_new)
        # print('256')
        # new_reward = []
        # for j in range(len(reward)):
        #     start = j - 5
        #     end = j + 5
        #     if start < 0:
        #         start = 0
        #     new_reward.append(np.mean(reward[start: end]))

        # reward = new_reward
        a, = plt.plot(game, reward, label='%s' % names[i])
        handles_list.append(a)


    plt.xlabel('episodes')
    plt.ylabel('average 30 episodes reward')
    plt.legend(handles=handles_list)
    plt.savefig('img/%s_avg_reward.png' % name, dpi=150)


if __name__ == "__main__":
    # file_name_list = [\
    # 'ADLxMLDS2017_hw3_model/dqn_henorm_replay10000_target50_exp2000_lstart1000_freq4/train_log.txt', \
    # 'ADLxMLDS2017_hw3_model/dqn_db_henorm_replay10000_target50_exp2000_lstart1000_freq4/train_log.txt', \
    # # 'ADLxMLDS2017_hw3_model/dqn_duel_henorm_replay10000_target50_exp2000_lstart1000_freq4/train_log.txt', \
    # ]
    # name_list = [\
    # 'base', \
    # 'double', \
    # # 'duel', \
    # ]     
    # plot_reward2game(file_name_list, name_list, 'dqn_db')

    # file_name_list = [\
    # 'ADLxMLDS2017_hw3_model/dqn_henorm_replay10000_target50_exp2000_lstart1000_freq4/train_log.txt', \
    # 'ADLxMLDS2017_hw3_model/dqn_henorm_replay30000_target50_exp2000_lstart1000_freq4/train_log.txt', \
    # 'ADLxMLDS2017_hw3_model/dqn_henorm_replay100000_target50_exp2000_lstart1000_freq4/train_log.txt', \
    # 'ADLxMLDS2017_hw3_model/dqn_henorm_replay300000_target50_exp2000_lstart1000_freq4/train_log.txt',
    # ]
    # name_list = [\
    # 'replay10000', \
    # 'replay30000', \
    # 'replay100000', \
    # 'replay300000',
    # ]
    # plot_reward2game(file_name_list, name_list, 'replaysize')
    # file_name_list = [
    # 'ADLxMLDS2017_hw3_model/dqn_henorm_replay10000_target50_exp2000_lstart1000_freq4/train_log.txt',
    # ]
    # name_list = [
    # 'dqn',
    # ]

    # plot_reward2game(file_name_list, name_list, 'dqn')

    file_name_list = [
    'ADLxMLDS2017_hw3_model/pg_batch1_rms1e-4/train_log.txt',
    'logdir/train_log.txt',
    ]
    name_list = [
    'pg',
    'a3c',
    ]

    plot_reward2game(file_name_list, name_list, 'pg_vs_a3c')