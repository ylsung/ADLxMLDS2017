"""

### NOTICE ###
You DO NOT need to upload this file

"""
import torch
import argparse
from test import test
from environment import Environment


def parse():
    parser = argparse.ArgumentParser(description="MLDS&ADL HW3")
    parser.add_argument('--env_name', default=None, help='environment name')
    parser.add_argument('--train_dqn_pm', action='store_true', help='whether test DQN')
    parser.add_argument('--train_dqn_duel', action='store_true', help='whether test DQN')
    parser.add_argument('--train_dqn_db', action='store_true', help='whether test DQN')
    parser.add_argument('--test_dqn_pm', action='store_true', help='whether test DQN')
    parser.add_argument('--test_dqn_duel', action='store_true', help='whether test DQN')
    parser.add_argument('--test_dqn_db', action='store_true', help='whether test DQN')
    parser.add_argument('--video_dir', default=None, help='output video directory')
    parser.add_argument('--do_render', action='store_true', help='whether render environment')
    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args


def run(args):
    if args.train_dqn_pm:
        env_name = args.env_name or 'BreakoutNoFrameskip-v4'
        env = Environment(env_name, args, atari_wrapper=True)
        from agent_dir.agent_dqn_pm import Agent_DQN
        agent = Agent_DQN(env, args)
        agent.train()

    if args.test_dqn_pm:
        env = Environment('BreakoutNoFrameskip-v4', args, atari_wrapper=True, test=True)
        from agent_dir.agent_dqn_pm import Agent_DQN
        agent = Agent_DQN(env, args)
        test(agent, env, total_episodes=100)

    if args.train_dqn_duel:
        env_name = args.env_name or 'BreakoutNoFrameskip-v4'
        env = Environment(env_name, args, atari_wrapper=True)
        from agent_dir.agent_dqn_duel import Agent_DQN
        agent = Agent_DQN(env, args)
        agent.train()

    if args.test_dqn_duel:
        env = Environment('BreakoutNoFrameskip-v4', args, atari_wrapper=True, test=True)
        from agent_dir.agent_dqn_duel import Agent_DQN
        agent = Agent_DQN(env, args)
        test(agent, env, total_episodes=100)

    if args.train_dqn_db:
        env_name = args.env_name or 'BreakoutNoFrameskip-v4'
        env = Environment(env_name, args, atari_wrapper=True)
        from agent_dir.agent_dqn_db import Agent_DQN
        agent = Agent_DQN(env, args)
        agent.train()

    if args.test_dqn_db:
        env = Environment('BreakoutNoFrameskip-v4', args, atari_wrapper=True, test=True)
        from agent_dir.agent_dqn_db import Agent_DQN
        agent = Agent_DQN(env, args)
        test(agent, env, total_episodes=100)



if __name__ == '__main__':
    args = parse()
    run(args)
