"""

### NOTICE ###
You DO NOT need to upload this file

"""
# import torch

import argparse
from test import test
from environment import Environment


def parse():
    parser = argparse.ArgumentParser(description="MLDS&ADL HW3")
    parser.add_argument('--env_name', default=None, help='environment name')
    parser.add_argument('--train_a2c', action='store_true', help='whether train DQN')
    parser.add_argument('--test_a2c', action='store_true', help='whether test DQN')
    parser.add_argument('--train_a3c', action='store_true', help='whether train DQN')
    parser.add_argument('--test_a3c', action='store_true', help='whether test DQN')
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
    if args.train_a2c:
        env_name = args.env_name or 'Pong-v0'
        env = Environment(env_name, args)
        from agent_dir.agent_a2c import Agent_PG
        agent = Agent_PG(env, args)
        agent.train()

    if args.test_a2c:
        env = Environment('Pong-v0', args, test=True)
        from agent_dir.agent_a2c import Agent_PG
        agent = Agent_PG(env, args)
        test(agent, env)

    if args.train_a3c:
        env_name = args.env_name or 'Pong-v0'
        env = Environment(env_name, args)
        from agent_dir.agent_a3c_tf import Agent_PG
        agent = Agent_PG(env, args)
        agent.train()

    if args.test_a3c:
        env = Environment('Pong-v0', args, test=True)
        from agent_dir.agent_a3c_tf import Agent_PG
        agent = Agent_PG(env, args)
        test(agent, env)



if __name__ == '__main__':
    args = parse()
    run(args)
