def add_arguments(parser):
    '''
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2.5e-4)
    parser.add_argument('--start_exploration', type=float, default=1.0)
    parser.add_argument('--end_exploration', type=float, default=0.05)
    parser.add_argument('--decay_exploration', type=float, default=400)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--load_path', default='')
    parser.add_argument('--save_path', default='')
    parser.add_argument('--memory_size', type=int, default=10000)
    parser.add_argument('--games', type=int, default=1001)
    parser.add_argument('--start_game', type=int, default=0)
    parser.add_argument('--keep_training', action='store_true')
    parser.add_argument('--model_id', default='')
    parser.add_argument('--learning_start', default=0, type=int)
    parser.add_argument('--update_target', default=1000, type=int)
    parser.add_argument('--online_update_freq', default=4, type=int)
    parser.add_argument('--entropy_loss_coeff', default=0.01, type=float)
    parser.add_argument('--value_loss_coeff', default=0.5, type=float)
    parser.add_argument('--n_envs', default=16, type=int)
    parser.add_argument('--logdir' , default='./logdir')


    return parser
