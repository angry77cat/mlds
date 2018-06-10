def add_arguments(parser):
    '''
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    
    parser.add_argument('-b', '--batch_size', type=int, default=5, help='batch size for training')
    parser.add_argument('-e', '--episode', type=int, default=6000, help='episode')
    parser.add_argument('-g', '--gamma', type=float, default=0.99, help='gamma')
    parser.add_argument('-l', '--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('-p', '--pretrain', action='store_true', help='load pretrain model')

    return parser
