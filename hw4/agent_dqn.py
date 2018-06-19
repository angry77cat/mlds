from agent_dir.agent import Agent
from agent_dir.model import DQN

import scipy.misc

from random import sample
from collections import namedtuple, deque

USE_CUDA = torch.cuda.is_available()


def prepro(o, image_size=[120, 120]):

    o = o[32:, 8:-8, :]
    y = 0.2126 * o[:, :, 0] + 0.7152 * o[:, :, 1] + 0.0722 * o[:, :, 2]

    y = y.astype(np.uint8)
    resized = scipy.misc.imresize(y, image_size)
    return np.expand_dims(resized.astype(np.float32),axis=2)/255.

class Replay:
    def __init__(self):
        pass

    def sample(self, batch):
        pass

    def push(self):
        pass

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_DQN,self).__init__(env)

        ##################
        # YOUR CODE HERE #
        ##################

        # initialize model
        self.Q = DQN()
        self.target = DQN()
        self.target.eval()

        # load environment
        self.env = env.env

        # load args
        self.episode = args.episode
        self.render = args.render
        self.gamma = args.gamma
        self.max_buffer = args.max_buffer
        self.batch = 32

        if args.pretrain or args.test_dqn:
            print('loading trained model')
            self.model.load_state_dict(torch.load('model/dqn_model.pth'))

        self.optimizer = optim.RMSprop(params=self.model.parameters(), lr=args.learning_rate)
        self.criterion = nn.MSELoss()

    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        self.env.reset()


    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        replay_buffer = deque()

        for episode in range(self.episode):
            state = self.env.reset()
            state = torch.FloatTensor(state)
            state = Variable(state).cuda() if USE_CUDA else Variable(state)

            while True:
                if self.render:
                    self.env.render()

                action = self.Q(state)
                #
                #
                #
                next_state, reward, done ,info = self.env.step(action)

                # create memory
                m = [state, action, next_state, reward]
                # store it in the replay_buffer
                replay_buffer.append(m)
                if len(replay_buffer) >= self.max_buffer:
                    replay_buffer.popleft()
                if done:
                    break

            # update parameters
            if episode > 0:
                # target network
                self.target.load_state_dict(self.Q.state_dict)

                # regression here
                for i in range(10):
                    # sample a batch from buffer
                    a_batch = Variable(torch.FloatTensor(sample(replay_buffer, self.batch)))
                    if USE_CUDA:
                        a_batch = a_batch.cuda()

                    a_batch_update = torch.cat([a_batch.s, a_batch.a])
                    a_batch_fixed = torch.cat([a_batch.s_, a_batch.a])

                    self.optimizer.zero_grad()
                    pred = self.Q(a_batch_update)
                    ans = target(a_batch_fixed)
                    loss = self.criterion(pred, ans)
                    loss.backward()
                    self.optimizer.step()






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
        return self.env.get_random_action()

