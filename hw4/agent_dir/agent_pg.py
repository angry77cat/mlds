from agent_dir.agent import Agent
from agent_dir.model import Net

import scipy.misc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Bernoulli, Categorical

import time
import random
from itertools import count

torch.manual_seed(2222)
USE_CUDA = torch.cuda.is_available()
# USE_CUDA = False

def prepro(o,image_size=[80,80]):
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
    #############
    # toast add #
    #############
    y = 0.2126 * o[:, :, 0] + 0.7152 * o[:, :, 1] + 0.0722 * o[:, :, 2]

    #############
    # toast end #
    #############
    y = y.astype(np.uint8)
    resized = scipy.misc.imresize(y, image_size)
    return np.expand_dims(resized.astype(np.float32),axis=2)/255.


class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)

        ##################
        # YOUR CODE HERE #
        ##################

        # environment
        self.env = env.env

        # args
        self.gamma = args.gamma
        self.episode = args.episode
        self.batch_size = args.batch_size

        # initialize model
        self.model = Net()
        if USE_CUDA:
            self.model = self.model.cuda()

        # optimizer
        self.optimizer = optim.RMSprop(params=self.model.parameters(), lr=args.learning_rate)

        if args.test_pg:
            #you can load your model here
            print('loading trained model')
            self.model.load_state_dict(torch.load('model/pg_model.pth'))
            self.model.eval()


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
        
        

        self.model.train()

        for episode in range(self.episode):
            print('episode: ', episode)
            # setup
            state = self.env.reset()
            state = prepro(state)
            state = torch.FloatTensor(state)
            state = Variable(state).cuda() if USE_CUDA else Variable(state)
            
            # memory
            state_pool = []
            action_pool = []
            reward_pool = []
            
            # start playing
            while True:
                self.env.render()

                prob = self.model(state)
                m = Categorical(prob)
                
                # memorize!
                state_pool.append(state)

                # take action
                # 1-> do nothing, 2->up, 3->down
                # action = int(prob.topk(1)[1]) + 1
                print(prob) 
                action = m.sample() + 1
                action_pool.append(action)

                if USE_CUDA:
                    action = action.cpu().data.numpy().astype(int)[0]
                else:
                    action = action.data.numpy().astype(int)[0]

                # action = random.sample([1, 2, 3], 1)

                state, reward, done, info = self.env.step(action)
                state = prepro(state)
                state = torch.FloatTensor(state)
                state = Variable(state).cuda() if USE_CUDA else Variable(state)

                # memorize!
                reward_pool.append(reward)

                # if terminated
                if done:
                    break

            if episode > 0 and episode % self.batch_size == 0:
                print('update policy..')
                # update network (policy)
                # Discount reward
                running_add = 0
                for i in reversed(range(len(state_pool))):
                    if reward_pool[i] == 0:
                        running_add = 0
                    else:
                        running_add = running_add * self.gamma + reward_pool[i]
                        reward_pool[i] = running_add

                # Normalize reward
                reward_mean = np.mean(reward_pool)
                reward_std = np.std(reward_pool)
                for i in range(len(state_pool)):
                    reward_pool[i] = (reward_pool[i] - reward_mean) / reward_std

                # optimize!
                self.optimizer.zero_grad()
                loss = 0
                for i in range(len(state_pool)):
                    state = state_pool[i]
                    action = action_pool[i]
                    action = int(action)

                    # action = Variable(torch.FloatTensor([float(action_pool[i])]))
                    # action = action.cuda() if USE_CUDA else action

                    reward = reward_pool[i]

                    prob = self.model(state)
                    # m = Bernoulli(prob[action_pool[i] - 1])
                    # m = Categorical(prob)
                    # loss += -m.log_prob(action) * reward
                    loss += -torch.log(prob[action-1]) * (reward - 0.5) # substract baseline (arbitrary)
                loss /= len(state_pool)
                loss.backward()
                self.optimizer.step()

                # clear memory!
                state_pool = []
                action_pool = []
                reward_pool = []

                torch.save(self.model.state_dict, 'model/pg_model.pth')








































        


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
        action = self.model(observation)
        action = action.topk(1)
        
        return action
        # return self.env.get_random_action()


