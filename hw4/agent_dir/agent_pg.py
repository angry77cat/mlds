from agent_dir.agent import Agent
from agent_dir.model import Net, CNN2

import scipy.misc
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Bernoulli, Categorical

import time
import random
from itertools import count

torch.manual_seed(22)
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
    o = o[35:190, :, :]     # remove the score board
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
        self.render = args.render
        self.save_state = None
        # initialize model
        self.model = CNN2()
        if USE_CUDA:
            self.model = self.model.cuda()

        # optimizer
        self.optimizer = optim.RMSprop(params=self.model.parameters(), lr=args.learning_rate)

        if args.test_pg or args.pretrain:
            #you can load your model here
            print('loading trained model -- cnn2')
            self.model.load_state_dict(torch.load('model/pg_model.pth'))
            # self.model.eval()


    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        torch.manual_seed(1)


    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        
        

        self.model.train()

        # memory
        state_pool = []
        action_pool = []
        reward_pool = []
        prob_pool = []
        log_prob_pool = []
        # for observation
        # frame_pool = []

        for episode in range(self.episode):
            # print('episode: ', episode)
            # setup
            state = self.env.reset()
            state = prepro(state)
            state = torch.FloatTensor(state)
            state = Variable(state).cuda() if USE_CUDA else Variable(state)

            new_state = state
            save_state = state
            
            # start playing
            while True:
                if self.render:
                    self.env.render()

                dif_state = new_state - save_state
                # plt.imshow(dif_state.cpu().data.numpy()[:, :, 0], cmap='gray')
                # plt.colorbar()
                # plt.show()
                save_state = new_state
                prob = self.model(dif_state)
                # if len(frame_pool) <= 150:
                #     frame_pool.append(dif_state.cpu().data.numpy())
                prob_pool.append(prob.cpu().data.numpy())
                m = Categorical(prob)
                
                # memorize!
                state_pool.append(dif_state)

                # take action
                # 1-> do nothing, 2->up, 3->down
                # action = int(prob.topk(1)[1]) + 1
                
                # print('episode: ', episode)
                # print(prob) 
                action = m.sample() + 1

                ######
                # test 
                ######
                log_prob_pool.append(m.log_prob(action-1))
                ######
                # test
                ######
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
                new_state = state
                # memorize!
                reward_pool.append(reward)


                # if terminated
                if done:
                    break

            if episode > 0 and episode % self.batch_size == 0:
                # print('update policy..')
                # update network (policy)
                # Discount reward
                running_add = 0.
                # print(reward_pool[-5:])

                avg_reward = np.sum(reward_pool)/self.batch_size
                print("episode: %d | average reward: %.2f" % (episode, avg_reward))
                with open('reward_log_cnn2.csv', 'a+') as log:
                    log.write('%d,%f\n' % (episode, avg_reward))

                for i in reversed(range(len(state_pool))):
                    if reward_pool[i] == 1 or reward_pool[i] == -1:
                        running_add = reward_pool[i]
                    else:
                        running_add = running_add * self.gamma + reward_pool[i]
                        reward_pool[i] = running_add
                        # print(running_add)

                # Normalize reward
                # reward_mean = np.mean(reward_pool)
                # reward_std = np.std(reward_pool)


                # for i in range(len(state_pool)):
                #     reward_pool[i] = (reward_pool[i] - reward_mean) / reward_std


                # print(reward_pool[-5:])
                # plt.plot(np.arange(len(reward_pool)), reward_pool)
                # plt.xlabel('frame')
                # plt.ylabel('reward')
                # plt.title('moving average reward')
                # plt.show()
                # assert False
                # optimize!
                self.optimizer.zero_grad()
                # loss = 0
                for i in range(len(state_pool)):
                    state = state_pool[i]
                    # action = action_pool[i]
                    # action = int(action)

                    # action = Variable(torch.FloatTensor([action_pool[i]]))
                    # action = action.cuda() if USE_CUDA else action

                    reward = reward_pool[i]

                    # prob = self.model(state)
                    # m = Bernoulli(prob[action_pool[i] - 1])
                    # m = Categorical(prob)
                    # loss += -m.log_prob(action) * reward
                    # loss += -torch.log(prob[action-1]) * (reward - reward_mean) # substract baseline (arbitrary)
                    # loss = -torch.log(prob[action-1]) * reward # substract baseline (arbitrary)
                    
                    loss = -log_prob_pool[i] * reward
                    loss.backward()

                # loss /= len(state_pool)
                # loss.backward()
                self.optimizer.step()

                ###############
                # observe the std of probability of each action
                # (does the network really react to different frame?)
                #
                # print("the std of each action over an episode:")
                # print(np.asarray(prob_pool).std(0))
                # plt.plot(np.arange(len(prob_pool)), np.asarray(prob_pool).T[0], label='none')
                # plt.plot(np.arange(len(prob_pool)), np.asarray(prob_pool).T[1], label='up')
                # plt.plot(np.arange(len(prob_pool)), np.asarray(prob_pool).T[2], label='down')
                plt.plot(np.arange(150), np.asarray(prob_pool[:150]).T[0], label='none')
                plt.plot(np.arange(150), np.asarray(prob_pool[:150]).T[1], label='up')
                plt.plot(np.arange(150), np.asarray(prob_pool[:150]).T[2], label='down')
                plt.xlabel('t')
                plt.ylabel('probability')
                plt.title('std of none: %.4f, up: %.4f, down: %.4f' % (np.asarray(prob_pool).std(0)[0], np.asarray(prob_pool).std(0)[1], np.asarray(prob_pool).std(0)[2]))
                plt.legend()
                plt.savefig('plot/std_cnn2.png')
                plt.close()

                ###########
                # observe
                ###########
                # for idx, frame in enumerate(frame_pool):
                #     plt.imshow(frame[:, :, 0])
                #     plt.savefig('plot/frames/frame%d.png' % idx)
                #     plt.close()

                # clear memory!
                state_pool = []
                action_pool = []
                reward_pool = []
                prob_pool = []
                log_prob_pool = []
                # frame_pool = []
                # assert False

                torch.save(self.model.state_dict(), 'model/pg_model_cnn2.pth')

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
        observation = prepro(observation)
        observation = torch.FloatTensor(observation)
        observation = Variable(observation).cuda() if USE_CUDA else Variable(observation)

        if self.save_state is None:
            self.save_state = observation
        dif_state = observation - self.save_state
        self.save_state = observation

        prob = self.model(dif_state)
        # action = int(action.topk(1)[1]) + 1
        m = Categorical(prob)

        
        action =  m.sample() + 1
        if USE_CUDA:
            action = action.cpu().data.numpy().astype(int)[0]
        else:
            action = action.data.numpy().astype(int)[0]

        return action
        # return self.env.get_random_action()


