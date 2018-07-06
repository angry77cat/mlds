import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from agent_dir.agent import Agent
from agent_dir.model import DQN

import scipy.misc
import numpy as np
from collections import namedtuple
from tqdm import tqdm

import time
from random import sample, random, randint
from copy import deepcopy
from collections import deque, defaultdict
import matplotlib.pyplot as plt

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

############
# toast add
############
def prepro(o, image_size=[120, 120]):

    o = o[32:, 8:-8, :]
    y = 0.2126 * o[:, :, 0] + 0.7152 * o[:, :, 1] + 0.0722 * o[:, :, 2]

    y = y.astype(np.uint8)
    resized = scipy.misc.imresize(y, image_size)
    return np.expand_dims(resized.astype(np.float32),axis=2)/255.

def prepro2(o, image_size=[84, 84]):

    # import matplotlib.pyplot as plt
    # print(o.shape)
    # resized = scipy.misc.imresize(o[37:,4:80,0], image_size)/255.
    # plt.imshow(resized)
    # plt.show()
    # for i in range(80):
    #     print(scipy.misc.imresize(o[38:,5:79,:], image_size)[i,:,0])

    # a = a
    return scipy.misc.imresize(o[38:,5:79,:], image_size)/255.

############
# toast add
############
Transition = namedtuple('Transition',
                        ('s', 'a', 'next_s', 'r'))
class Replay2:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        # print(len(self.memory))
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


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

        if args.pretrain or args.test_dqn:
            print('loading trained model')
            self.Q.load_state_dict(torch.load('model/dqn_model.pth'))

        self.target = deepcopy(self.Q)
        self.target.eval()

        if USE_CUDA:
            self.Q.cuda()
            self.target.cuda()
        # load environment
        self.env = env

        # load args
        self.episode = args.episode
        self.render = args.render
        self.gamma = args.gamma
        self.max_buffer = args.max_buffer
        self.batch = 32

        self.epsilon = 0.9
        self.optimizer = optim.RMSprop(params=self.Q.parameters(), lr=args.learning_rate)
        self.criterion = nn.MSELoss()


    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        torch.manual_seed(1)

    def play(self, obs, epi):
        sample = random()
        # eps_threshold = exploration.value(t)
        stop_ratio = 0.5
        if epi/self.episode > stop_ratio:
            if sample > 0.1:
                obs = torch.from_numpy(obs).type(dtype).permute(2, 0, 1).unsqueeze(0) 
                return self.Q(Variable(obs, volatile=True)).data.max(1)[1].cpu().numpy()[0] 
            else:
                return randint(0,3)
        else:
            if sample > self.epsilon - 0.8 * (epi/(self.episode* stop_ratio)):
            # if sample > 0.1:
            # if True:
                # obs = torch.from_numpy(obs).type(dtype).permute(2, 0, 1).unsqueeze(0) / 255.0
                obs = torch.from_numpy(obs).type(dtype).permute(2, 0, 1).unsqueeze(0) 
                # print(torch.max(obs))
                # print(self.Q(Variable(obs, volatile=True)).data.max(1)[1].cpu().numpy()[0])
                return self.Q(Variable(obs, volatile=True)).data.max(1)[1].cpu().numpy()[0] 
            else:
                return randint(0,3)


    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        # replay_buffer = Replay()    # customize class
        avg_reward_hist = []
        reward_hist = []
        loss_hist = []
        Q_hist = []
        replay_buffer = Replay2(self.max_buffer) 
        epsilon = 0.9
        # state_pool, action_pool, reward_pool, state_next_pool = [], [], [], []
        Q_pool = []
        step = 0
        for episode in range(self.episode):

            state = self.env.reset()
            # state = prepro2(state)
            # fixed_noise = (np.random.random((84,84,4))-0.5) * 0.1
            # state += fixed_noise
            # print(self.play(state, 3))
            #  print(np.max(state))
            # state = torch.FloatTensor(state)
            # state = Variable(state).cuda() if USE_CUDA else Variable(state)
            # state = state.permute(2, 0, 1)
            # state_pool.append(state)
            # decay epsilon in every iteration
            # epsilon -= 0.8 * 1/self.episode
            state_pool, action_pool, next_state_pool = [], [], []
            loss_pool, Q_pool, reward_pool = [], [], []
            state_pool.append(state)

            while True:
                step += 1
                if self.render:
                    self.env.env.render()
                    time.sleep(0.01)

                # Variable(4,84,84) -> Variable(1,4,84,84)
                # state = state.unsqueeze(0)
                action = self.play(state, episode)
                # print(action)
                # if epsilon < random():
                #     action = self.Q(state)
                #     action = action.topk(1)[1].cpu().data.numpy()[0][0] + 1
                # else:
                #     action = randint(1,3)

                next_state, reward, done ,_ = self.env.step(action)
                # next_state = prepro2(next_state)
                # next_state += fixed_noise

                reward = max(-1.0, min(reward, 1.0))
                # replay_buffer.push(state, action, next_state, reward)
                action_pool.append(action)
                reward_pool.append(reward)
                next_state_pool.append(next_state)     

                # action_pool.append(action)        
                # next_state += fixed_noise
                # next_state = torch.FloatTensor(next_state)
                # next_state = Variable(next_state).cuda() if USE_CUDA else Variable(next_state)
                # next_state = next_state.permute(2, 0, 1) # Variable(4,84,84)
                # state_next_pool.append(next_state)

                # replay_buffer.push(state.squeeze(0), action, next_state, reward)

                if len(replay_buffer) >= self.max_buffer:
                # if len(replay_buffer)>128:
                    if step % 4 == 0: # replay 
                        loss, Q = self.update_model(replay_buffer)
                        loss_pool.append(loss)
                        Q_pool.append(Q)

                    if step % 2500 == 0: # update Q
                        print('update target network')
                        self.target.load_state_dict(self.Q.state_dict())
                        torch.save(self.Q.state_dict(), 'model/dqn_model.pth')

                    if done:
                        ep_reward = np.sum(reward_pool)
                        loss = np.mean(loss_pool)
                        Q = np.mean(Q_pool)
                        print('episode: {} | reward: {} | loss: {:.5f} | Q: {:.5f}'.format(episode+1, ep_reward, loss, Q))
                        # reward_pool = self.reward_prepro(reward_pool)
                        for i in range(len(state_pool)):
                            replay_buffer.push(state_pool[i], action_pool[i], next_state_pool[i], reward_pool[i])
                        
                        reward_hist.append(ep_reward)
                        loss_hist.append(loss)
                        Q_hist.append(Q)

                        if len(reward_hist)>= 30:
                            avg_reward_hist.append(np.mean(reward_hist[-30:]))
                            self.record(avg_reward_hist, 'reward')
                            self.record(loss_hist, 'loss')
                            self.record(Q_hist, 'Q')
                        break

                else:
                    if done:
                        # reward_pool = self.reward_prepro(reward_pool)
                        for i in range(len(state_pool)):
                            replay_buffer.push(state_pool[i], action_pool[i], next_state_pool[i], reward_pool[i])
                        break


                state = next_state
                state_pool.append(state)
                # if done: 
                #     # reward_pool = self.reward_prepro(reward_pool)
                #     for i in range(len(state_pool)-1):
                #         replay_buffer.push(state_pool[i], action_pool[i], state_next_pool[i], reward_pool[i])
                #     state_pool, action_pool, state_next_pool, reward_pool = [], [], [], []
                #     loss_pool = []
                #     Q_pool = []
                #     break
                # state = next_state
                # state.append(state_next_pool[-1])

    def record(self, hist, name):
        plt.plot(np.arange(len(hist)), np.array(hist))
        plt.xlabel('episode')
        plt.ylabel('%s' % name)
        plt.savefig('fig/%s.png' % name)
        plt.clf()

    def update_model(self, replay_buffer):
        transitions = replay_buffer.sample(self.batch)
        batch = Transition(*zip(*transitions))
        s = Variable(torch.from_numpy(np.asarray(batch.s)).type(dtype).permute(0,3,1,2)) #(32,4,84,84)
        a = Variable(torch.from_numpy(np.array(batch.a)).long()) #32
        r = Variable(torch.FloatTensor(batch.r)) #32
        s_next = Variable(torch.from_numpy(np.asarray(batch.next_s)).type(dtype).permute(0,3,1,2), volatile=True)


        if USE_CUDA:
            a, r = a.cuda(), r.cuda()

        current_Q = self.Q(s, self.batch).gather(1,a.unsqueeze(1)).squeeze() #(32,1)

        v_pred = self.Q(s_next, self.batch).max(1, keepdim= True)[1]
        # print(v_pred)
        next_max_Q = self.target(s_next, self.batch).gather(1, (v_pred)).detach().squeeze()
        
        next_max_Q.volatile = False

        target_Q = next_max_Q * 0.99 + r #32

        # print(current_Q.shape)
        # print(next_max_Q.shape)
        # print(target_Q.shape)
        # print(r.shape)

        # print(target_Q.shape)
        # print(current_Q.shape)
        # bellman_error = target_Q - current_Q
        # clipped_bellman_error = bellman_error.clamp(-1, 1)
        # d_error = clipped_bellman_error * -1.0


        self.optimizer.zero_grad()
        d_error = self.criterion(current_Q, target_Q)
        # print(d_error)
        # current_Q.backward(d_error.data)
        d_error.backward()
        for param in self.Q.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        # num_param_updates += 1
        # print(pred)

        # print(a)
        # print(v_pred)
        # print(v_pred)
        # action = v_pred.topk(1)[1].cpu().data.numpy()[0][0] + 1
        # print(action)

        # print(pred.gather(1,(a-1).view(-1,1))[0].cpu().data.numpy())

        '''
        v_pred = self.Q(s_next, self.batch).max(1, keepdim= True)[1]
        v = self.target(s_next, self.batch).gather(1,(v_pred)).detach()
        v.volatile = False
        '''
        # print(ans)
        # pred.gather(1,(a-1).view(-1,1)): Q value for specfic state and action
        # self.optimizer.zero_grad()
        # loss = self.criterion(pred.gather(1,(a-1).view(-1,1)).squeeze(), 0.99 * v.squeeze() + r)
        #  loss = loss.clamp(-1,1)
        # loss.backward()
        # for param in self.Q.parameters():
         	# param.grad.data.clamp_(-1,1)
        # print('loss: {}'.format(torch.mean(loss.cpu()).data.numpy()[0]))
        # print("loss: {}".format(loss.cpu().data[0]))
        # self.optimizer.step()
        return torch.mean(d_error.cpu()).data.numpy()[0], torch.mean(current_Q.cpu()).data.numpy()[0]

    def reward_prepro(self, reward_pool):
        running_add = 0.
        for i in reversed(range(len(reward_pool))):
            if reward_pool[i] == 1 or reward_pool[i] == -1:
                running_add = reward_pool[i]
            else:
                running_add = running_add * self.gamma + reward_pool[i]
                reward_pool[i] = running_add

            # Normalize reward
            # when every reward == 0, the division term will occur exception.
            # if np.sum(reward_pool) > 0.:
            #     reward_mean = np.mean(reward_pool)
            #     reward_std = np.std(reward_pool)
            #     for i in range(len(reward_pool)):
            #         reward_pool[i] = (reward_pool[i] - reward_mean) / reward_std
        return reward_pool

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
        # return self.env.get_random_action() 
        # observation = prepro2(observation)
        state = torch.FloatTensor(observation)
        state = Variable(state).cuda() if USE_CUDA else Variable(state)
        state = state.permute(2, 0, 1).unsqueeze(0)

        action = self.Q(state)
        if USE_CUDA:
            action = action.topk(1)[1].cpu().data.numpy()[0][0] 
        else:
            action = action.topk(1)[1].data.numpy()[0][0] 

        return action






