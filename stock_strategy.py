import numpy as np
import pandas as pd
from knapsack import knapsack_unbounded_dp
from itertools import product
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import os
from os import listdir
from os.path import isfile, join
import sys
from sklearn.preprocessing import StandardScaler

class Environment():

    def __init__(self, data_file, budget, test_share=0.3):
        self.budget = budget
        self.data = pd.read_csv(data_file,header=0).values
        self.train_start_idx = 0
        self.num_stocks = self.data.shape[1]
        self.train_end_idx = round(len(self.data)*(1-test_share))
        self.test_start_idx = self.train_end_idx + 1
        self.test_end_idx = len(self.data) - 1
        self.scaler = StandardScaler()
        self.scale_data()
    
    def scale_data(self):
        train_data = self.data[self.train_start_idx:self.test_start_idx, :].copy()
        test_data = self.data[self.test_start_idx:self.test_end_idx+1, :].copy()
        scaled_train_data = self.scaler.fit_transform(train_data)
        scaled_test_data = self.scaler.transform(test_data)
        self.scaled_data =  np.concatenate((scaled_train_data, scaled_test_data), axis=0)


    
    def initialize_curr_state(self,idx):
        self.pointer = idx
        s = self.data[idx,:].tolist()
        s_scaled = self.scaled_data[idx, :].tolist()
        w = [0] * self.num_stocks
        cash = self.budget
        w.append(cash)
        self.curr_state = np.array(s + w)
        self.curr_scaled_state = np.array(s_scaled + w)   
    
    
    def get_new_state_scaled_state(self,a,s,prev_state):
        
        assert all(el in ['S','B','H'] for el in a)
        cash = prev_state[-1]        
        w = prev_state[-self.num_stocks-1:-1] 
        new_w = w.copy()
        #first perfoem selling
        if "S" in a:
            sell_indexes = np.where(np.array(a) == 'S')[0]
            cash += np.dot(np.array(w[sell_indexes]), np.array(s[sell_indexes]))
            new_w[sell_indexes] = 0
        if 'B' in a:
            buy_indexes =  np.where(np.array(a) == 'B')[0]
            can_buy = np.array([True]*len(buy_indexes))
            while np.any(can_buy):
                for i,idx in enumerate(buy_indexes):
                    if s[idx] <= cash:
                        new_w[idx] += 1
                        cash -= s[idx]
                    else:
                        can_buy[i] = False
        new_w_list = new_w.tolist()
        new_w_list.append(cash)
        new_state = np.array( s.tolist() + new_w_list)
        new_scaled_state = self.get_scaled_state(new_state)
        return new_state, new_scaled_state

            
    def get_value_of_state(self, state):
        s = state[:self.num_stocks]
        w = state[self.num_stocks: 2*self.num_stocks]
        cash = state[-1]
        return np.dot(w,s) + cash
    
    def get_return(self, portfolio_value):
        return (portfolio_value - self.budget) / self.budget

    def move(self,a):        
        prev_state = self.curr_state.copy()
        #get new state
        self.pointer += 1
        s = self.data[self.pointer,:]
        self.curr_state, self.curr_scaled_state = self.get_new_state_scaled_state(a,s,prev_state = prev_state)
        #get reward from new state
        r = (self.get_value_of_state(self.curr_state) - self.get_value_of_state(prev_state)) / self.get_value_of_state(prev_state)
        return r
    
    
    def initialize_possible_actions(self):
        self.possible_actions = list(product('BSH', repeat=self.num_stocks))
    
    def check_if_curr_state_terminal(self,mode):
        if mode == 'train':
            return self.pointer >= self.train_end_idx
        elif mode == 'test':
            return self.pointer >= self.test_end_idx
        else:
            raise Exception('Wrong mode')
    
    def get_sigmoid(self,x):
        return  1 / (1 + np.exp(-x))
    
    def get_centered_target(self,x):
        return np.tanh(x - self.budget)

    def get_scaled_state(self, state):
        s = state[:self.num_stocks].reshape(1,-1)
        w = state[self.num_stocks:2*self.num_stocks]
        scaled_w = self.get_sigmoid(w).tolist()        
        cash = state[-1]
        scaled_cash = self.get_sigmoid(cash)
        scaled_w.append(scaled_cash)
        s_scaled = self.scaler.transform(s)
        s_scaled = s_scaled.reshape((-1,))
        s_scaled = s_scaled.tolist()
        scaled_state = np.array(s_scaled + scaled_w)
        return scaled_state

    
    


class Agent():
    def __init__(self,data_file, budget, lr, gamma = 0.9, test_share=0.3, step=0, restore=True):
        self.env = Environment(data_file, budget, test_share)
        self.gamma = gamma
        self.step = step
        self.restore = restore
        self.lr= lr
    
    
    def initalize_model(self):
        self.Q_model = Model(state_length = len(self.env.curr_state), action_length = len(self.env.possible_actions), step=self.step,lr=self.lr)
        self.Q_model.build_model()
        self.Q_model.initialize_variables_and_sess()
        self.Q_model.initialize_saver()

    def pick_action_eps_greedy(self,eps,state,mode='train'):
        u = np.random.uniform()        
        if u < eps:
            idx = random.randint(0,len(self.env.possible_actions)-1)
            action = self.env.possible_actions[idx]
        else:
            scaled_state = self.env.get_scaled_state(state) 
            #print('scaled_state', scaled_state)           
            predictions = self.Q_model.predict(scaled_state)[0]
            #predictions = self.Q_model.predict(state)
            #print('predictions shape =',predictions.shape)            
            idx = np.argmax(predictions)
            if mode=='train':
                #print('idx = ',idx)
                #print('Q model in state = ',np.max(predictions))
                pass
            if np.isnan(predictions[idx]):
                raise Exception('nan produced')
            action = self.env.possible_actions[idx]
        return action

    
    def create_full_target(self,a,target,s):
        s_scaled = self.env.get_scaled_state(s)        
        predictions = self.Q_model.predict(s_scaled)[0]                
        full_target = predictions.copy()
        idx = self.env.possible_actions.index(a)        
        full_target[idx] = target
        return full_target


    def restore_model(self):
        self.Q_model = Model(state_length = len(self.env.curr_state), action_length = len(self.env.possible_actions), step=self.step,lr=self.lr)
        self.Q_model.build_model()
        self.Q_model.restore__latest_session()          


    def train_using_q_learning(self,N,alpha0,eps=0.5):
        #init current state
        self.env.initialize_curr_state(self.env.train_start_idx)
        #initilaze possible actions
        self.env.initialize_possible_actions()        
                
        #initalize Q model and session
        if self.restore:
            self.restore_model()
        else:
            self.initalize_model()
              
        #initialize count_s_a
        self.count_s_a = {}        
        #self.actions_taken = []
        self.rewards = []
        self.port_values_train = []
        cd = False
        for i in range(N):
            if i == (N - 1):
                cd = True
            #play episode using q-learning
            self.play_episode_q_learning(eps=eps,alpha0=alpha0,collect_data = cd)
            self.Q_model.step += 1
            print('i = ',i)
        
        self.Q_model.save_model()
        #print(self.actions_taken)
        fig = plt.figure()
        plt.hist(np.array(self.rewards))
        plt.title('historgram of training rewards')
        plt.savefig('hist_rewards.png')

        fig1 = plt.figure()
        plt.plot(np.array(self.port_values_train))
        plt.title('Portfolio values during training')
        plt.savefig('portfolio_value_train.png')

    
    def play_episode_q_learning(self,eps,alpha0,collect_data=False):
        
        #initialize state
        self.env.initialize_curr_state(self.env.train_start_idx)
        s = self.env.curr_state

        while True:
            #pick a random action a from state s
            #print('start to pick new action')
            a = self.pick_action_eps_greedy(eps=eps,state=s)
            #print('end pick new action')
            #if a not in self.actions_taken:
            #    self.actions_taken.append(a)
            
            #get next state s_prime based on s,a and get reward
            #print('start move')
            r = self.env.move(a)
            #print('end move')
            
            #print('at point {} of {}'.format(self.env.pointer, self.env.train_end_idx))
            if collect_data:
                self.rewards.append(r)
                port_value = self.env.get_value_of_state(s)
                self.port_values_train.append(port_value)

            #update count_s_a
            if (tuple(s.tolist()),a) in self.count_s_a:
                self.count_s_a[(tuple(s.tolist()),a)] += 0.005
            else:
                self.count_s_a[(tuple(s.tolist()),a)] = 1
            
            s_prime = self.env.curr_state

            #if s_prime is terminal:
            if self.env.check_if_curr_state_terminal(mode = 'train'):
                target = r
            else:
                #print('start predictions for update')
                s_prime_scaled = self.env.get_scaled_state(s_prime)                
                predictions = self.Q_model.predict(s_prime_scaled)[0]
                #print('end predictions for update')
                q_s_prime_max = np.max(predictions)
                target = r + self.gamma * q_s_prime_max
            
            #create full target
            #print('start full target')
            full_target = self.create_full_target(a,target,s)
            #print('end full target')
            

            #update Q model based on target and state s
            s_scaled = self.env.get_scaled_state(s)            
            self.Q_model.update_model(scaled_state=s_scaled,target=full_target)

            if self.env.check_if_curr_state_terminal(mode = 'train'):
                break

            s = s_prime

    def test_policy(self):
        self.env.initialize_curr_state(idx=self.env.test_start_idx)
        portfolio_values = []
        returns = []
        real_values_list = []
        actions_taken_test = []
        while True:
            #get the value of portfolio
            port_value = self.env.get_value_of_state(self.env.curr_state)
            real_values = self.env.data[self.env.pointer, :].tolist()
            real_values_list.append(real_values)
            #check if current state terminal
            if self.env.check_if_curr_state_terminal(mode='test'):                
                break
            portfolio_values.append(port_value)
            returns.append(self.env.get_return(port_value))

            #pick greedy action a
            a = self.pick_action_eps_greedy(state = self.env.curr_state,eps=0, mode='test')
            if a not in actions_taken_test:
                actions_taken_test.append(a)
            #print('a = ',a)

            #move using action a
            self.env.move(a)

        print(actions_taken_test)
        portfolio_values = np.array(portfolio_values)
        returns = np.array(returns)
        real_values_np = np.array(real_values_list)

        fig1 = plt.figure()
        plt.plot(portfolio_values)
        plt.title("Portfolio values")
        plt.savefig('portfolio_value.png')

        fig2 = plt.figure()
        plt.plot(returns)
        plt.title('Returns')
        plt.savefig('portfolio_returns.png')

        fig3 = plt.figure()
        plt.plot(real_values_np[:,0],'r', real_values_np[:,1], 'b', real_values_np[:,2], 'g')
        plt.title('real prices')
        plt.savefig('real_prices.png')
        

class Model():
    def __init__(self, state_length, action_length,step, lr = 0.00001):        
        self.input_length = state_length
        self.output_length = action_length
        self.lr = lr
        self.step = step
    
    def build_model(self):        
        Input = tf.placeholder(tf.float32, [None, self.input_length], name = 'input')
        Label = tf.placeholder(tf.float32, [None, self.output_length], name='label')

        n1 = 50
        W1 = tf.Variable(tf.random_normal([self.input_length, n1], stddev=0.03), name='W1')
        b1 = tf.Variable(tf.random_normal([n1]), name='b1')
        
        W2 = tf.Variable(tf.random_normal([n1, self.output_length], stddev=0.03), name='W2')
        b2 = tf.Variable(tf.random_normal([self.output_length]), name='b1')

        net = tf.add(tf.matmul(Input, W1), b1)
        net = tf.nn.relu(net)
        output = tf.add(tf.matmul(net, W2), b2)

        loss = tf.reduce_mean(tf.squared_difference(output, Label))
        init_op = tf.global_variables_initializer()

        optimiser = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(loss)

        self.Input = Input
        self.Label = Label
        self.output = output
        self.init_op = init_op
        self.loss = loss
        self.optimiser = optimiser

    def initialize_variables_and_sess(self):
        self.sess = tf.Session()        
        self.sess.run(self.init_op)
    
    
    
    def update_model(self,scaled_state,target):
        target = np.reshape(target, (1,-1))        
        scaled_state = np.reshape(scaled_state, (1,-1)) 
        if np.any(np.isnan(scaled_state)):
            print(scaled_state)
        #print('start update')     
        _ = self.sess.run(self.optimiser, 
                      feed_dict={self.Label: target, self.Input: scaled_state})
        #print('end update')
        
    
    def predict(self,scaled_state):
        scaled_state = np.reshape(scaled_state, (1,-1))        
        output = self.sess.run(self.output, feed_dict={self.Input:scaled_state})
        return output
    
    def close_session(self):
        self.sess.close()

    def initialize_saver(self):
        self.saver = tf.train.Saver()

    
    def maybe_make_ckpt_dir(self,directory='./checkpoint'):
        if not os.path.isdir(directory):
            os.mkdir(directory)

    
    def save_model(self,directory='./checkpoint'):
        self.maybe_make_ckpt_dir(directory=directory)
        filename = directory + '/' + 'stock_stratefy_model'
        self.saver.save(self.sess, filename, global_step= self.step)
    
    def get_latest_checkpoint(self,directory='./checkpoint'):
        mypath = directory
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        only_meta = [f for f in onlyfiles if f[-4:] == 'meta']
        maxi = -1
        #print(only_meta)
        for f in only_meta:
            print(f[:-5])
            print(f[:-5].split('-'))
            num = int(f[:-5].split('-')[1])
            if num > maxi:
                maxi = num
                filename = f
        
        self.step = maxi + 1
        self.latest_metafile = filename

    def restore__latest_session(self, directory='./checkpoint'):
        self.sess = tf.Session()
        self.get_latest_checkpoint(directory=directory)
        print('Restoring from model '+self.latest_metafile)
        filename = directory + '/' + self.latest_metafile
        self.saver = tf.train.import_meta_graph(filename)
        ckpt =  tf.train.latest_checkpoint(directory)
        self.saver.restore(self.sess,ckpt)
        self.sess.run(self.init_op)
        

def get_bool(string):
    if string == 'False':
        return False
    elif string == 'True':
        return True
    else:
        raise Exception('Wrong argument')

if __name__ == '__main__':
    restore_str = sys.argv[1]
    restore = get_bool(restore_str)    
    agent = Agent(data_file='stock_data_adjusted.csv', budget = 100000,lr=0.001, gamma = 0.9, test_share=0.3, step=0, restore=restore)    
    agent.train_using_q_learning(N=50,alpha0=1,eps=0.5)
    agent.test_policy()
    agent.Q_model.close_session()

    
    









    

        










