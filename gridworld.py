import pandas as pd
import numpy as np
import random

class Grid():
    def __init__(self, length, height, gamma, wall_blocks_coords, pos_reward_coords, neg_reward_coords):
        self.length = length
        self.height = height
        self.gamma = gamma
        self.wall_blocks_coords = wall_blocks_coords #must be list of tuples
        self.pos_reward_coords = pos_reward_coords #must be list of tuples
        self.neg_reward_coords = neg_reward_coords #must be list of tuples
        self.pos_reward = 1
        self.neg_reward = -1
        self.build_board()
        self.define_states_and_actions_objects()

    def build_board(self):
        self.board = np.zeros((self.height, self.length))
        for i in range(len(self.wall_blocks_coords)):
            coord = self.wall_blocks_coords[i]
            self.board[coord] = np.nan
        
    

    def check_if_state_is_legal(self,coord):
        if coord[0] < 0 or coord[0] >= self.height or coord[1] < 0 or coord[1] >= self.length:
            return False
        elif coord in self.wall_blocks_coords:
            return False
        else:
            return True

    def get_possible_a_s_prime_from_coord(self,coord):
        action_list = ['R','D','L','U']
        change_list = [[0,1],[1,0],[0,-1],[-1,0]]
        action_list_for_state = []
        state_list_for_state = []
        for i  in range(len(change_list)):
            change = change_list[i]
            action = action_list[i]
            new_coord = (coord[0] + change[0], coord[1] + change[1])
            if self.check_if_state_is_legal(new_coord):
                action_list_for_state.append(action)
                state_list_for_state.append(new_coord)

        if coord in self.pos_reward_coords:
            raise Exception('You can not move from reward state')
        elif coord in self.neg_reward_coords:
            raise Exception('You can not move from reward state')
        return action_list_for_state , state_list_for_state

            

    def define_states_and_actions_objects(self):
        self.s_and_s_prime_dict = {} # {s1:[s_prime1,s_prime1,...],s2:[s_prime1,s_prime2],...}
        self.s_and_a_dict = {} #{s1: [a1,a2,..], s2:[a1,a2,..],...}
        for i in range(self.height):
            for j in range(self.length):
                curr_state = self.board[i,j]
                if np.isnan(curr_state):
                    continue
                elif (i,j) in self.pos_reward_coords:                    
                    continue
                elif (i,j) in self.neg_reward_coords:                    
                    continue
                action_list, state_list = self.get_possible_a_s_prime_from_coord((i,j))
                self.s_and_s_prime_dict[(i,j)] = state_list
                self.s_and_a_dict[(i,j)] = action_list

    def check_if_terminal(self,coord):
        if coord in self.pos_reward_coords:
            return True
        elif coord in self.neg_reward_coords:
            return True
        else:
            return False
    

    def get_random_p_s_prime_s_a(self,s,a, primary_prob = 0.5):
        # s is a tuple of coordinates, a must be 'U','D','L','R' (and it must be in possible actions)
        # 
        # result: two {}, {(i1,j1):p1, (i2,j2):p2, ...} = {s_prime1:p1, s_prime2:p2,...} next state probabilities 
        #

        if a not in self.s_and_a_dict[s]:
            raise Exception('Can not do that action in that state')

        idx = self.s_and_a_dict[s].index(a)        
        res_s_prime_prob_dict = {}
        #res_a_prob_dict = {}
        if len(self.s_and_a_dict[s]) == 1:
            res_s_prime_prob_dict[self.s_and_s_prime_dict[s][0]] = 1.
            #res_a_prob_dict[a] = 1.
        elif len(self.s_and_a_dict[s]) > 1:
            n = len(self.s_and_a_dict[s]) - 1
            p = (1 - primary_prob) / n
            for i in range(len(self.s_and_a_dict[s])):
                if i == idx:
                    res_s_prime_prob_dict[self.s_and_s_prime_dict[s][i]] = primary_prob
                    #res_a_prob_dict[a] = primary_prob
                else:
                    res_s_prime_prob_dict[self.s_and_s_prime_dict[s][i]] = p
                    #res_a_prob_dict[self.s_and_a_dict[s][i]] = p
        else:
            raise Exception('s has no possible actions to use')

        return res_s_prime_prob_dict
    
    def get_deterministic_p_s_prime_s_a(self,s,a):
        #returns {s_prime:p}
        if a not in self.s_and_a_dict[s]:
            raise Exception('Can not do that action in that state')
        res_s_prime_prob_dict = {}
        idx = self.s_and_a_dict[s].index(a)
        s_prime = self.s_and_s_prime_dict[s][idx]
        res_s_prime_prob_dict[s_prime] = 1.
        return res_s_prime_prob_dict
    
    def initialize_V(self):
        V = {}
        for i in range(self.height):
            for j in range(self.length):
                state = (i,j)        
                if state in self.wall_blocks_coords:
                    continue
                else:
                    V[state] = 0
        return V 

    def iterative_policy_evaluation(self, policy, delta=0.01):
        # return s value function in the form of a grid
        V = self.initialize_V()
        error = np.inf
        while error > delta:
            V_old = V.copy()
            #iterate for all states
            for state in V:
                if state in self.pos_reward_coords:
                    continue
                elif state in self.neg_reward_coords:
                    continue               
                V[state] = self.get_new_V_s_policy_evaluation(state,V_old)
            
            error = self.calculate_error(V,V_old)
            #print('error = ',error)
        return V

    def calculate_error(self,V1,V2):
        e = 0
        for s in V1:
            e += np.abs(V1[s] - V2[s])
        return e
                
    def get_new_V_s_policy_evaluation(self,s,V):
        v = 0
        V_old = V.copy()
        #for a in possible actions from s
        for a in self.policy[s]:            
            # v1 = 0
            v1 = 0
            # get possible next states s_prime from (s,a) and their probabilities p(s_prime | s,a)
            s_prime_dict = self.get_deterministic_p_s_prime_s_a(s,a)

            # for s_prime in possible new states:
            for s_prime in s_prime_dict:
                # check if s_prime is terminal , if it is define reward, else reward r is 0
                if s_prime in self.pos_reward_coords:
                    r = self.pos_reward
                elif s_prime in self.neg_reward_coords:
                    r = self.neg_reward
                else:
                    r = 0
                # v1 += p(s_prime|s,a)*(r + self.gamma*V(s_prime))
                v1 += s_prime_dict[s_prime] * (r + self.gamma * V_old[s_prime])
            # v += pi(a|s) * v1
            v += self.policy[s][a] * v1
        return v 

    def initialize_random_policy(self):
        #initializes policy {s1:{a11:p11,a12:p12,..}, s2:{a21:p21,...}} where a11,itd are only possible actions from given state
        self.policy = {}
        for state in self.s_and_a_dict:
            if state in self.wall_blocks_coords:
                continue
            else:
                self.policy[state] = {}
                p = 1. / len(self.s_and_a_dict[state])
                for a in self.s_and_a_dict[state]:
                    self.policy[state][a] = p
    
    def get_Q_pi_s_a(self,V):
        # returns {s1:{a1:q1,a2:q1,...}, s2:{a1:q1,...}}

        q = {}
        # for s in possible states:
        for s in self.s_and_a_dict:
            q[s] = {}
            # for a in possible actions from s:
            for  a in self.s_and_a_dict[s]:
                v = 0
                #get possible next states from s,a , that is s_prime
                p_s_prime_s_a = self.get_deterministic_p_s_prime_s_a(s,a)
                for s_prime in p_s_prime_s_a:
                    # get p(s_prime | s,a)
                    p = p_s_prime_s_a[s_prime]
                    # get reward r for s_prime
                    if s_prime in self.pos_reward_coords:
                        r = self.pos_reward
                    elif s_prime in self.neg_reward_coords:
                        r = self.neg_reward
                    else:
                        r = 0 
                    #v += p(s_prime | s,a) * (r + gamma*V[s_prime] )
                    v += p * (r + self.gamma * V[s_prime])
                q[s][a] = v

        return q

    def get_improved_policy(self,Q):
        #Q is value action function for some policy
        #returns {s1:{a1 : 1.}, s2:{a2 : 1.}, ...}
        policy = {}
        for s in self.s_and_a_dict:
            policy[s] = {}
            maxi = -np.inf
            best_a = 'david'
            for a in self.s_and_a_dict[s]:
                q = Q.copy()[s][a]
                if q > maxi:
                    maxi = q
                    best_a = a
            policy[s][best_a] = 1.
        return policy

    def train_using_value_iteration(self,delta = 0.01):
        V = self.initialize_V()        
        while True:
            V_old = V.copy()
            for s in V:
                if s in self.pos_reward_coords:
                    continue
                elif s in self.neg_reward_coords:
                    continue
                v_s = []
                actions = self.s_and_a_dict[s].copy()
                for a in actions:
                    v = 0
                    p_s_prime_s_a = self.get_deterministic_p_s_prime_s_a(s,a)
                    for s_prime in p_s_prime_s_a:
                        p = p_s_prime_s_a[s_prime]
                        if s_prime in self.neg_reward_coords:
                            r = -1
                        elif s_prime in self.pos_reward_coords:
                            r = 1
                        else:
                            r = 0
                        v += p * (r + self.gamma * V_old[s_prime])
                    v_s.append(v)
                v_s = np.array(v_s)
                V[s] = np.max(v_s)
            
            #check if V has converged
            error = self.calculate_error(V,V_old)
            if error < delta:
                break
        
        

        #get Q
        Q = self.get_Q_pi_s_a(V)

        #get the policy
        self.policy = self.get_improved_policy(Q)
        self.V = V
                

    
    def train_using_policy_iteration(self,N):
        #initilize policy
        self.initialize_random_policy()
        for i in range(N):
            #get V for policy
            self.V = self.iterative_policy_evaluation(self.policy, delta=0.01)
            #get Q for V
            self.Q = self.get_Q_pi_s_a(self.V)
            #get updated policy
            self.policy = self.get_improved_policy(self.Q)
            
            print('i = ',i)

    
    def print_current_V(self):
        board = self.board.copy()
        for s in self.V:
            board[s] = self.V[s]
        print(board)
    
    def print_current_policy(self):
        board = np.chararray((self.height, self.length))
        board[:] = ''
        for s in self.policy:
            d = self.policy[s]
            board[s] = list(d.keys())[0]
        print(board)

    
    def pick_initial_state(self):
        possible_states = list(self.s_and_a_dict.keys())
        idx = random.randint(0,len(possible_states) - 1)
        return possible_states[idx]

    def pick_action_eps_greedy(self,eps,Q,s):
        # Q is of the form {s1:{a1:q1, a2:q2 ,...}, s2:{a1:q1, a2:q2 ,...} , ...}
        u = np.random.uniform()
        d = Q[s]
        possible_actions = list(d.keys())
        q_values = list(d.values())
        if u < eps:
            #pick random action
            idx = random.randint(0,len(possible_actions)-1)
            action = possible_actions[idx]
        else:
            #pick optimal action
            idx = np.argmax(np.array(q_values))
            action = possible_actions[idx]
        return action

    def get_reward(self,s):
        if s in self.pos_reward_coords:
            return self.pos_reward
        elif s in self.neg_reward_coords:
            return self.neg_reward
        else:
            return 0
    
    def get_s_prime_from_a_s(self,s,a):
        possible_actions = self.s_and_a_dict[s]
        if a not in possible_actions:
            raise Exception("This action can not be taken in this state")

        idx = possible_actions.index(a)
        s_prime = self.s_and_s_prime_dict[s][idx]
        return s_prime
    
    def play_episode(self, eps, Q):
        #initialieze list s_a_r [(s1,a1,r1), (s2,a2,r2), ...] in state s_i take action a_i and get instance reward r_i
        s_a_r  = []
        
        #initialize random state and action
        s = self.pick_initial_state()
        a = self.pick_action_eps_greedy(eps=1,Q=Q,s=s)
        

        
        while True:            
            s_prime = self.get_s_prime_from_a_s(s,a)
            #get reward from r s_prime
            r = self.get_reward(s_prime)
            #add (s , a , r) to list
            s_a_r.append((s,a,r))
            #if s_prime is termianal : break
            if self.check_if_terminal(s_prime):
                break
            s = s_prime
            #use eps-greedy strategy to pick next state s_prime
            a = self.pick_action_eps_greedy(eps=eps,Q=Q,s=s)
        
        return s_a_r

    def get_G(self,s_a_r):
        #s_a_r is a [(s1,a1,r1), (s2,a2,r2), ...] in state s_i take action a_i and get instance reward r_i
        #we use first seen principle
        # returns G_s = {s1:g1, s2:g2 ,...} and G_s_a = {(s1,a1):g1 , (s2,a2):g2 , ....}
        G_s = {}
        G_s_a = {}
        #G_s_david = []
        G_s_a_david = []
        seen_s = []
        seen_s_twice = []
        seen_s_a = []
        seen_s_a_twice = []
        G = 0
        for i in range(len(s_a_r)-1,-1,-1):
            s,a,r = s_a_r[i]            
            G = r + self.gamma * G
            #G_s_david.append([s, G])
            G_s_a_david.append([(s,a),G])           
        #G_s_david.reverse()
        G_s_a_david.reverse()
        for i in range(len(G_s_a_david)):
            s,a =  G_s_a_david[i][0]
            G = G_s_a_david[i][1]

            if s not in seen_s:
                G_s[s] = G
                seen_s.append(s)
            
            if (s,a) not in seen_s_a:
                G_s_a[(s,a)] = G
                seen_s_a.append((s,a))


        return G_s, G_s_a

    def update_V(self,G_s):
        for s in G_s:
            self.V[s] = np.mean(np.array(G_s[s]))
    
    def update_Q(self,G_s_a):
        for (s,a) in G_s_a:
            self.Q[s][a] = np.mean(np.array(G_s_a[(s,a)]))

    
    def initialize_Q(self):
        Q = {}
        for s in self.s_and_a_dict:
            Q[s] = {}
            for a in self.s_and_a_dict[s]:
                Q[s][a] = 0
        return Q
    
    def train_using_mc(self,N, eps=0.1):
        #initialize self.Q
        self.Q = self.initialize_Q()
        #initialize self.V
        self.V = self.initialize_V()

        G_s_david = {}
        G_s_a_david = {}

        for i in range(N):
            n = i + 1
            #play episode and get list of (s,a,r)
            s_a_r = self.play_episode(eps = eps, Q =self.Q)
            #constrcut function G(s) and G(s,a)
            G_s, G_s_a = self.get_G(s_a_r)

            for s in G_s:
                if s not in G_s_david:
                    G_s_david[s] = [G_s[s]]
                else:
                    G_s_david[s].append(G_s[s])
            
            for (s,a) in G_s_a:
                if (s,a) not in G_s_a_david:
                    G_s_a_david[(s,a)] = [G_s_a[(s,a)]]
                else:
                    G_s_a_david[(s,a)].append(G_s_a[(s,a)])

            #update self.Q
            self.update_Q(G_s_a_david)
            #update self.V
            self.update_V(G_s_david)
            print('i = ',i)

        # get policy optimal from Q
        self.policy = self.get_improved_policy(self.Q)

    
    def get_V_and_polciy_from_Q(self,Q):
        V = {}
        policy = {}
        for s in Q:
            maxi = -np.inf
            best_a = 'david'
            if s == (2,3):
                print(Q[s])
            for a in Q[s]:
                if Q[s][a] > maxi:
                    maxi = Q[s][a]
                    best_a = a
            V[s] = maxi
            policy[s] = {best_a:1}
        return V, policy
            
    def get_max_dict(self,d):
        keys = list(d.keys())
        values = list(d.values())
        idx = np.argmax(np.array(values))
        return keys[idx], values[idx]

    def play_episode_q_learning(self, alpha0):
        #pick initial state s
        s = self.pick_initial_state()        
        
        while True:
            #pick random action a from state s 
            a = self.pick_action_eps_greedy(eps=1,Q=self.Q,s=s)        
            #eps = 0.5/self.t
            #get next state s_prime
            s_prime = self.get_s_prime_from_a_s(s,a)
            #get reward r from (s,a)
            r = self.get_reward(s_prime)

            if (s,a) in self.count_s_a:
                self.count_s_a[(s,a)] += 0.005
            else:
                self.count_s_a[(s,a)] = 1

            if self.check_if_terminal(s_prime):
                self.Q[s][a] = r                
                break
            else:
                #pick action a_prime from state s_prime                    
                #a_prime = self.pick_action_eps_greedy(eps=eps,Q=self.Q,s=s_prime)
                alpha = alpha0 / self.count_s_a[(s,a)]
                Q_s_a = self.Q[s][a]
                _, Q_s_prime_max = self.get_max_dict(self.Q[s_prime])
                self.Q[s][a] = Q_s_a + alpha*(r + self.gamma*Q_s_prime_max - Q_s_a)

            s = s_prime

    def train_unsing_q_learning(self,N,alpha0):
        self.t = 1

        self.count_s_a = {}
        #initialze Q
        self.Q = self.initialize_Q()
        #self.V = self.initialize_V()
        for i in range(N):
            if i % 100 == 0:
                self.t += 0.01

            #play episode
            self.play_episode_q_learning(alpha0)
            print('i = ',i)
        
        #get V function
        self.V, self.policy = self.get_V_and_polciy_from_Q(self.Q) 

    def play_episode_sarsa(self,alpha0):
        #pick initial state s
        s = self.pick_initial_state()       
        #s = (self.height - 1, 0)
        t = 1
        #eps = 1/t
        eps = 0.1
        #pick random action a from state s eps
        a = self.pick_action_eps_greedy(eps=eps,Q=self.Q,s=s)
        
        while True:
            #print('t = ',t)
            #t += 1
            eps = 0.5/self.t
            #eps = 0.1
            #get next state s_prime
            s_prime = self.get_s_prime_from_a_s(s,a)
            #get reward r from (s,a)
            r = self.get_reward(s_prime)

            if (s,a) in self.count_s_a:
                self.count_s_a[(s,a)] += 0.005
            else:
                self.count_s_a[(s,a)] = 1

            if self.check_if_terminal(s_prime):
                self.Q[s][a] = r
                #self.V[s] = r
                break
            else:
                #pick action a_prime from state s_prime                    
                a_prime = self.pick_action_eps_greedy(eps=eps,Q=self.Q,s=s_prime)
                alpha = alpha0 / self.count_s_a[(s,a)]
                Q_s_a = self.Q[s][a]
                Q_s_prime_a_prime = self.Q[s_prime][a_prime]                
                self.Q[s][a] = Q_s_a + alpha*(r + self.gamma*Q_s_prime_a_prime - Q_s_a)
                #V_s = self.V[s]
                #V_s_prime = self.V[s_prime]
                #self.V[s] = V_s + 0.5 * (r + self.gamma * V_s_prime - V_s)

                        
            a = a_prime
            s = s_prime

    def train_unsing_srsa(self,N,alpha0):
        self.t = 1

        self.count_s_a = {}
        #initialze Q
        self.Q = self.initialize_Q()
        #self.V = self.initialize_V()
        for i in range(N):
            if i % 100 == 0:
                self.t += 0.01

            #play episode
            self.play_episode_sarsa(alpha0)
            print('i = ',i)
        
        #get V function
        self.V, self.policy = self.get_V_and_polciy_from_Q(self.Q)
        




if __name__ == '__main__':

    grid = Grid(length = 5,
                height = 3,
                gamma = 0.9,
                wall_blocks_coords = [(1,1)],
                pos_reward_coords = [(0,4)],
                neg_reward_coords = [(1,4)])
    #grid.train_using_policy_iteration(N=100)
    #grid.train_using_value_iteration(delta = 0.01)
    #grid.train_using_mc(N = 10000, eps=0.1)
    #grid.train_unsing_srsa(N=100000,alpha0=1)
    #grid.train_unsing_q_learning(N=10000,alpha0=1)
    print("CURRENT V")
    grid.print_current_V()
    print("CURRENT POLICY")
    grid.print_current_policy()


        

        

        

        

        

            




    

                
         
        
            




                
                