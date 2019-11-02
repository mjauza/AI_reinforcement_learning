import numpy as np
from itertools import product
import random
import pandas as pd

class Agent():
    def __init__(self,mark,alpha,eps):
        self.initialize_V()
        self.mark = mark
        self.alpha = alpha
        self.eps = eps
    
    def initialize_V(self):        
        keys = []
        david = product('012', repeat=9)
        for d in david:
            keys.append(''.join(d))
        self.keys = keys
        self.V = pd.Series(np.zeros((len(self.keys),)), index = self.keys)
    
    def get_optimal_move(self, possible_move_keys):
        u = np.random.uniform()        
        if u < self.eps:
            selected_move_key_index =  random.randint(0,len(possible_move_keys)-1)
            selected_move_key = possible_move_keys[selected_move_key_index]
            return selected_move_key           
        else:
            possible_V = self.V[possible_move_keys].values.copy()
            best_V_index = np.argmax(possible_V)
            selected_move_key = possible_move_keys[best_V_index]
            return selected_move_key
    
    def make_move(self,possible_move_keys):
        optimal_move_key = self.get_optimal_move(possible_move_keys)
        #self.state_list.append(optimal_move_key)
        self.cumulative_rewards += -1
        return optimal_move_key
    
    def update_state_list(self, state):
        self.state_list.append(state)

    def make_move_against_david(self,possible_move_keys):
        possible_V = self.V[possible_move_keys].values.copy()
        best_V_index = np.argmax(possible_V)
        optimal_move_key = possible_move_keys[best_V_index]
        self.state_list.append(optimal_move_key)
        #self.cumulative_rewards += -1
        return optimal_move_key

    
    def update_V(self):
        self.V[self.state_list[-1]] = self.cumulative_rewards        
        for i in range(len(self.state_list)-2,-1,-1):
            V_s_prime = self.V[self.state_list[i+1]].copy()
            V_s = self.V[self.state_list[i]].copy()
            self.V[self.state_list[i]] = V_s + self.alpha*(V_s_prime - V_s)
    
    def intialize_game_objects(self):
        self.state_list = []
        self.cumulative_rewards = 0

class Environment():
    def __init__(self):
        pass

    def build_board(self):
        self.board = np.zeros((3,3))
    
    def get_possible_positions(self,mark):
        pos = np.where(self.board == 0)        
        keys_list = []
        for i in range(len(pos[0])):
            board = self.board.copy()
            board[pos[0][i], pos[1][i]] = mark
            key = self.pos2key(board)
            keys_list.append(key)
        
        return keys_list
    
    def pos2key(self,board):
        board_list = board.reshape((9,)).tolist()
        board_str_list = [str(int(v)) for v in board_list]
        return ''.join(board_str_list)

    def check_winner(self,marks):
        mark1 = marks[0]
        mark2 = marks[1]
        first_diagonal = [[0,1,2],[0,1,2]]
        second_diagonal = [[0,1,2],[2,1,0]]

        #CHECK FOR WINNER
        for i in range(3):
            if np.sum(self.board[i,:] == mark1) == 3:
                return 'player1'
            elif np.sum(self.board[i,:] == mark2) == 3:
                return 'player2'
            elif np.sum(self.board[:,i] == mark1) == 3:
                return 'player1'
            elif np.sum(self.board[:,i] == mark2) == 3:
                return 'player2'
        if np.sum(self.board[first_diagonal[0],first_diagonal[1]] == mark1) == 3:
            return 'player1'
        elif np.sum(self.board[first_diagonal[0],first_diagonal[1]] == mark2) == 3:
            return 'player2'
        elif np.sum(self.board[second_diagonal[0],second_diagonal[1]] == mark1) == 3:
            return 'player1'
        elif np.sum(self.board[second_diagonal[0],second_diagonal[1]] == mark2) == 3:
            return 'player2'
        
        #CHECK FOR DRAW
        pos = np.where(self.board == 0)
        if len(pos[0]) == 0:
            return 'draw'
        
        return 'no winner'
    
    def update_board(self,position_key):
        position_key_list = list(position_key)
        position_key_int = [int(s) for s in position_key_list]
        position_np = np.array(position_key_int).reshape((3,3))
        self.board = position_np.copy()
    

class Game():
    def __init__(self,N):
        self.N = N
        self.games_won_agent1 = 0
        self.games_won_agent2 = 0
        self.marks = [1,2]
        self.agent1 = Agent(mark=self.marks[0],alpha=0.1,eps=0.1)
        self.agent2 = Agent(mark=self.marks[1],alpha=0.1,eps=0.1)
        self.first_to_play = 'player1'

        
    def play_game(self):
        self.env = Environment()
        self.env.build_board()
        self.game_over = False
        self.agent1.intialize_game_objects()
        self.agent2.intialize_game_objects()
        n = 0
        while True:            
            self.pre_move_procedure()
            if self.game_over:                            
                self.agent1.update_V()
                self.agent2.update_V()
                #print('cumulative rewards agent 1 = ',self.agent1.cumulative_rewards)
                #print('cumulative rewards agent 2 = ',self.agent2.cumulative_rewards)
                break
            
            #move agent
            if self.first_to_play == 'player1':
                possible_moves_agent1 = self.env.get_possible_positions(self.agent1.mark)            
                agent1_move = self.agent1.make_move(possible_moves_agent1)
                self.agent1.update_state_list(agent1_move)
                self.agent2.update_state_list(agent1_move)        
                self.env.update_board(agent1_move)
            else:
                possible_moves_agent2 = self.env.get_possible_positions(self.agent2.mark)
                agent2_move = self.agent2.make_move(possible_moves_agent2)
                self.agent1.update_state_list(agent2_move)
                self.agent2.update_state_list(agent2_move)
                self.env.update_board(agent2_move)
            n += 1           

            self.pre_move_procedure()
            if self.game_over:                
                self.agent1.update_V()
                self.agent2.update_V()
                #print('cumulative rewards agent 1 = ',self.agent1.cumulative_rewards)
                #print('cumulative rewards agent 2 = ',self.agent2.cumulative_rewards)
                break

            #move agent
            if self.first_to_play == 'player1':
                possible_moves_agent2 = self.env.get_possible_positions(self.agent2.mark)
                agent2_move = self.agent2.make_move(possible_moves_agent2)
                self.agent1.update_state_list(agent2_move)
                self.agent2.update_state_list(agent2_move)
                self.env.update_board(agent2_move)
            else:
                possible_moves_agent1 = self.env.get_possible_positions(self.agent1.mark)            
                agent1_move = self.agent1.make_move(possible_moves_agent1)
                self.agent1.update_state_list(agent1_move)
                self.agent2.update_state_list(agent1_move)
                self.env.update_board(agent1_move)
            n += 1
        
        if self.first_to_play == 'player1':
            self.first_to_play = 'player2'
        else:
            self.first_to_play = 'player1'

        if n > self.max_duration:
            self.max_duration = n
            
    
    def pre_move_procedure(self):
        #check if there is winner:
        w = self.env.check_winner(marks = self.marks)
        if w == 'player1':
            self.agent1.cumulative_rewards = 10
            self.agent2.cumulative_rewards = -10
            self.game_over = True
            self.games_won_agent1 += 1
            #print('player 1 has won')
        elif w == 'player2':
            self.agent2.cumulative_rewards = 10
            self.agent1.cumulative_rewards = -10
            self.game_over = True
            self.games_won_agent2 += 1
            #print('player 2 has won')
        elif w == 'draw':
            self.agent2.cumulative_rewards = -1
            self.agent1.cumulative_rewards = -1
            self.game_over = True
            #print('it is a draw')
        else:
            return None

    def train(self):        
        self.max_duration = 0
        for i in range(self.N):
            self.play_game()
            print('i = ',i)
        
        print('max game duration = ',self.max_duration)
    
    def get_better_player(self):
        if self.games_won_agent1 > self.games_won_agent2:
            print("better agent is agent 1")
            return self.agent1            
        else:
            print("better agent is agent 2")
            return self.agent2

    def play_vs_david(self):
        self.env_david = Environment()
        self.env_david.build_board()
        
        agent = self.get_better_player()
        self.david_mark = 1 if agent.mark == 2 else 2
        print('david mark = ', self.david_mark)
        self.marks_david_agent = [self.david_mark, agent.mark]
        self.david_game_over = False        
        while True:
            #check if there is a end of
            self.pre_move_procedure_david()
            if self.david_game_over:
                break
            #david make a move
            print(self.env_david.board)
            self.make_move_david()

            self.pre_move_procedure_david()
            if self.david_game_over:
                break

            #agent make move
            possible_moves = self.env_david.get_possible_positions(agent.mark)
            agent_move = agent.make_move_against_david(possible_moves)            
            self.env_david.update_board(agent_move)

    
    def pre_move_procedure_david(self):        
        #check if there is winner:
        w = self.env_david.check_winner(marks = self.marks_david_agent)
        if w == 'player1':            
            self.david_game_over = True
            print('David Won!')
            
        elif w == 'player2':            
            self.david_game_over = True
            print('Agent won!')
            
        elif w == 'draw':
            self.david_game_over = True
            print('It is a draw!!')
        else:
            return None

    def draw_board(self):
        print(self.env_david.board)
    
    def get_move_coords_from_input(self):
        row = int(input("Type row coord (0-2)"))
        column = int(input("Type column coord (0-2)"))
        return [row,column]
    
    def make_move_david(self):
        while True:
            coords = self.get_move_coords_from_input()
            if coords[0] >= 0 and coords[1] >= 0 and coords[0] <= 2 and coords[1] <= 2:
                if self.env_david.board[coords[0],coords[1]] == 0:
                    self.env_david.board[coords[0],coords[1]] = self.david_mark
                    break
                else: 
                    print("Wrong position choosen")
            else:
                print("Wrong position choosen")


if __name__ == '__main__':
    game = Game(10000)
    game.train()
    print('games won by agent 1 = ',game.games_won_agent1)
    print('games won by agent 2 = ',game.games_won_agent2)
    while True:
        game.play_vs_david()
        s = input("Next game? y/n")
        while s != 'y' and s != 'n':
            s = input("Next game? y/n")
        
        if s=='n':
            break
     
        