import numpy as np
from itertools import product
import random

class TicTacToe():
    def __init__(self, N):                
        self.coords_list = [[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]]
        self.alpha = 0.1
        self.initialize_V()
        self.eps = 0.1
        self.games_won = 0
        self.N = N

    
    def play_game_vs_player(self):
        self.game_over = False
        self.build_board()
        while True:
            self.display_board()
            self.procedure_before_move_vs_player()
            if self.game_over:
                break
            #david move            
            self.david_make_move()
            #check game over
            self.procedure_before_move_vs_player()
            if self.game_over:
                break
            #player move
            self.make_player_move()


    def procedure_before_move_vs_player(self):
        cw = self.check_winner()
        if cw == 'player':            
            self.game_over = True 
            print("Player won")          
        elif cw == 'system':            
            self.game_over = True
            print('You won')
        else: 
            poss = self.check_possible_moves()
            if len(poss[0]) == 0:          
                self.game_over = True
                print('It is a draw')     
    
    def david_make_move(self):
        while True:
            coords = self.get_move_coords_from_input()
            res = self.move_david_to_coord(coords)
            if res != 'invalid coord':
                break

    def get_move_coords_from_input(self):
        row = int(input("Type row coord (0-2)"))
        column = int(input("Type column coord (0-2)"))
        return [row,column]
    
    def move_david_to_coord(self,coord):
        if coord[0] <= 2 and coord[0] >= 0 and coord[1] <= 2 and coord[1] >= 0:            
            if self.board[coord[0],coord[1]] == 0:
                self.board[coord[0],coord[1]] = 2
            else:            
                return 'invalid coord'
        else:
            return 'invalid coord'

    def display_board(self):
        print(self.board)

    def train_player(self):
        self.max_duration = 0
        for n in range(self.N):
            self.play_game()
            print('n = ',n)
        
        print("max duration = ",self.max_duration)

    def play_game(self):
        self.game_over = False
        self.state_list = []
        #initialize board        
        self.build_board()
        self.cumulative_rewards = 0       
        #while true
        n = 0
        while True:            
            #check for game over
            self.procedure_before_move()            
            if self.game_over:
                #print("AAAAAAAAAAAAAAAAAAAAAAAA")
                break
            #system move
            self.make_system_move()
            n += 1
            #check game over
            self.procedure_before_move()
            if self.game_over:
                #print("AAAAAAAAAAAAAAAAAAAAAAAA")
                break
            #player move
            self.make_player_move()
            self.cumulative_rewards += -1
            n += 1
        #print('length of game = ',n)
        if n > self.max_duration:
            self.max_duration = n
        


    def build_board(self):
        self.board = np.zeros((3,3))

    def procedure_before_move(self):
        cw = self.check_winner()
        if cw == 'player':            
            state_key = self.get_key_from_position(self.board.copy())
            #self.state_list.append(state_key)
            final_reward = 10             
            self.game_over = True
            self.games_won += 1
            #print("PLAYER WON")
        elif cw == 'system':
            final_reward = -1
            self.game_over = True            
            #print("SYSTEM WON")
        else:
            poss = self.check_possible_moves()            
            if len(poss[0]) == 0:
                final_reward = -1
                self.game_over = True        
                #print("DRAW")
            else:    
                #print("DAVID WON")
                final_reward = 0
        self.cumulative_rewards += final_reward
        if self.game_over:
            self.update_V(self.state_list.copy(),self.cumulative_rewards)

        #print('game over = ',self.game_over)
    


    def make_player_move(self):        
        best_position = self.pick_best_move()
        self.move_to_position(best_position)
        state_key = self.get_key_from_position(self.board.copy())
        self.state_list.append(state_key)

    
    def pick_best_move(self):
        u = np.random.uniform()        
        if u < self.eps:
            possibilites = self.check_possible_moves()    
            #selected_idx = np.random.random_integers(low=0,high=len(possibilites[0])-1)
            selected_idx = random.randint(0,len(possibilites[0])-1)
            selected_position = [possibilites[0][selected_idx],possibilites[1][selected_idx]]
        else:
            #print(self.board)            
            possible_next_states_keys = self.possible_next_states()
            indexes = []
            #print(possible_next_states_keys)
            for i in range(len(possible_next_states_keys)):
                idx = self.keys.index(possible_next_states_keys[i])
                indexes.append(idx)
            possible_V = self.V[indexes].copy()
            best_V_idx = np.argmax(possible_V)
            best_key = possible_next_states_keys[best_V_idx]
            #print('best_key = ',best_key)
            board_key = self.get_key_from_position(self.board.copy())
            selected_position = self.get_coords_from_positions(board_key, best_key)
            #print(selected_position)
        
        return selected_position
    

    
    def update_V(self,state_list,final_V):
        self.V[self.keys.index(state_list[-1])] = final_V
        for i in range(len(state_list)-2,-1,-1):
            V_s_prime = self.V[self.keys.index(state_list[i+1])].copy()
            V_s = self.V[self.keys.index(state_list[i])].copy()
            self.V[self.keys.index(state_list[i])] = V_s + self.alpha*(V_s_prime - V_s)

            
    
    def get_coords_from_positions(self,pos_key1,pos_key2):
        pos_key_list1 = list(pos_key1)
        pos_key_list2 = list(pos_key2)
        for i in range(len(pos_key_list1)):
            if pos_key_list1[i] != pos_key_list2[i]:
                idx = i
                break
        
        return self.coords_list[idx]


    def get_key_from_position(self,pos):
        key_list = pos.reshape(9,).tolist()
        key = ''
        for i in range(9):
            key += str(int(key_list[i]))        
        return key


    def possible_next_states(self):
        possibilites = self.check_possible_moves()
        key_list = []
        for i in range(len(possibilites[0])):
            p = [possibilites[0][i],possibilites[1][i]]
            pos = self.board.copy()
            pos[p[0],p[1]] = 1
            key = self.get_key_from_position(pos)
            key_list.append(key)
        return key_list.copy()

    def move_to_position(self, position):
        if self.board[position[0],position[1]] == 0:
            self.board[position[0],position[1]] = 1
        else:
            raise Exception("can not make move on this position")
    
    def check_possible_moves(self):
        posibilites = np.where(self.board == 0)
        return posibilites

    
    def make_system_move(self):
        posibilites = self.check_possible_moves()
        #res = np.random.random_integers(low=0,high=len(posibilites[0])-1)
        res = random.randint(0,len(posibilites[0])-1)
        selected_pos = [posibilites[0][int(res)], posibilites[1][int(res)]]
        if self.board[selected_pos[0],selected_pos[1]] == 0:
            self.board[selected_pos[0],selected_pos[1]] = 2
        else:
            raise Exception('Wrong position to pick')
    
    def check_winner(self):
        #user = 'player' is 1
        #user = 'system' is 2
        first_diagonal = [[0,1,2],[0,1,2]]
        second_diagonal = [[0,1,2],[2,1,0]]
        for i in range(3):
            if np.sum(self.board[i,:] == 1) == 3:
                return 'player'
            elif np.sum(self.board[i,:] == 2) == 3:
                return 'system'
            elif np.sum(self.board[:,i] == 1) == 3:
                return 'player'
            elif np.sum(self.board[:,i] == 2) == 3:
                return 'system'
        if np.sum(self.board[first_diagonal[0],first_diagonal[1]] == 1) == 3:
            return 'player'
        elif np.sum(self.board[first_diagonal[0],first_diagonal[1]] == 2) == 3:
            return 'system'
        elif np.sum(self.board[second_diagonal[0],second_diagonal[1]] == 1) == 3:
            return 'player'
        elif np.sum(self.board[second_diagonal[0],second_diagonal[1]] == 2) == 3:
            return 'system'
        
        return 'no winner'
    
    def initialize_V(self):
        keys = []
        david = product('012', repeat=9)
        for d in david:
            keys.append(''.join(d))
        self.keys = keys
        self.V = np.zeros((len(self.keys),))


if __name__ == '__main__':
    game = TicTacToe(100000)
    game.train_player()
    print('game accuracy = {}'.format(game.games_won / game.N))
    game.play_game_vs_player()