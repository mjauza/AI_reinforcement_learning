import numpy as np

class BanditNormal():
    def __init__(self,m):
        self.m = m
        self.m_hat = 0
        self.N = 0
        self.s = 0
    
    def pull(self):
        res = np.random.normal(self.m)
        self.N += 1
        self.s += res
        self.m_hat = self.s / self.N

    def get_ucb1(self,N):
        N1 = self.N if self.N > 0 else 0.0001
        return self.m_hat + np.sqrt(2*np.log(N)/N1)



def pick_bandit_normal(eps,bandit_list,N):
    maxi = -np.inf
    for i in range(len(bandit_list)):        
        if bandit_list[i].get_ucb1(N) > maxi:
            selected_index = i
            maxi = bandit_list[i].get_ucb1(N)
    return selected_index

def run_experiment_normal(m1,m2,m3,eps,N):
    bandit_list = [BanditNormal(m1), BanditNormal(m2), BanditNormal(m3)]
    for i in range(N):
        selected_index = pick_bandit_normal(eps, bandit_list, i+1)
        selected_bandit = bandit_list[int(selected_index)]
        selected_bandit.pull()
    
    for i in range(len(bandit_list)):
        print('Bandit number {} , m = {} , m_hat = {} , N = {}'.format(i, bandit_list[i].m, bandit_list[i].m_hat, bandit_list[i].N))

if __name__ == '__main__':
    run_experiment_normal(m1=0.3,m2=2,m3=2.1,eps=0.1,N=10000)