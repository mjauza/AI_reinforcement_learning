import numpy as np

class BanditBinary():
    def __init__(self,p):
        self.p = p
        self.p_hat = 0
        self.N = 0
        self.s = 0
    
    def pull(self):
        res = np.random.binomial(1,p=self.p, size=1)
        self.N += 1
        self.s += res
        self.p_hat = self.s / self.N

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

def pick_bandit_binary(eps,bandit_list):
    res = np.random.binomial(1,p=eps,size=1)[0]
    
    if res == 1:
        selected_index = np.random.randint(len(bandit_list), size=1)
    else:
        maxi = -1
        for i in range(len(bandit_list)):
            if bandit_list[i].p_hat > maxi:
                selected_index = i
                maxi = bandit_list[i].p_hat
    return selected_index


def pick_bandit_normal(eps,bandit_list):
    res = np.random.binomial(1,p=eps,size=1)[0]
    
    if res == 1:
        selected_index = np.random.randint(len(bandit_list), size=1)
    else:
        maxi = -1
        for i in range(len(bandit_list)):
            if bandit_list[i].m_hat > maxi:
                selected_index = i
                maxi = bandit_list[i].m_hat
    return selected_index

def run_experiment_binary(p1,p2,p3,eps,N):
    bandit_list = [BanditBinary(p1), BanditBinary(p2), BanditBinary(p3)]
    for i in range(N):
        selected_index = pick_bandit_binary(eps, bandit_list)
        selected_bandit = bandit_list[int(selected_index)]
        selected_bandit.pull()
    
    for i in range(len(bandit_list)):
        print('Bandit number {} , p = {} , p_hat = {} , N = {}'.format(i, bandit_list[i].p, bandit_list[i].p_hat, bandit_list[i].N))


def run_experiment_normal(m1,m2,m3,eps,N):
    bandit_list = [BanditNormal(m1), BanditNormal(m2), BanditNormal(m3)]
    for i in range(N):
        selected_index = pick_bandit_normal(eps, bandit_list)
        selected_bandit = bandit_list[int(selected_index)]
        selected_bandit.pull()
    
    for i in range(len(bandit_list)):
        print('Bandit number {} , m = {} , m_hat = {} , N = {}'.format(i, bandit_list[i].m, bandit_list[i].m_hat, bandit_list[i].N))

    


if __name__ == '__main__':
    run_experiment_normal(m1=0.3,m2=2,m3=2.1,eps=0.1,N=10000)