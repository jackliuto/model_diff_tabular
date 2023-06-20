from collections import defaultdict
import random
import numpy as np
import copy

from gym.envs.toy_text.utils import categorical_sample

from utils import plot_matrix, plot_policy_matrix, plot_line_dict, plot_Qdiff_matrix


class DPAgent:
    def __init__(self, env, gamma=1, theta=1e-8):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.S = np.arange(0, self.env.nS).reshape(self.env.nrow, self.env.ncol)

    def policy_evaluation(self, policy, gamma=None, max_iter=np.inf) :
        n = 0
        if gamma == None:
            gamma = self.gamma
        V = np.zeros(self.env.nS)
        num_iter = 0
        while True and num_iter < max_iter:
            delta = 0
            V_prev = V.copy()
            for s in range(self.env.nS):
                Vs = 0
                for a, action_prob in enumerate(policy[s]):
                    for prob, next_state, reward, done in self.env.P[s][a]:
                        Vs += action_prob * prob * (reward + self.gamma * V_prev[next_state])
                delta = max(delta, np.abs(V[s]-Vs))
                V[s] = Vs
                # if s == 48 or s==45:
                #     V[s] = 0
            num_iter += 1
            if delta < self.theta:
                break
        Q = self.gen_q_table(V)
        return V, Q, num_iter

    def q_from_v(self, V, s):
        q = np.zeros(self.env.nA)
        for a in range(self.env.nA):
            for prob, next_state, reward, done in self.env.P[s][a]:
                q[a] += prob * (reward + self.gamma * V[next_state])
        return q

    def policy_improvement(self, V, det_policy=False):
        policy = np.zeros([self.env.nS, self.env.nA]) / self.env.nA
        for s in range(self.env.nS):
            q = self.q_from_v(V, s)

            # OPTION 1: construct a deterministic policy
            # policy[s][np.argmax(q)] = 1

            # OPTION 2: construct a stochastic policy that puts equal probability on maximizing actions
            # best_a = np.argwhere(q==np.max(q)).flatten()
            # policy[s] = np.sum([np.eye(self.env.nA)[i] for i in best_a], axis=0)/len(best_a)

            if det_policy:
                policy[s][np.argmax(q)] = 1
            else:
                best_a = np.argwhere(q==np.max(q)).flatten()
                policy[s] = np.sum([np.eye(self.env.nA)[i] for i in best_a], axis=0)/len(best_a)

            # if self.env.desc[s//self.env.ncol][s%self.env.ncol] in b"G":
            #     policy[s] = [0.0, 0.0, 0.0, 0.0, 1.0]


        return policy

    def policy_iteration(self, init_P = []):
        if init_P == []:
            policy = np.ones([self.env.nS, self.env.nA]) / self.env.nA
        else:
            policy = init_P
        steps = 0
        while True:
            V, Q, _ = self.policy_evaluation(policy)
            new_policy = self.policy_improvement(V)

            # OPTION 1: stop if the policy is unchanged after an improvement step
            if (new_policy == policy).all():
                break

            # OPTION 2: stop if the value function estimates for successive policies has converged
            # if np.max(abs(policy_evaluation(env, policy) - policy_evaluation(env, new_policy))) < theta*1e2:
            #    break;

            policy = copy.copy(new_policy)
            steps += 1
        return policy, V, steps
    
    def gen_q_table(self, V):
        Q = []
        for s in range(len(V)):
            Q.append(self.q_from_v(V,s))
        return np.array(Q)


    def value_iteration(self, max_iter=np.inf, max_steps=np.inf, det_policy = False):
        V = np.zeros(self.env.nS)
        break_loop = False
        steps = 0 
        num_iter = 0
        while True and num_iter < max_iter:
            delta = 0
            V_prev = V.copy()
            for s in range(self.env.nS): 
                v = V[s]
                V[s] = max(self.q_from_v(V_prev, s))
                delta = max(delta, abs(V[s]-v))
                steps += 1
                if steps >= max_steps:
                    break_loop = True
            num_iter += 1
            if delta < self.theta or break_loop:
                break
        policy = self.policy_improvement(V, det_policy)
        Q = self.gen_q_table(V)
        return policy, V, Q, steps, num_iter

    def value_iteration_warmstart(self, max_iter=np.inf, max_steps=np.inf, det_policy = False, init_V=None):
        V = init_V.copy()
        break_loop = False
        steps = 0 
        num_iter = 0
        while True and num_iter < max_iter:
            delta = 0
            V_prev = V.copy()
            for s in range(self.env.nS): 
                v = V[s]
                V[s] = max(self.q_from_v(V_prev, s))
                delta = max(delta, abs(V[s]-v))
                steps += 1
                if steps >= max_steps:
                    break_loop = True
            num_iter += 1
            if delta < self.theta or break_loop:
                break
            # np.set_printoptions(precision=2)
            # print(sum(abs(V - init_V)))
            
        policy = self.policy_improvement(V, det_policy)
        Q = self.gen_q_table(V)
        return policy, V, Q, steps, num_iter



class QLearningAgent:
    def __init__(self, env, gamma=0.9, alpha=0.5, epsilon=0.1, theta=1e-8):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.theta = theta
        self.V = np.zeros([self.env.nS])
        self.Q = np.zeros([self.env.nS, self.env.nA])
        self.Policy = np.ones([self.env.nS, self.env.nA]) / self.env.nA
        self.Vdiff = None
        self.S = np.arange(0, self.env.nS).reshape(self.env.nrow, self.env.ncol)
        self.Q_heur = None

        self.ppr_psi = None
        self.ppr_upsilon = None

    def softmax_t(self, x, temp):

        e_x = np.exp((1/temp)*(np.array(x))) ## need to be corrected
        dist = e_x / e_x.sum()
        return dist

    def warm_start_q(self, Q):
        self.Q = Q.copy()
        for s in range(len(self.Policy)):
            self.update_policy(s)

    def q_from_v(self, V, s):
        q = np.zeros(self.env.nA)
        for a in range(self.env.nA):
            for prob, next_state, reward, done in self.env.P[s][a]:
                q[a] += prob * (reward + self.gamma * V[next_state])
        return q
    
    def choose_action_egreedy(self, s):

        action_dist = self.Policy[s]

        if np.random.random() < self.epsilon:
            a = self.env.action_space.sample()
        else:
            a = categorical_sample(action_dist, self.env.np_random)
        return a

    def choose_action_ppr(self, s, psi):
        if np.random.random() < psi:
            q = self.Q_heur[s]
            best_a = np.argwhere(q==np.max(q)).flatten()
            action_dist = np.sum([np.eye(self.env.nA)[i] for i in best_a], axis=0)/len(best_a)
            a = categorical_sample(action_dist, self.env.np_random)
        else:
            if np.random.random() < self.epsilon:
                a = self.env.action_space.sample()
            else:
                action_dist = self.Policy[s]
                a = categorical_sample(action_dist, self.env.np_random)
        return a

    def update_value(self, beg_s, end_s, r):
        self.V[beg_s] = max(self.V[beg_s], r + self.gamma*self.V[end_s])

    def update_q(self, beg_s, end_s, a, r):
        self.Q[beg_s][a] += self.alpha * (r + self.gamma * np.max(self.Q[end_s]) - self.Q[beg_s][a])
    
    def update_q_bound(self, beg_s, end_s, a, r, update_q_lower):
        Q = self.Q[beg_s][a] + self.alpha * (r + self.gamma * np.max(self.Q[end_s]) - self.Q[beg_s][a])
        Q_lower = self.Q_lower[beg_s][a]
        if Q >= Q_lower:
            self.Q[beg_s][a] = Q
        else:
            self.Q[beg_s][a] = Q_lower
        if update_q_lower:
            self.Q_lower[beg_s][a] = self.Q_lower[beg_s][a] + self.alpha * (r + self.gamma * np.max(self.Q_lower[end_s]) - self.Q_lower[beg_s][a])

        
    def update_policy(self, s):
        q = self.Q[s]
        best_a = np.argwhere(q==np.max(q)).flatten()
        self.Policy[s] = np.sum([np.eye(self.env.nA)[i] for i in best_a], axis=0)/len(best_a)

    def run_warmstart(self, init_Q, max_step, num_eps, explore='e-greedy'):
        self.warm_start_q(init_Q)
        dis_reward = []
        for i in range(num_eps):
            beg_s, init_prob = self.env.reset()
            steps = 0
            dis_r = 0
            while steps < max_step:
                if explore == 'e-greedy':
                    a = self.choose_action_egreedy(beg_s)
                else:
                    raise ValueError
                end_s, r, t, _, _ = self.env.step(a)
                # self.update_value(beg_s, end_s, r)
                self.update_q(beg_s, end_s, a, r)
                self.update_policy(beg_s)
                beg_s = end_s
                steps += 1
                dis_r += self.gamma**steps*r
            dis_reward.append(dis_r)
        return dis_reward

    def run_lowerbound(self, Q_lower, max_step, num_eps, explore='e-greedy', update_q_lower=False):
        dis_reward = []
        self.Q_lower = Q_lower
        for i in range(num_eps):
            beg_s, init_prob = self.env.reset()
            steps = 0
            dis_r = 0
            while steps < max_step:
                if explore == 'e-greedy':
                    a = self.choose_action_egreedy(beg_s)
                else:
                    raise ValueError
                end_s, r, t, _, _ = self.env.step(a)
                # self.update_value(beg_s, end_s, r)
                self.update_q_bound(beg_s, end_s, a, r, update_q_lower)
                self.update_policy(beg_s)
                beg_s = end_s
                steps += 1
                dis_r += self.gamma**steps*r
            dis_reward.append(dis_r)
        return dis_reward
    
    def run_ppr(self, Q_heur, max_step, num_eps, psi, upsilon, explore='ppr'):
            self.Q_heur = Q_heur
            dis_reward = []
            for i in range(num_eps):
                
                beg_s, init_prob = self.env.reset()
                steps = 0
                dis_r = 0
                while steps < max_step:
                    if explore == 'ppr':
                        a = self.choose_action_ppr(beg_s, psi)
                    else:
                        raise ValueError
                    end_s, r, t, _, _ = self.env.step(a)
                    # self.update_value(beg_s, end_s, r)
                    self.update_q(beg_s, end_s, a, r)
                    self.update_policy(beg_s)
                    psi = psi * upsilon
                    beg_s = end_s
                    steps += 1
                    dis_r += self.gamma**steps*r
                dis_reward.append(dis_r)
            return dis_reward

    def run_ppr(self, Q_heur, max_step, num_eps, psi, upsilon, explore='ppr'):
            self.Q_heur = Q_heur
            dis_reward = []
            for i in range(num_eps):
                
                beg_s, init_prob = self.env.reset()
                steps = 0
                dis_r = 0
                while steps < max_step:
                    if explore == 'ppr':
                        a = self.choose_action_ppr(beg_s, psi)
                    else:
                        raise ValueError
                    end_s, r, t, _, _ = self.env.step(a)
                    # self.update_value(beg_s, end_s, r)
                    self.update_q(beg_s, end_s, a, r)
                    self.update_policy(beg_s)
                    psi = psi * upsilon
                    beg_s = end_s
                    steps += 1
                    dis_r += self.gamma**steps*r
                dis_reward.append(dis_r)
            return dis_reward





