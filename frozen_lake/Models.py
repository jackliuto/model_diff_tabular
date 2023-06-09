from collections import defaultdict
import random
import numpy as np
import copy

from gym.envs.toy_text.utils import categorical_sample


class DPAgent:
    def __init__(self, env, gamma=1, theta=1e-8):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.S = np.arange(0, self.env.nS).reshape(self.env.nrow, self.env.ncol)

    def policy_evaluation(self, policy, gamma=None, steps=-1) :
        if gamma == None:
            gamma = self.gamma
        V = np.zeros(self.env.nS)
        if steps > -1:
            iterations = 0
        else:
            iterations = -2
        while True and iterations < steps:
            delta = 0
            for s in range(self.env.nS):
                Vs = 0
                for a, action_prob in enumerate(policy[s]):
                    for prob, next_state, reward, done in self.env.P[s][a]:
                        Vs += action_prob * prob * (reward + self.gamma * V[next_state])
                delta = max(delta, np.abs(V[s]-Vs))
                V[s] = Vs
            if delta < self.theta:
                break
            if steps > -1:
                iterations += 1
        return V

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

            if self.env.desc[s//self.env.ncol][s%self.env.ncol] in b"G":
                policy[s] = [0.0, 0.0, 0.0, 0.0, 1.0]


        return policy

    def policy_iteration(self, init_P = []):
        if init_P == []:
            policy = np.ones([self.env.nS, self.env.nA]) / self.env.nA
        else:
            policy = init_P
        steps = 0
        while True:
            V = self.policy_evaluation(policy)
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

    def value_iteration(self, rank_V = [], init_V = [], max_steps=np.inf, det_policy = False):
        if init_V == []: ## used to initialize V
            V = np.zeros(self.env.nS)
        else:
            V = init_V
        if rank_V == []: # used to rank which V to update first
            s_idx_lst = list(range(self.env.nS))
            random.shuffle(s_idx_lst)
        else:
            s_idx_lst = np.argsort(rank_V)[::-1]
        steps = 0
        break_loop = False
        while True:
            delta = 0
            for s in s_idx_lst:
                v = V[s]
                V[s] = max(self.q_from_v(V, s))
                delta = max(delta,abs(V[s]-v))
                steps += 1
                if steps >= max_steps:
                    break_loop = True
                    break

            if delta < self.theta or break_loop:
                break


        policy = self.policy_improvement(V, det_policy)
        return policy, V, steps



class RTDPAgent:
    def __init__(self, env, gamma=0.9, epsilon=0.1, theta=1e-8):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.theta = theta
        self.V = np.zeros([self.env.nS])
        self.Policy = np.ones([self.env.nS, self.env.nA]) / self.env.nA
        self.Vdiff = None
        self.S = np.arange(0, self.env.nS).reshape(self.env.nrow, self.env.ncol)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def q_from_v(self, s):
        q = np.zeros(self.env.nA)
        for a in range(self.env.nA):
            for prob, next_state, reward, done in self.env.P[s][a]:
                q[a] += prob * (reward + self.gamma * self.V[next_state])
        return q

    def choose_action(self, s, V_heur):
        # action_dist = self.Policy[s]
        # if V_heur == []:
        #     if np.random.random() < self.epsilon:
        #          a = self.env.action_space.sample()
        #     else:
        #         a = categorical_sample(action_dist, self.env.np_random)
        # else:
        #     if np.random.random() < self.epsilon:
        #         q = []
        #         for a in range(self.env.nA):
        #             for prob, next_state, reward, done in self.env.P[s][a]:
        #                 q.append(V_heur[next_state]+0.001) #take account zero divison
        #         softmax_q = self.softmax(q)
        #         a = categorical_sample(softmax_q, self.env.np_random)
        #     else:
        #         a = categorical_sample(action_dist, self.env.np_random)
        # return a


        action_dist = self.Policy[s]
        if np.random.random() < self.epsilon:
            if V_heur== []:
                a = self.env.action_space.sample()
            else:
                if np.random.random() < 0.5:
                    a = self.env.action_space.sample()
                else:
                    q = []
                    for a in range(self.env.nA):
                        for prob, next_state, reward, done in self.env.P[s][a]:
                            q.append(V_heur[next_state]+0.01) #take account zero divison
                    a = random.choice(np.argwhere(q==np.max(q)).flatten())
        else:
            a = categorical_sample(action_dist, self.env.np_random)
        return a

    def update_value(self, beg_s, end_s, r):
        self.V[beg_s] = max(self.V[beg_s], r + self.gamma*self.V[end_s])

    def update_policy(self, s):
        q = self.q_from_v(s)
        best_a = np.argwhere(q==np.max(q)).flatten()
        self.Policy[s] = np.sum([np.eye(self.env.nA)[i] for i in best_a], axis=0)/len(best_a)

    def run_eps(self, V_heur=[], max_step = 1000, num_eps=10):
        if V_heur == []:
            V_heur = np.ones(self.env.nS)
        num_steps = []
        for i in range(num_eps):
            beg_s, init_prob = self.env.reset()
            counter = 0
            t = False
            while not t and counter < max_step:
                a = self.choose_action(beg_s, V_heur)
                end_s, r, t, _, _ = self.env.step(a)
                self.update_value(beg_s, end_s, r)
                self.update_policy(beg_s)
                beg_s = end_s
                counter += 1
            num_steps.append(counter)

        return num_steps

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

    def set_policy(self, policy):
        self.Policy = policy.copy()

    # def q_from_v(self, s):
    #     q = np.zeros(self.env.nA)
    #     for a in range(self.env.nA):
    #         for prob, next_state, reward, done in self.env.P[s][a]:
    #             q[a] += prob * (reward + self.gamma * self.V[next_state])
    #     return q

    def choose_action(self, s, Vdiff):
        action_dist = self.Policy[s]
        if np.random.random() < self.epsilon:
            if Vdiff == []:
                a = self.env.action_space.sample()
            else:
                if np.random.random() < 0.5:
                    a = self.env.action_space.sample()
                else:
                    q = []
                    for a in range(self.env.nA):
                        for prob, next_state, reward, done in self.env.P[s][a]:
                            q.append(Vdiff[next_state]+0.01) #take account zero divison
                    a = random.choice(np.argwhere(q==np.max(q)).flatten())

                # action_dist = []
                # for a in range(self.env.nA):
                #     for prob, next_state, reward, done in self.env.P[s][a]:
                #         action_dist.append(Vdiff[next_state]+0.01) #take account zero divison
                # action_dist = action_dist/np.sum(action_dist)
                # a = categorical_sample(action_dist, self.env.np_random)
                # print(action_dist)
        else:
            a = categorical_sample(action_dist, self.env.np_random)
        return a

    def update_value(self, beg_s, end_s, r):
        self.V[beg_s] = max(self.V[beg_s], r + self.gamma*self.V[end_s])

    def update_q(self, beg_s, end_s, a, r):
        self.Q[beg_s][a] += self.alpha * (r + self.gamma * np.max(self.Q[end_s]) - self.Q[beg_s][a])

    def update_policy(self, s):
        q = self.Q[s]
        best_a = np.argwhere(q==np.max(q)).flatten()
        self.Policy[s] = np.sum([np.eye(self.env.nA)[i] for i in best_a], axis=0)/len(best_a)

    def run_eps(self, V_heur=[], max_step = 1000, num_eps=10):
        if V_heur == []:
            V_heur = np.ones(self.env.nS)
        num_steps = []
        for i in range(num_eps):
            beg_s, init_prob = self.env.reset()
            counter = 0
            t = False
            while not t and counter < max_step:
                a = self.choose_action(beg_s, V_heur)
                end_s, r, t, _, _ = self.env.step(a)
                self.update_value(beg_s, end_s, r)
                self.update_q(beg_s, end_s, a, r)
                self.update_policy(beg_s)
                beg_s = end_s
                counter += 1
            num_steps.append(counter)

        return num_steps
