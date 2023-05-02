from collections import defaultdict
import numpy as np
import copy

from gym.envs.toy_text.utils import categorical_sample

class DPAgent:
    def __init__(self, env, gamma=1, theta=1e-8):
        self.env = env
        self.gamma = gamma
        self.theta = theta

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

    def policy_improvement(self, V, deterministic=False):
        policy = np.zeros([self.env.nS, self.env.nA]) / self.env.nA
        for s in range(self.env.nS):
            q = self.q_from_v(V, s)
            
            # OPTION 1: construct a deterministic policy 
            # policy[s][np.argmax(q)] = 1
            
            # OPTION 2: construct a stochastic policy that puts equal probability on maximizing actions
            # best_a = np.argwhere(q==np.max(q)).flatten()
            # policy[s] = np.sum([np.eye(self.env.nA)[i] for i in best_a], axis=0)/len(best_a)
            
            if deterministic:
                policy[s][np.argmax(q)] = 1
            else:
                best_a = np.argwhere(q==np.max(q)).flatten()
                policy[s] = np.sum([np.eye(self.env.nA)[i] for i in best_a], axis=0)/len(best_a)

            if self.env.desc[s//self.env.ncol][s%self.env.ncol] in b"G":
                policy[s] = [0.0, 0.0, 0.0, 0.0, 1.0]


        return policy

    def policy_iteration(self):
        policy = np.ones([self.env.nS, self.env.nA]) / self.env.nA
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
        return policy, V

    def value_iteration(self, steps=-1, det_policy = False):
        V = np.zeros(self.env.nS)
        if steps > -1:
            iterations = 0
        else:
            iterations = -2
        while True and iterations < steps:
            delta = 0
            for s in range(self.env.nS):
                v = V[s]
                V[s] = max(self.q_from_v(V, s))
                delta = max(delta,abs(V[s]-v))
            if delta < self.theta:
                break
            if steps > -1:
                iterations += 1
        policy = self.policy_improvement(V, det_policy)
        
        return policy, V

class RTDPAgent:
    def __init__(self, env, gamma=0.9, epsilon=0.1, theta=1e-8):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.theta = theta
        self.V = np.zeros([self.env.nS])
        self.Policy = np.ones([self.env.nS, self.env.nA]) / self.env.nA
        self.Vdiff = None
    
    def set_policy(self, policy):
        self.Policy = policy.copy()

    def q_from_v(self, s):
        q = np.zeros(self.env.nA)
        for a in range(self.env.nA):
            for prob, next_state, reward, done in self.env.P[s][a]:
                q[a] += prob * (reward + self.gamma * self.V[next_state])
        return q

    def choose_action(self, s):
        action_dist = self.Policy[s]
        if np.random.random() < self.epsilon:
            a = self.env.action_space.sample()
        else:
            a = categorical_sample(action_dist, self.env.np_random)
        return a
    
    def update_value(self, beg_s, end_s, r):
        self.V[beg_s] = max(self.V[beg_s], r + self.gamma*self.V[end_s])
    
    def update_policy(self, s):
        q = self.q_from_v(s)
        best_a = np.argwhere(q==np.max(q)).flatten()
        self.Policy[s] = np.sum([np.eye(self.env.nA)[i] for i in best_a], axis=0)/len(best_a)

    def run_eps(self, max_step = 1000, num_eps=10):
        
        total_steps = 0
        total_rewards = 0 
        for i in range(num_eps):
            beg_s, init_prob = self.env.reset()
            counter = 0
            t = False
            while not t and counter < max_step:
                a = self.choose_action(beg_s)
                end_s, r, t, _, _ = self.env.step(a)
                self.update_value(beg_s, end_s, r)
                self.update_policy(beg_s)
                beg_s = end_s
                total_steps += 1
                counter += 1
                total_rewards += r
        
        return total_steps, total_rewards
    
class QLearningAgent:
    def __init__(self, env, gamma=0.9, alpha=0.5, epsilon=0.1, theta=1e-8):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.theta = theta
        # self.V = np.zeros([self.env.nS])
        self.Q = np.zeros([self.env.nS, self.env.nA])
        self.Policy = np.ones([self.env.nS, self.env.nA]) / self.env.nA
        self.Vdiff = None
    
    def set_policy(self, policy):
        self.Policy = policy.copy()

    # def q_from_v(self, s):
    #     q = np.zeros(self.env.nA)
    #     for a in range(self.env.nA):
    #         for prob, next_state, reward, done in self.env.P[s][a]:
    #             q[a] += prob * (reward + self.gamma * self.V[next_state])
    #     return q

    def choose_action(self, s):
        action_dist = self.Policy[s]
        if np.random.random() < self.epsilon:
            a = self.env.action_space.sample()
        else:
            a = categorical_sample(action_dist, self.env.np_random)
        return a
    
    # def update_value(self, beg_s, end_s, r):
    #     self.V[beg_s] = max(self.V[beg_s], r + self.gamma*self.V[end_s])

    def update_q(self, beg_s, end_s, a, r):
        self.Q[beg_s][a] += self.alpha * (r + self.gamma * np.max(self.Q[end_s]) - self.Q[beg_s][a])
    
    def update_policy(self, s):
        q = self.Q[s]
        best_a = np.argwhere(q==np.max(q)).flatten()
        self.Policy[s] = np.sum([np.eye(self.env.nA)[i] for i in best_a], axis=0)/len(best_a)

    def run_eps(self, max_step = 1000, num_eps=10):
        
        total_steps = 0
        total_rewards = 0 
        for i in range(num_eps):
            beg_s, init_prob = self.env.reset()
            counter = 0
            t = False
            while not t and counter < max_step:
                a = self.choose_action(beg_s)
                end_s, r, t, _, _ = self.env.step(a)
                # self.update_value(beg_s, end_s, r)
                self.update_q(beg_s, end_s, a, r)
                self.update_policy(beg_s)
                beg_s = end_s
                total_steps += 1
                counter += 1
                total_rewards += r
        
        return total_steps, total_rewards
