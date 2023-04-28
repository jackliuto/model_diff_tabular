from collections import defaultdict
import numpy as np
import copy

class DPAgent:
    def __init__(self, env, gamma=1, theta=1e-8):
        self.env = env
        self.gamma = gamma
        self.theta = theta

    def policy_evaluation(self, env=self.env, gamma=self.gamma, theta=self.theta, policy) :
        V = np.zeros(env.nS)
        while True:
            delta = 0
            for s in range(env.nS):
                Vs = 0
                for a, action_prob in enumerate(policy[s]):
                    for prob, next_state, reward, done in env.P[s][a]:
                        Vs += action_prob * prob * (reward + gamma * V[next_state])
                delta = max(delta, np.abs(V[s]-Vs))
                V[s] = Vs
            if delta < theta:
                break
        return V

    def q_from_v(self, env=self.env, gamma=self.gamma, V, s):
        q = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[s][a]:
                q[a] += prob * (reward + gamma * V[next_state])
        return q

    def policy_improvement(self, env=self.env, gamma=self.gamma, v):
        policy = np.zeros([env.nS, env.nA]) / env.nA
        for s in range(env.nS):
            q = q_from_v(env, V, s, gamma)
            
            # OPTION 1: construct a deterministic policy 
            # policy[s][np.argmax(q)] = 1
            
            # OPTION 2: construct a stochastic policy that puts equal probability on maximizing actions
            best_a = np.argwhere(q==np.max(q)).flatten()
            policy[s] = np.sum([np.eye(env.nA)[i] for i in best_a], axis=0)/len(best_a)
            
        return policy

    def policy_iteration(env, gamma=1, theta=1e-8):
        policy = np.ones([env.nS, env.nA]) / env.nA
        while True:
            V = policy_evaluation(policy)
            new_policy = policy_improvement(V)
            
            # OPTION 1: stop if the policy is unchanged after an improvement step
            if (new_policy == policy).all():
                break
            
            # OPTION 2: stop if the value function estimates for successive policies has converged
            # if np.max(abs(policy_evaluation(env, policy) - policy_evaluation(env, new_policy))) < theta*1e2:
            #    break;
            
            policy = copy.copy(new_policy)
        return policy, V

    def value_iteration(env, gamma=1, theta=1e-8):
        V = np.zeros(env.nS)
        for i in range(10):
        # while True:
            delta = 0
            for s in range(env.nS):
                v = V[s]
                V[s] = max(q_from_v(env, V, s, gamma))
                delta = max(delta,abs(V[s]-v))
            if delta < theta:
                break
        policy = policy_improvement(env, V, gamma)
        return policy, V

