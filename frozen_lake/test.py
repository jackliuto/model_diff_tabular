from collections import defaultdict

import numpy as np

import copy

from frozen_lake_5action import FrozenLakeEnv

from plot_utils import plot_matrix

from pprint import pprint

def policy_evaluation(env, policy, gamma=1, theta=1e-8):
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

def q_from_v(env, V, s, gamma=1):
    q = np.zeros(env.nA)
    for a in range(env.nA):
        for prob, next_state, reward, done in env.P[s][a]:
            q[a] += prob * (reward + gamma * V[next_state])
    return q

def policy_improvement(env, V, gamma=1):
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
        V = policy_evaluation(env, policy, gamma, theta)
        new_policy = policy_improvement(env, V)
        
        # OPTION 1: stop if the policy is unchanged after an improvement step
        if (new_policy == policy).all():
            break;
        
        # OPTION 2: stop if the value function estimates for successive policies has converged
        # if np.max(abs(policy_evaluation(env, policy) - policy_evaluation(env, new_policy))) < theta*1e2:
        #    break;
        
        policy = copy.copy(new_policy)
    return policy, V

def value_iteration(env, gamma=1, theta=1e-8):
    V = np.zeros(env.nS)
    # for i in range(10):
    n = 0
    while True:
        delta = 0
        for s in range(env.nS):
            v = V[s]
            V[s] = max(q_from_v(env, V, s, gamma))
            delta = max(delta,abs(V[s]-v))
        n += 1
        if delta < theta:
            print(n)
            break
    policy = policy_improvement(env, V, gamma)
    return policy, V


env1 = FrozenLakeEnv(is_slippery=False, map_name="7x7_1")
env2 = FrozenLakeEnv(is_slippery=False, map_name="7x7_2")
env3 = FrozenLakeEnv(is_slippery=False, map_name="5x5_wall")

# pprint(env.P)

policy1, V1 = value_iteration(env1, gamma=0.9)
policy2, V2 = value_iteration(env2, gamma=0.9)
policy3, V3 = value_iteration(env3, gamma=0.9)

# print(policy3)
plot_matrix(V3.reshape(env3.nrow,env3.ncol))


V1_pi_1 = V1
V2_pi_2 = V2
V1_pi_2 = policy_evaluation(env1, policy2, gamma=0.9)
V2_pi_1 = policy_evaluation(env2, policy1, gamma=0.9)

Vdiff = V2_pi_2 - V1_pi_1 
Vdiff_lower = V2_pi_1 - V1_pi_1
Vdiff_upper = V2_pi_2 - V1_pi_2

# plot_matrix((Vdiff_upper - Vdiff_lower).reshape(env1.nrow,env1.ncol))
# plot_matrix(V2.reshape(env1.nrow,env1.ncol))

# plot_matrix(Vdiff.reshape(env1.nrow,env1.ncol))


# print(V2_pi_1)
# print(V1_pi_1)










