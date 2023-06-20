from collections import defaultdict
import numpy as np
import copy

from Gridworld import GridWorldEnv
from utils import plot_matrix, plot_policy_matrix, plot_line_dict, plot_Qdiff_matrix
from pprint import pprint
from Models import DPAgent, QLearningAgent


RANDOM_START = False
GAMMA = 0.9
THETA = 1e-6

env_1 = GridWorldEnv(is_slippery=False, map_name="7x7_S00G77", random_start=RANDOM_START)
env_2 = GridWorldEnv(is_slippery=False, map_name="7x7_S00G7377", random_start=RANDOM_START)

env_diff_21 = GridWorldEnv(is_slippery=False, map_name="7x7_S00G7377", random_start=RANDOM_START, reward_matrix=env_2.reward_matrix.copy()-env_1.reward_matrix.copy())

# env_diff_31 = GridWorldEnv(is_slippery=False, map_name="7x7_S00G66", random_start=RANDOM_START, reward_matrix=env_3.reward_matrix-env_1.reward_matrix)
# env_diff_41 = GridWorldEnv(is_slippery=False, map_name="7x7_S77G00", random_start=RANDOM_START, reward_matrix=env_4.reward_matrix-env_1.reward_matrix)

DPAgent_1 = DPAgent(env_1, gamma=GAMMA, theta=THETA)
DPAgent_2 = DPAgent(env_2, gamma=GAMMA, theta=THETA)
# DPAgent_3 = DPAgent(env_3, gamma=GAMMA, theta=THETA)
# DPAgent_4 = DPAgent(env_4, gamma=GAMMA, theta=THETA)

DPAgent_diff_21 = DPAgent(env_diff_21, gamma=GAMMA, theta=THETA)
# DPAgent_diff_31 = DPAgent(env_diff_31, gamma=GAMMA, theta=THETA)
# DPAgent_diff_41 = DPAgent(env_diff_41, gamma=GAMMA, theta=THETA)




policy1_converge, V1_converge, Q1_converge, steps1_converge, iter1_converge = DPAgent_1.value_iteration()

print(V1_converge)

policy2_converge, V2_converge, Q2_converge, steps2_converge, iter2_converge = DPAgent_2.value_iteration()
policy21_converge, V21_converge, Q21_converge, steps21_converge, iter21_converge = DPAgent_diff_21.value_iteration()
print(iter1_converge, iter21_converge, iter21_converge)

# policy3_converge, V3_converge, Q3_converge, steps3_converge, iter3_converge = DPAgent_3.value_iteration()
# policy4_converge, V4_converge, Q4_converge, steps4_converge, iter4_converge = DPAgent_4.value_iteration()

_, _, n_diff = DPAgent_diff_21.policy_evaluation(policy1_converge, max_iter=np.inf)
V_PE_21, _, n_2 = DPAgent_2.policy_evaluation(policy1_converge, max_iter=np.inf)

# print(V_PE_21.reshape(7,7))
# print(V2_converge.reshape(7,7))
# print(V1_converge.reshape(7,7))

print(n_diff, n_2)

V_zero = np.zeros(49)


# print(sum(abs(V2_converge - V1_converge)))
# print(sum(abs(V_PE_21- V1_converge)))

_, _, _, _, iter2_warmstart1 = DPAgent_2.value_iteration_warmstart(init_V=V1_converge)
_, _, _, _, iter2_warmstartPE21 = DPAgent_2.value_iteration_warmstart(init_V=V_PE_21)
_, _, _, _, iter2_warmstartZero = DPAgent_2.value_iteration_warmstart(init_V=V_zero)




print(iter2_warmstart1)
print(iter2_warmstartPE21)
print(iter2_warmstartZero)



raise ValueError