from collections import defaultdict
import numpy as np
import copy
import json

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt  

from Gridworld import GridWorldEnv
from utils import plot_matrix, plot_policy_matrix, plot_line_dict, plot_Qdiff_matrix
from pprint import pprint
from Models import DPAgent, QLearningAgent

# np.random.seed(0)


RANDOM_START = False
GAMMA = 0.9
THETA = 1e-6

env_1 = GridWorldEnv(is_slippery=False, map_name="7x7_S00G77", random_start=RANDOM_START)
env_2 = GridWorldEnv(is_slippery=False, map_name="7x7_S00G73", random_start=RANDOM_START)
env_3 = GridWorldEnv(is_slippery=False, map_name="7x7_S00G66", random_start=RANDOM_START)
env_4 = GridWorldEnv(is_slippery=False, map_name="7x7_S77G00", random_start=RANDOM_START)

env_diff_21 = GridWorldEnv(is_slippery=False, map_name="7x7_S00G73", random_start=RANDOM_START, reward_matrix=env_2.reward_matrix-env_1.reward_matrix)
env_diff_31 = GridWorldEnv(is_slippery=False, map_name="7x7_S00G66", random_start=RANDOM_START, reward_matrix=env_3.reward_matrix-env_1.reward_matrix)
env_diff_41 = GridWorldEnv(is_slippery=False, map_name="7x7_S77G00", random_start=RANDOM_START, reward_matrix=env_4.reward_matrix-env_1.reward_matrix)

DPAgent_1 = DPAgent(env_1, gamma=GAMMA, theta=THETA)
DPAgent_2 = DPAgent(env_2, gamma=GAMMA, theta=THETA)
DPAgent_3 = DPAgent(env_3, gamma=GAMMA, theta=THETA)
DPAgent_4 = DPAgent(env_4, gamma=GAMMA, theta=THETA)

DPAgent_diff_21 = DPAgent(env_diff_21, gamma=GAMMA, theta=THETA)
DPAgent_diff_31 = DPAgent(env_diff_31, gamma=GAMMA, theta=THETA)
DPAgent_diff_41 = DPAgent(env_diff_41, gamma=GAMMA, theta=THETA)


policy1_converge, V1_converge, Q1_converge, steps1_converge, iter1_converge = DPAgent_1.value_iteration()
policy2_converge, V2_converge, Q2_converge, steps2_converge, iter2_converge = DPAgent_2.value_iteration()
policy3_converge, V3_converge, Q3_converge, steps3_converge, iter3_converge = DPAgent_3.value_iteration()
policy4_converge, V4_converge, Q4_converge, steps4_converge, iter4_converge = DPAgent_4.value_iteration()

V1_pi_1_c, Q1_pi_1_c = V1_converge, Q1_converge
V2_pi_2_c, Q2_pi_2_c = V2_converge, Q2_converge
V1_pi_2_c, Q1_pi_2_c, _ = DPAgent_1.policy_evaluation(policy2_converge)
V2_pi_1_c, Q2_pi_1_c, _ = DPAgent_2.policy_evaluation(policy1_converge)

V1_pi_1_c, Q1_pi_1_c = V1_converge, Q1_converge
V3_pi_3_c, Q3_pi_3_c = V3_converge, Q3_converge
V1_pi_3_c, Q1_pi_3_c, _ = DPAgent_1.policy_evaluation(policy3_converge)
V3_pi_1_c, Q3_pi_1_c, _ = DPAgent_3.policy_evaluation(policy1_converge)

V1_pi_1_c, Q1_pi_1_c = V1_converge, Q1_converge
V4_pi_4_c, Q4_pi_4_c = V4_converge, Q4_converge
V1_pi_4_c, Q1_pi_4_c, _ = DPAgent_1.policy_evaluation(policy4_converge)
V4_pi_1_c, Q4_pi_1_c, _ = DPAgent_4.policy_evaluation(policy1_converge)

np.set_printoptions(precision=2)

### Vdiff bounds
Vdiff_2_LB_c = V2_pi_1_c - V1_pi_1_c
Vdiff_2_OP_c = V2_pi_2_c - V1_pi_1_c
Vdiff_2_UP_c = V2_pi_2_c - V1_pi_2_c

Vdiff_3_LB_c = V3_pi_1_c - V1_pi_1_c
Vdiff_3_OP_c = V3_pi_3_c - V1_pi_1_c
Vdiff_3_UP_c = V3_pi_3_c - V1_pi_3_c

Vdiff_4_LB_c = V4_pi_1_c - V1_pi_1_c
Vdiff_4_OP_c = V4_pi_4_c - V1_pi_1_c
Vdiff_4_UP_c = V4_pi_4_c - V1_pi_4_c

### V bounds
V_2_LB_c = Vdiff_2_LB_c + V1_pi_1_c
V_2_OP_c = Vdiff_2_OP_c + V1_pi_1_c
V_2_UP_c = Vdiff_2_UP_c + V1_pi_1_c

V_3_LB_c = Vdiff_3_LB_c + V1_pi_1_c
V_3_OP_c = Vdiff_3_OP_c + V1_pi_1_c
V_3_UP_c = Vdiff_3_UP_c + V1_pi_1_c

V_4_LB_c = Vdiff_4_LB_c + V1_pi_1_c
V_4_OP_c = Vdiff_4_OP_c + V1_pi_1_c
V_4_UP_c = Vdiff_4_UP_c + V1_pi_1_c

def cal_smooth(X,Y,V):
    shape = (X.shape[0], Y.shape[1])
    Z = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            x = X[i][j]
            y = Y[i][j]
            v = V[int(x)][int(y)]
            v_plus_x = V[min(int(x+1), V.shape[0]-1)][int(y)]
            v_plus_y = V[int(x)][min(int(y+1), V.shape[1]-1)]
            v_linear = v + (x - int(x))*(v_plus_x - v) + (y - int(y))*(v_plus_y - y)
            Z[i][j] = v_linear
          
    return Z


def plot_bounds(lower, optimal, upper, shape=(7,7)):

    lower_mat = lower.reshape(shape)
    optimal_mat = optimal.reshape(shape)
    upper_mat = upper.reshape(shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(0, shape[0],1)
    y = np.arange(0, shape[1],1)
    X, Y = np.meshgrid(x, y)

    ax.plot_surface(X, Y, lower_mat, label='Vdiff_Lower')
    ax.plot_surface(X, Y, optimal_mat, label='Vdiff_Optimal')
    ax.plot_surface(X, Y, upper_mat, label='Vdiff_Upper')


    ax.set_xlabel('X coord')
    ax.set_ylabel('Y coord')
    ax.set_zlabel('Value')
    

    plt.title('Vdiff Bound Viz')

    plt.show()

    return 

# print(V2_pi_1_c.reshape(7,7))
# raise ValueError

plot_bounds(Vdiff_2_LB_c, Vdiff_2_OP_c, Vdiff_2_UP_c)

raise ValueError 

