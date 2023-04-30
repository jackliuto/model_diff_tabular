from collections import defaultdict

import numpy as np

import copy

from frozen_lake_5action import FrozenLakeEnv

from utils import plot_matrix

from pprint import pprint

from Models import DPAgent


env1 = FrozenLakeEnv(is_slippery=False, map_name="7x7_1")

DPAgent_1 = DPAgent(env1, gamma=0.9, theta=1e-8)

policy1, V1 = DPAgent_1.value_iteration()


plot_matrix(V1.reshape(env1.nrow,env1.ncol))



# env1 = FrozenLakeEnv(is_slippery=False, map_name="7x7_1")
# env2 = FrozenLakeEnv(is_slippery=False, map_name="7x7_2")
# env3 = FrozenLakeEnv(is_slippery=False, map_name="5x5_wall")
# env4 = FrozenLakeEnv(is_slippery=False, map_name="2x2",terminal_states="H")


# policy1, V1 = value_iteration(env1, gamma=0.9)
# policy2, V2 = value_iteration(env2, gamma=0.9)
# policy3, V3 = value_iteration(env3, gamma=0.9)
# policy4, V4 = value_iteration(env4, gamma=1)

# print(policy3)
# plot_matrix(V3.reshape(env3.nrow,env3.ncol))


# V1_pi_1 = V1
# V2_pi_2 = V2
# V1_pi_2 = policy_evaluation(env1, policy2, gamma=0.9)
# V2_pi_1 = policy_evaluation(env2, policy1, gamma=0.9)

# Vdiff = V2_pi_2 - V1_pi_1 
# Vdiff_lower = V2_pi_1 - V1_pi_1
# Vdiff_upper = V2_pi_2 - V1_pi_2

# print(policy1)

# plot_matrix((Vdiff_upper - Vdiff_lower).reshape(env1.nrow,env1.ncol))
# plot_matrix(V2.reshape(env1.nrow,env1.ncol))

# plot_matrix(Vdiff.reshape(env1.nrow,env1.ncol))


# print(V2_pi_1)
# print(V1_pi_1)










