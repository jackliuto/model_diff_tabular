from collections import defaultdict
import numpy as np
import copy

from frozen_lake_5action import FrozenLakeEnv
from utils import plot_matrix, plot_policy_matrix
from pprint import pprint
from Models import DPAgent, RTDPAgent, QLearningAgent


# env1_G = FrozenLakeEnv(is_slippery=False, map_name="20x20_1", terminal_states="GH")
env1_G = FrozenLakeEnv(is_slippery=False, map_name="7x7_S00G77", terminal_states="GH")
env2_G = FrozenLakeEnv(is_slippery=False, map_name="7x7_S00G73", terminal_states="GH")
env1_nG = FrozenLakeEnv(is_slippery=False, map_name="7x7_S00G77", terminal_states="GH")
env2_nG = FrozenLakeEnv(is_slippery=False, map_name="7x7_S00G73", terminal_states="GH")
# env2_nG = FrozenLakeEnv(is_slippery=False, map_name="2x2", terminal_states="H")


DPAgent_1_converge = DPAgent(env1_G, gamma=0.9, theta=1e-6)
DPAgent_2_converge = DPAgent(env2_G, gamma=0.9, theta=1e-6)

DPAgent_1_fixstep = DPAgent(env1_nG, gamma=0.9, theta=1e-6)
DPAgent_2_fixstep = DPAgent(env2_nG, gamma=0.9, theta=1e-6)

policy1_converge, V1_converge, steps1_converge = DPAgent_1_converge.value_iteration()
policy2_converge, V2_converge, steps2_converge = DPAgent_2_converge.value_iteration()


policy1_fixstep, V1_fixstep, steps1_fixstep = DPAgent_1_fixstep.value_iteration(max_steps=1000)
policy2_fixstep, V2_fixstep, steps2_fixstep = DPAgent_2_fixstep.value_iteration(max_steps=1000)

V1_pi_1 = V1_converge
V2_pi_2 = V2_converge
V1_pi_2 = DPAgent_1_converge.policy_evaluation(policy2_converge)
V2_pi_1 = DPAgent_2_converge.policy_evaluation(policy1_converge)

Vdiff = V2_pi_2 - V1_pi_1 
Vdiff_lower = V2_pi_1 - V1_pi_1
Vdiff_upper = V2_pi_2 - V1_pi_2
Vdiff_gap = Vdiff_upper - Vdiff_lower


RTDPAgent_2 = RTDPAgent(env2_nG, gamma=0.9, epsilon=0.4)


# avg = 0
# total_steps, total_rewards = RTDPAgent_2.run_eps(Vdiff_gap, max_step = 50, num_eps = 100)
# print(total_steps, total_rewards, total_rewards/total_steps)

# plot_matrix(RTDPAgent_2.env, Vdiff_gap)
# plot_matrix(RTDPAgent_2.env, RTDPAgent_2.V)
# plot_policy_matrix(RTDPAgent_2.Policy, DPAgent_1_converge.S)

# avg += total_rewards/total_steps
# print(avg*100/(1+1))

QLearningAgent_2 = QLearningAgent(env2_nG, gamma=0.9, epsilon=0.4)
total_steps, total_rewards = QLearningAgent_2.run_eps([], max_step = 50, num_eps = 100)
print(total_steps, total_rewards, total_rewards/total_steps)

# avg = 0
# for i in range(100):
#     QLearningAgent_2 = QLearningAgent(env2_nG, gamma=0.9, epsilon=0.1)
#     total_steps, total_rewards = QLearningAgent_2.run_eps(Vdiff_gap, max_step = 50, num_eps = 100)
#     avg += total_rewards/total_steps
# print(avg/(i+1))

# print(total_steps, total_rewards, total_rewards/total_steps)
# plot_matrix(RTDPAgent_2.env, Vdiff_upper)
# plot_matrix(RTDPAgent_2.env, Vdiff_gap)



# plot_matrix(DPAgent_1_converge.env, V2_converge)

# policy1_converge, V1_converge, steps1_converge = DPAgent_1_converge.value_iteration(rank_V=V2_fixstep)

# print(V1_fixstep)

# DPAgent_1_converge.rank_V(V1_conve
# policy2_converge, V2_converge, steps2_converge = DPAgent_2_converge.value_iteration(V1_fixstep)


# plot_matrix(DPAgent_1_converge.env, V1_converge)
# plot_policy_matrix(policy1_converge, DPAgent_1_converge.S)
# plot_policy_matrix(RTDPAgent_1.Policy, RTDPAgent_1.S)

# plot_matrix(DPAgent_1_fixstep.env, V1_fixstep)
# plot_policy_matrix(policy1_fixstep, DPAgent_1_fixstep.S)
# plot_policy_matrix(RTDPAgent_1.Policy, RTDPAgent_1.S)

# V1_pi_1 = V1_converge
# V2_pi_2 = V2_converge
# V1_pi_2 = DPAgent_1_converge.policy_evaluation(policy2_converge)
# V2_pi_1 = DPAgent_2_converge.policy_evaluation(policy1_converge)

# Vdiff = V2_pi_2 - V1_pi_1 
# Vdiff_lower = V2_pi_1 - V1_pi_1
# Vdiff_upper = V2_pi_2 - V1_pi_2
# Vdiff_gap = Vdiff_upper - Vdiff_lower


# policy2_converge, V2_converge, steps2_converge = DPAgent_2_converge.value_iteration(Vdiff_gap)
# print(steps2_converge)


# p, v = DPAgent_1_converge.policy_iteration()

# plot_policy_matrix(p, DPAgent_1_converge.S)

# plot_matrix(DPAgent_2_converge.env, Vdiff)


# V1_pi_1 = V1
# V2_pi_2 = V2
# V1_pi_2 = DPAgent_1.policy_evaluation(policy2, steps=20)
# V2_pi_1 = DPAgent_2.policy_evaluation(policy1, steps=20)

# Vdiff = V2_pi_2 - V1_pi_1 
# Vdiff_lower = V2_pi_1 - V1_pi_1
# Vdiff_upper = V2_pi_2 - V1_pi_2
# Vdiff_gp = Vdiff_upper - Vdiff_lower

# RTDPAgent_1 = RTDPAgent(env3, gamma=0.9, epsilon=0.1)
# QLearningAgent_1 = QLearningAgent(env3, gamma=0.9, alpha=0.5, epsilon=0.1)

# print(V1)
# print(policy1)

# RTDPAgent.set_policy(RTDPAgent_1, policy1)


# total_steps, total_rewards = RTDPAgent_1.run_eps(max_step = 50, num_eps = 50)
# print(total_rewards)

# total_steps, total_rewards = QLearningAgent_1.run_eps(max_step = 50, num_eps = 50)
# print(total_rewards)

# print(RTDPAgent_1.V)
# print(RTDPAgent_1.Policy)

# plot_matrix(RTDPAgent_1.V.reshape(env1.nrow,env1.ncol))
# plot_policy_matrix(RTDPAgent_1.Policy, RTDPAgent_1.S)

# print(RTDPAgent_1.Policy)













