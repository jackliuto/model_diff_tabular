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
env3_G = FrozenLakeEnv(is_slippery=False, map_name="7x7_S00G66", terminal_states="GH")
env4_G = FrozenLakeEnv(is_slippery=False, map_name="7x7_S77G00", terminal_states="GH")
env5_G = FrozenLakeEnv(is_slippery=False, map_name="20x20_S00G1919", terminal_states="GH")
env6_G = FrozenLakeEnv(is_slippery=False, map_name="20x20_S00G1515", terminal_states="GH")

DPAgent_1_converge = DPAgent(env1_G, gamma=0.9, theta=1e-6)
DPAgent_2_converge = DPAgent(env2_G, gamma=0.9, theta=1e-6)
DPAgent_3_converge = DPAgent(env3_G, gamma=0.9, theta=1e-6)
DPAgent_4_converge = DPAgent(env4_G, gamma=0.9, theta=1e-6)
DPAgent_5_converge = DPAgent(env5_G, gamma=0.9, theta=1e-6)
DPAgent_6_converge = DPAgent(env6_G, gamma=0.9, theta=1e-6)

policy1_converge, V1_converge, steps1_converge = DPAgent_1_converge.value_iteration()
policy2_converge, V2_converge, steps2_converge = DPAgent_2_converge.value_iteration()
policy3_converge, V3_converge, steps3_converge = DPAgent_3_converge.value_iteration()
policy4_converge, V4_converge, steps4_converge = DPAgent_4_converge.value_iteration()
policy5_converge, V5_converge, steps5_converge = DPAgent_5_converge.value_iteration()
policy6_converge, V6_converge, steps6_converge = DPAgent_6_converge.value_iteration()

V1_pi_1 = V1_converge
V2_pi_2 = V2_converge
V1_pi_2 = DPAgent_1_converge.policy_evaluation(policy2_converge)
V2_pi_1 = DPAgent_2_converge.policy_evaluation(policy1_converge)

Vdiff12 = V2_pi_2 - V1_pi_1 
Vdiff12_lower = V2_pi_1 - V1_pi_1
Vdiff12_upper = V2_pi_2 - V1_pi_2
Vdiff12_gap = Vdiff12_upper - Vdiff12_lower

V1_pi_1 = V1_converge
V3_pi_3 = V3_converge
V1_pi_3 = DPAgent_1_converge.policy_evaluation(policy3_converge)
V3_pi_1 = DPAgent_3_converge.policy_evaluation(policy1_converge)

Vdiff13 = V3_pi_3 - V1_pi_1 
Vdiff13_lower = V3_pi_1 - V1_pi_1
Vdiff13_upper = V3_pi_3 - V1_pi_3
Vdiff13_gap = Vdiff13_upper - Vdiff13_lower

V1_pi_1 = V1_converge
V4_pi_4 = V4_converge
V1_pi_4 = DPAgent_1_converge.policy_evaluation(policy4_converge)
V4_pi_1 = DPAgent_4_converge.policy_evaluation(policy1_converge)

Vdiff14 = V4_pi_4 - V1_pi_1 
Vdiff14_lower = V4_pi_1 - V1_pi_1
Vdiff14_upper = V4_pi_4 - V1_pi_4
Vdiff14_gap = Vdiff14_upper - Vdiff14_lower

# # VI Methods
# _, _, steps2_converge = DPAgent_2_converge.value_iteration()
# _, _, steps2_CS1 = DPAgent_2_converge.value_iteration(init_V=V1_converge)
# _, _, steps2_Vdiffgap = DPAgent_2_converge.value_iteration(rank_V=Vdiff12_gap)
# _, _, steps2_Vdiffupper = DPAgent_2_converge.value_iteration(rank_V=Vdiff12_upper)
# _, _, steps2_Vdifflower = DPAgent_2_converge.value_iteration(rank_V=Vdiff12_lower)
# _, _, steps2_Vdiffgap_CS1 = DPAgent_2_converge.value_iteration(init_V=V1_converge, rank_V=Vdiff12_gap)

# print("Random: {}, Cold Start: {}, Gap: {}, Upper: {}, Lower: {}, CS+GAP: {}".format(steps2_converge, \
# steps2_CS1, steps2_Vdiffgap, steps2_Vdiffupper, steps2_Vdifflower, steps2_Vdiffgap_CS1))

# # VI Methods
# _, _, steps3_converge = DPAgent_3_converge.value_iteration()
# _, _, steps3_CS1 = DPAgent_3_converge.value_iteration(init_V=V1_converge)
# _, _, steps3_Vdiffgap = DPAgent_3_converge.value_iteration(rank_V=Vdiff13_gap)
# _, _, steps3_Vdiffupper = DPAgent_3_converge.value_iteration(rank_V=Vdiff13_upper)
# _, _, steps3_Vdifflower = DPAgent_3_converge.value_iteration(rank_V=Vdiff13_lower)
# _, _, steps3_Vdiffgap_CS1 = DPAgent_3_converge.value_iteration(init_V=V1_converge, rank_V=Vdiff13_gap)

# print("Random: {}, Cold Start: {}, Gap: {}, Upper: {}, Lower: {}, CS+GAP: {}".format(steps3_converge, \
# steps3_CS1, steps3_Vdiffgap, steps3_Vdiffupper, steps3_Vdifflower, steps3_Vdiffgap_CS1))

# # VI Methods
# _, _, steps4_converge = DPAgent_4_converge.value_iteration()
# _, _, steps4_CS1 = DPAgent_4_converge.value_iteration(init_V=V1_converge)
# _, _, steps4_Vdiffgap = DPAgent_4_converge.value_iteration(rank_V=Vdiff14_gap)
# _, _, steps4_Vdiffupper = DPAgent_4_converge.value_iteration(rank_V=Vdiff14_upper)
# _, _, steps4_Vdifflower = DPAgent_4_converge.value_iteration(rank_V=Vdiff14_lower)
# _, _, steps4_Vdiffgap_CS1 = DPAgent_4_converge.value_iteration(init_V=V1_converge, rank_V=Vdiff14_gap)

# print("Random: {}, Cold Start: {}, Gap: {}, Upper: {}, Lower: {}, CS+GAP: {}".format(steps4_converge, \
# steps4_CS1, steps4_Vdiffgap, steps4_Vdiffupper, steps4_Vdifflower, steps4_Vdiffgap_CS1))

# PI Methods
_, _, steps5_converge = DPAgent_5_converge.policy_iteration(init_P=[])

print(steps5_converge)
# _, _, steps2_CS1 = DPAgent_2_converge.value_iteration(init_V=V1_converge)
# _, _, steps2_Vdiffgap = DPAgent_2_converge.value_iteration(rank_V=Vdiff12_gap)
# _, _, steps2_Vdiffupper = DPAgent_2_converge.value_iteration(rank_V=Vdiff12_upper)
# _, _, steps2_Vdifflower = DPAgent_2_converge.value_iteration(rank_V=Vdiff12_lower)
# _, _, steps2_Vdiffgap_CS1 = DPAgent_2_converge.value_iteration(init_V=V1_converge, rank_V=Vdiff12_gap)

# print("Random: {}, Cold Start: {}, Gap: {}, Upper: {}, Lower: {}, CS+GAP: {}".format(steps2_converge, \
# steps2_CS1, steps2_Vdiffgap, steps2_Vdiffupper, steps2_Vdifflower, steps2_Vdiffgap_CS1))



# ## RTDP Methods
# total_steps=0,
# total_rewards=0,
# for i in range(10):
#     RTDPAgent_2 = RTDPAgent(env2_G, gamma=0.9, epsilon=0.2)
#     total_steps, total_rewards = RTDPAgent_2.run_eps([], max_step = 50, num_eps = 100)
#     total_steps += total_steps
#     total_rewards += total_rewards
# print("Avg Random: {}".format(total_rewards/total_steps))

# total_steps=0,
# total_rewards=0,
# for i in range(10):
#     RTDPAgent_2 = RTDPAgent(env2_G, gamma=0.9, epsilon=0.2)
#     total_steps, total_rewards = RTDPAgent_2.run_eps(Vdiff12_gap, max_step = 50, num_eps = 100)
#     total_steps += total_steps
#     total_rewards += total_rewards
# print("Avg Gap: {}".format(total_rewards/total_steps))

# total_steps=0,
# total_rewards=0,
# for i in range(10):
#     RTDPAgent_2 = RTDPAgent(env2_G, gamma=0.9, epsilon=0.2)
#     total_steps, total_rewards = RTDPAgent_2.run_eps(Vdiff12_upper, max_step = 50, num_eps = 100)
#     total_steps += total_steps
#     total_rewards += total_rewards
# print("Avg Upper: {}".format(total_rewards/total_steps))

# total_steps=0,
# total_rewards=0,
# for i in range(10):
#     RTDPAgent_2 = RTDPAgent(env2_G, gamma=0.9, epsilon=0.2)
#     total_steps, total_rewards = RTDPAgent_2.run_eps(Vdiff12_lower, max_step = 50, num_eps = 100)
#     total_steps += total_steps
#     total_rewards += total_rewards
# print("Avg Lower: {}".format(total_rewards/total_steps))

# total_steps=0,
# total_rewards=0,
# for i in range(10):
#     RTDPAgent_2 = RTDPAgent(env2_G, gamma=0.9, epsilon=0.2)
#     RTDPAgent_2.Policy = policy1_converge 
#     total_steps, total_rewards = RTDPAgent_2.run_eps([], max_step = 50, num_eps = 100)
#     total_steps += total_steps
#     total_rewards += total_rewards
# print("Avg CS: {}".format(total_rewards/total_steps))

# total_steps=0,
# total_rewards=0,
# for i in range(10):
#     RTDPAgent_2 = RTDPAgent(env2_G, gamma=0.9, epsilon=0.2)
#     RTDPAgent_2.Policy = policy1_converge 
#     total_steps, total_rewards = RTDPAgent_2.run_eps(Vdiff12_gap, max_step = 50, num_eps = 100)
#     total_steps += total_steps
#     total_rewards += total_rewards
# print("Avg CS + Gap: {}".format(total_rewards/total_steps))

# ## RTDP Methods
# total_steps=0,
# total_rewards=0,
# for i in range(10):
#     RTDPAgent_3 = RTDPAgent(env3_G, gamma=0.9, epsilon=0.2)
#     total_steps, total_rewards = RTDPAgent_3.run_eps([], max_step = 50, num_eps = 100)
#     total_steps += total_steps
#     total_rewards += total_rewards
# print("Avg Random: {}".format(total_rewards/total_steps))

# total_steps=0,
# total_rewards=0,
# for i in range(10):
#     RTDPAgent_3 = RTDPAgent(env3_G, gamma=0.9, epsilon=0.2)
#     total_steps, total_rewards = RTDPAgent_3.run_eps(Vdiff13_gap, max_step = 50, num_eps = 100)
#     total_steps += total_steps
#     total_rewards += total_rewards
# print("Avg Gap: {}".format(total_rewards/total_steps))

# total_steps=0,
# total_rewards=0,
# for i in range(10):
#     RTDPAgent_3 = RTDPAgent(env3_G, gamma=0.9, epsilon=0.2)
#     total_steps, total_rewards = RTDPAgent_3.run_eps(Vdiff13_upper, max_step = 50, num_eps = 100)
#     total_steps += total_steps
#     total_rewards += total_rewards
# print("Avg Upper: {}".format(total_rewards/total_steps))

# total_steps=0,
# total_rewards=0,
# for i in range(10):
#     RTDPAgent_3 = RTDPAgent(env3_G, gamma=0.9, epsilon=0.2)
#     total_steps, total_rewards = RTDPAgent_3.run_eps(Vdiff13_lower, max_step = 50, num_eps = 100)
#     total_steps += total_steps
#     total_rewards += total_rewards
# print("Avg Lower: {}".format(total_rewards/total_steps))

# total_steps=0,
# total_rewards=0,
# for i in range(10):
#     RTDPAgent_3 = RTDPAgent(env3_G, gamma=0.9, epsilon=0.2)
#     RTDPAgent_3.Policy = policy1_converge 
#     total_steps, total_rewards = RTDPAgent_3.run_eps([], max_step = 50, num_eps = 100)
#     total_steps += total_steps
#     total_rewards += total_rewards
# print("Avg CS: {}".format(total_rewards/total_steps))

# total_steps=0,
# total_rewards=0,
# for i in range(10):
#     RTDPAgent_3 = RTDPAgent(env3_G, gamma=0.9, epsilon=0.2)
#     RTDPAgent_3.Policy = policy1_converge 
#     total_steps, total_rewards = RTDPAgent_3.run_eps(Vdiff13_gap, max_step = 50, num_eps = 100)
#     total_steps += total_steps
#     total_rewards += total_rewards
# print("Avg CS + Gap: {}".format(total_rewards/total_steps))

# ## RTDP Methods
# total_steps=0,
# total_rewards=0,
# for i in range(10):
#     RTDPAgent_4 = RTDPAgent(env4_G, gamma=0.9, epsilon=0.2)
#     total_steps, total_rewards = RTDPAgent_4.run_eps([], max_step = 50, num_eps = 100)
#     total_steps += total_steps
#     total_rewards += total_rewards
# print("Avg Random: {}".format(total_rewards/total_steps))

# total_steps=0,
# total_rewards=0,
# for i in range(10):
#     RTDPAgent_4 = RTDPAgent(env4_G, gamma=0.9, epsilon=0.2)
#     total_steps, total_rewards = RTDPAgent_4.run_eps(Vdiff14_gap, max_step = 50, num_eps = 100)
#     total_steps += total_steps
#     total_rewards += total_rewards
# print("Avg Gap: {}".format(total_rewards/total_steps))

# total_steps=0,
# total_rewards=0,
# for i in range(10):
#     RTDPAgent_4 = RTDPAgent(env4_G, gamma=0.9, epsilon=0.2)
#     total_steps, total_rewards = RTDPAgent_4.run_eps(Vdiff14_upper, max_step = 50, num_eps = 100)
#     total_steps += total_steps
#     total_rewards += total_rewards
# print("Avg Upper: {}".format(total_rewards/total_steps))

# total_steps=0,
# total_rewards=0,
# for i in range(10):
#     RTDPAgent_4 = RTDPAgent(env4_G, gamma=0.9, epsilon=0.2)
#     total_steps, total_rewards = RTDPAgent_4.run_eps(Vdiff14_lower, max_step = 50, num_eps = 100)
#     total_steps += total_steps
#     total_rewards += total_rewards
# print("Avg Lower: {}".format(total_rewards/total_steps))

# total_steps=0,
# total_rewards=0,
# for i in range(10):
#     RTDPAgent_4 = RTDPAgent(env4_G, gamma=0.9, epsilon=0.2)
#     RTDPAgent_4.Policy = policy1_converge 
#     total_steps, total_rewards = RTDPAgent_4.run_eps([], max_step = 50, num_eps = 100)
#     total_steps += total_steps
#     total_rewards += total_rewards
# print("Avg CS: {}".format(total_rewards/total_steps))

# total_steps=0,
# total_rewards=0,
# for i in range(10):
#     RTDPAgent_4 = RTDPAgent(env4_G, gamma=0.9, epsilon=0.2)
#     RTDPAgent_4.Policy = policy1_converge 
#     total_steps, total_rewards = RTDPAgent_4.run_eps(Vdiff14_gap, max_step = 50, num_eps = 100)
#     total_steps += total_steps
#     total_rewards += total_rewards
# print("Avg CS + Gap: {}".format(total_rewards/total_steps))



# ## QL Methods
# total_steps=0,
# total_rewards=0,
# for i in range(10):
#     QLAgent_2 = QLearningAgent(env2_G, gamma=0.9, epsilon=0.2)
#     total_steps, total_rewards = QLAgent_2.run_eps([], max_step = 50, num_eps = 100)
#     total_steps += total_steps
#     total_rewards += total_rewards
# print("Avg Random: {}".format(total_rewards/total_steps))

# total_steps=0,
# total_rewards=0,
# for i in range(10):
#     QLAgent_2 = QLearningAgent(env2_G, gamma=0.9, epsilon=0.2)
#     total_steps, total_rewards = QLAgent_2.run_eps(Vdiff12_gap, max_step = 50, num_eps = 100)
#     total_steps += total_steps
#     total_rewards += total_rewards
# print("Avg Gap: {}".format(total_rewards/total_steps))

# total_steps=0,
# total_rewards=0,
# for i in range(10):
#     QLAgent_2 = QLearningAgent(env2_G, gamma=0.9, epsilon=0.2)
#     total_steps, total_rewards = QLAgent_2.run_eps(Vdiff12_upper, max_step = 50, num_eps = 100)
#     total_steps += total_steps
#     total_rewards += total_rewards
# print("Avg Upper: {}".format(total_rewards/total_steps))

# total_steps=0,
# total_rewards=0,
# for i in range(10):
#     QLAgent_2 = QLearningAgent(env2_G, gamma=0.9, epsilon=0.2)
#     total_steps, total_rewards = QLAgent_2.run_eps(Vdiff12_lower, max_step = 50, num_eps = 100)
#     total_steps += total_steps
#     total_rewards += total_rewards
# print("Avg Lower: {}".format(total_rewards/total_steps))

# total_steps=0,
# total_rewards=0,
# for i in range(10):
#     QLAgent_2 = QLearningAgent(env2_G, gamma=0.9, epsilon=0.2)
#     QLAgent_2.Policy = policy1_converge 
#     total_steps, total_rewards = QLAgent_2.run_eps([], max_step = 50, num_eps = 100)
#     total_steps += total_steps
#     total_rewards += total_rewards
# print("Avg CS: {}".format(total_rewards/total_steps))

# total_steps=0,
# total_rewards=0,
# for i in range(10):
#     QLAgent_2 = QLearningAgent(env2_G, gamma=0.9, epsilon=0.2)
#     QLAgent_2.Policy = policy1_converge 
#     total_steps, total_rewards = QLAgent_2.run_eps(Vdiff12_gap, max_step = 50, num_eps = 100)
#     total_steps += total_steps
#     total_rewards += total_rewards
# print("Avg CS + Gap: {}".format(total_rewards/total_steps))


# ## QL Methods
# total_steps=0,
# total_rewards=0,
# for i in range(10):
#     QLAgent_3 = QLearningAgent(env3_G, gamma=0.9, epsilon=0.2)
#     total_steps, total_rewards = QLAgent_3.run_eps([], max_step = 50, num_eps = 100)
#     total_steps += total_steps
#     total_rewards += total_rewards
# print("Avg Random: {}".format(total_rewards/total_steps))

# total_steps=0,
# total_rewards=0,
# for i in range(10):
#     QLAgent_3 = QLearningAgent(env3_G, gamma=0.9, epsilon=0.2)
#     total_steps, total_rewards = QLAgent_3.run_eps(Vdiff13_gap, max_step = 50, num_eps = 100)
#     total_steps += total_steps
#     total_rewards += total_rewards
# print("Avg Gap: {}".format(total_rewards/total_steps))

# total_steps=0,
# total_rewards=0,
# for i in range(10):
#     QLAgent_2 = QLearningAgent(env3_G, gamma=0.9, epsilon=0.2)
#     total_steps, total_rewards = QLAgent_3.run_eps(Vdiff13_upper, max_step = 50, num_eps = 100)
#     total_steps += total_steps
#     total_rewards += total_rewards
# print("Avg Upper: {}".format(total_rewards/total_steps))

# total_steps=0,
# total_rewards=0,
# for i in range(10):
#     QLAgent_3 = QLearningAgent(env3_G, gamma=0.9, epsilon=0.2)
#     total_steps, total_rewards = QLAgent_3.run_eps(Vdiff13_lower, max_step = 50, num_eps = 100)
#     total_steps += total_steps
#     total_rewards += total_rewards
# print("Avg Lower: {}".format(total_rewards/total_steps))

# total_steps=0,
# total_rewards=0,
# for i in range(10):
#     QLAgent_3 = QLearningAgent(env3_G, gamma=0.9, epsilon=0.2)
#     QLAgent_3.Policy = policy1_converge 
#     total_steps, total_rewards = QLAgent_3.run_eps([], max_step = 50, num_eps = 100)
#     total_steps += total_steps
#     total_rewards += total_rewards
# print("Avg CS: {}".format(total_rewards/total_steps))

# total_steps=0,
# total_rewards=0,
# for i in range(10):
#     QLAgent_3 = QLearningAgent(env3_G, gamma=0.9, epsilon=0.2)
#     QLAgent_3.Policy = policy1_converge 
#     total_steps, total_rewards = QLAgent_3.run_eps(Vdiff12_gap, max_step = 50, num_eps = 100)
#     total_steps += total_steps
#     total_rewards += total_rewards
# print("Avg CS + Gap: {}".format(total_rewards/total_steps))

# # QL Methods
# total_steps=0,
# total_rewards=0,
# for i in range(10):
#     QLAgent_4 = QLearningAgent(env4_G, gamma=0.9, epsilon=0.2)
#     total_steps, total_rewards = QLAgent_4.run_eps([], max_step = 50, num_eps = 100)
#     total_steps += total_steps
#     total_rewards += total_rewards
# print("Avg Random: {}".format(total_rewards/total_steps))

# total_steps=0,
# total_rewards=0,
# for i in range(10):
#     QLAgent_4 = QLearningAgent(env4_G, gamma=0.9, epsilon=0.2)
#     total_steps, total_rewards = QLAgent_4.run_eps(Vdiff14_gap, max_step = 50, num_eps = 100)
#     total_steps += total_steps
#     total_rewards += total_rewards
# print("Avg Gap: {}".format(total_rewards/total_steps))

# total_steps=0,
# total_rewards=0,
# for i in range(10):
#     QLAgent_2 = QLearningAgent(env4_G, gamma=0.9, epsilon=0.2)
#     total_steps, total_rewards = QLAgent_4.run_eps(Vdiff14_upper, max_step = 50, num_eps = 100)
#     total_steps += total_steps
#     total_rewards += total_rewards
# print("Avg Upper: {}".format(total_rewards/total_steps))

# total_steps=0,
# total_rewards=0,
# for i in range(10):
#     QLAgent_4 = QLearningAgent(env4_G, gamma=0.9, epsilon=0.2)
#     total_steps, total_rewards = QLAgent_4.run_eps(Vdiff14_lower, max_step = 50, num_eps = 100)
#     total_steps += total_steps
#     total_rewards += total_rewards
# print("Avg Lower: {}".format(total_rewards/total_steps))

# total_steps=0,
# total_rewards=0,
# for i in range(10):
#     QLAgent_4 = QLearningAgent(env4_G, gamma=0.9, epsilon=0.2)
#     QLAgent_4.Policy = policy1_converge 
#     total_steps, total_rewards = QLAgent_4.run_eps([], max_step = 50, num_eps = 100)
#     total_steps += total_steps
#     total_rewards += total_rewards
# print("Avg CS: {}".format(total_rewards/total_steps))

# total_steps=0,
# total_rewards=0,
# for i in range(10):
#     QLAgent_4 = QLearningAgent(env4_G, gamma=0.9, epsilon=0.2)
#     QLAgent_4.Policy = policy1_converge 
#     total_steps, total_rewards = QLAgent_4.run_eps(Vdiff14_gap, max_step = 50, num_eps = 100)
#     total_steps += total_steps
#     total_rewards += total_rewards
# print("Avg CS + Gap: {}".format(total_rewards/total_steps))










# plot_matrix(env1_G, V_0,goal_coords=[(6,6)],title='Goal at 77', save_path='./imgs/empty_g77')
# plot_matrix(env2_G, V_0,goal_coords=[(5,5)],title='Goal at 66', save_path='./imgs/empty_g66')
# plot_matrix(env3_G, V_0,goal_coords=[(0,0)],title='Goal at 00', save_path='./imgs/empty_g00')
# plot_matrix(env4_G, V_0,goal_coords=[(6,3)],title='Goal at 73', save_path='./imgs/empty_g73')

# plot_matrix(env1_G, V1_converge, goal_coords=[(6,6)],title='Goal at 77', save_path='./imgs/VI_g77')
# plot_matrix(env2_G, V2_converge, goal_coords=[(6,3)],title='Goal at 73', save_path='./imgs/VI_g73')
# plot_matrix(env3_G, V3_converge, goal_coords=[(5,5)],title='Goal at 66', save_path='./imgs/VI_g66')
# plot_matrix(env4_G, V4_converge, goal_coords=[(0,0)],title='Goal at 00', save_path='./imgs/VI_g00')

# plot_policy_matrix(policy1_converge, DPAgent_1_converge.S, goal_coords=[(6,6)],title='Goal at 77', save_path='./imgs/P_g77')
# plot_policy_matrix(policy2_converge, DPAgent_2_converge.S, goal_coords=[(6,3)],title='Goal at 73', save_path='./imgs/P_g73')
# plot_policy_matrix(policy3_converge, DPAgent_3_converge.S, goal_coords=[(5,5)],title='Goal at 66', save_path='./imgs/P_g66')
# plot_policy_matrix(policy4_converge, DPAgent_4_converge.S, goal_coords=[(0,0)],title='Goal at 00', save_path='./imgs/P_g00')

# plot_matrix(env2_G, Vdiff12, goal_coords=[(6,6), (6, 3)],title='Vdiff12', save_path='./imgs/Vdiff12')
# plot_matrix(env2_G, Vdiff12_gap, goal_coords=[(6,6), (6, 3)],title='Vdiff12_Gap', save_path='./imgs/Vdiff12_Gap')
# plot_matrix(env2_G, Vdiff12_upper, goal_coords=[(6,6), (6, 3)],title='Vdiff12_Upper', save_path='./imgs/Vdiff12_Upper')
# plot_matrix(env2_G, Vdiff12_lower, goal_coords=[(6,6), (6, 3)],title='Vdiff12_Lower', save_path='./imgs/Vdiff12_Lower')

# plot_matrix(env3_G, Vdiff13, goal_coords=[(6,6), (5, 5)],title='Vdiff13', save_path='./imgs/Vdiff13')
# plot_matrix(env3_G, Vdiff13_gap, goal_coords=[(6,6), (5, 5)],title='Vdiff13_Gap', save_path='./imgs/Vdiff13_Gap')
# plot_matrix(env3_G, Vdiff13_upper, goal_coords=[(6,6), (5, 5)],title='Vdiff13_Upper', save_path='./imgs/Vdiff13_Upper')
# plot_matrix(env3_G, Vdiff13_lower, goal_coords=[(6,6), (5, 5)],title='Vdiff13_Lower', save_path='./imgs/Vdiff13_Lower')

# plot_matrix(env4_G, Vdiff14, goal_coords=[(6,6), (0, 0)],title='Vdiff14', save_path='./imgs/Vdiff14')
# plot_matrix(env4_G, Vdiff14_gap, goal_coords=[(6,6), (0, 0)],title='Vdiff14_Gap', save_path='./imgs/Vdiff14_Gap')
# plot_matrix(env4_G, Vdiff14_upper, goal_coords=[(6,6), (0, 0)],title='Vdiff14_Upper', save_path='./imgs/Vdiff14_Upper')
# plot_matrix(env4_G, Vdiff14_lower, goal_coords=[(6,6), (0, 0)],title='Vdiff14_Lower', save_path='./imgs/Vdiff14_Lower')







