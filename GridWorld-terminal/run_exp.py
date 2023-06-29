from collections import defaultdict
import numpy as np
import copy

from Gridworld import GridWorldEnv
from utils import plot_matrix, plot_policy_matrix, plot_line_dict, plot_Qdiff_matrix
from pprint import pprint
from Models import DPAgent, RTDPAgent, QLearningAgent

np.random.seed(0)


def run_VI_exp(DPAgent,V_p1, Vdiff,  Vdiff_gap, Vdiff_upper, Vdiff_lower):
    _, _, _, steps_converge = DPAgent.value_iteration()
    _, _, _, steps_V_p1 = DPAgent.value_iteration(rank_V=V_p1)
    _, _, _, steps_Vdiff = DPAgent.value_iteration(rank_V=Vdiff)
    _, _, _, steps_Vdiff_gap = DPAgent.value_iteration(rank_V=Vdiff_gap)
    _, _, _, steps_Vdiff_upper = DPAgent.value_iteration(rank_V=Vdiff_upper)
    _, _, _, steps_Vdiff_lower = DPAgent.value_iteration(rank_V=Vdiff_lower)
    print("Random: {}, Cold Start: {}, Diff: {}, Gap: {}, Upper: {}, Lower: {}".format(steps_converge, \
        steps_V_p1, steps_Vdiff, steps_Vdiff_gap, steps_Vdiff_upper, steps_Vdiff_lower))


def run_RTDP_exp(env, gamma=0.9, epsilon=0.2, max_step=100, num_eps = 50, num_runs = 100, V_heur = [], init_policy=[]):
    total_steps = []
    for i in range(num_runs):
        agent = RTDPAgent(env, gamma, epsilon)
        if init_policy != []:
            agent.Policy = init_policy.copy()
        num_steps = agent.run_eps(V_heur, max_step, num_eps)
        total_steps.append(num_steps)
    avg_steps = np.average(total_steps, axis=0)
    return avg_steps

def run_QL_exp(env, gamma=0.9, epsilon=0.2, max_step=100, num_eps = 50, num_runs = 100, t=0, V_heur = [], init_q = [], init_policy=[]):
    total_steps = []
    for i in range(num_runs):
        agent = QLearningAgent(env, gamma, epsilon)
        if init_policy != []:
            agent.Policy = init_policy.copy()
        if init_q != []:
            agent.warm_start_q(init_q)
        num_steps = agent.run_eps(V_heur=V_heur, t=t, max_step=max_step, num_eps=num_eps)
        total_steps.append(num_steps)
    avg_steps = np.average(total_steps, axis=0)
    return avg_steps


env1_G = GridWorldEnv(is_slippery=False, map_name="7x7_S00G77", terminal_states="GH")
env2_G = GridWorldEnv(is_slippery=False, map_name="7x7_S00G73", terminal_states="GH")
env3_G = GridWorldEnv(is_slippery=False, map_name="7x7_S00G66", terminal_states="GH")
env4_G = GridWorldEnv(is_slippery=False, map_name="7x7_S77G00", terminal_states="GH")
env5_G = GridWorldEnv(is_slippery=False, map_name="20x20_S00G1919", terminal_states="GH")
env6_G = GridWorldEnv(is_slippery=False, map_name="20x20_S00G1515", terminal_states="GH")

DPAgent_1_converge = DPAgent(env1_G, gamma=0.9, theta=1e-6)
DPAgent_2_converge = DPAgent(env2_G, gamma=0.9, theta=1e-6)
DPAgent_3_converge = DPAgent(env3_G, gamma=0.9, theta=1e-6)
DPAgent_4_converge = DPAgent(env4_G, gamma=0.9, theta=1e-6)
DPAgent_5_converge = DPAgent(env5_G, gamma=0.9, theta=1e-6)
DPAgent_6_converge = DPAgent(env6_G, gamma=0.9, theta=1e-6)

policy1_converge, V1_converge, Q1_converge, steps1_converge = DPAgent_1_converge.value_iteration()
policy2_converge, V2_converge, Q2_converge, steps2_converge = DPAgent_2_converge.value_iteration()
policy3_converge, V3_converge, Q3_converge, steps3_converge = DPAgent_3_converge.value_iteration()
policy4_converge, V4_converge, Q4_converge, steps4_converge = DPAgent_4_converge.value_iteration()
policy5_converge, V5_converge, Q5_converge, steps5_converge = DPAgent_5_converge.value_iteration()
policy6_converge, V6_converge, Q6_converge, steps6_converge = DPAgent_6_converge.value_iteration()

V1_pi_1, Q1_pi_1 = V1_converge, Q1_converge
V2_pi_2, Q2_pi_2 = V2_converge, Q2_converge
V1_pi_2, Q1_pi_2 = DPAgent_1_converge.policy_evaluation(policy2_converge)
V2_pi_1, Q2_pi_1 = DPAgent_2_converge.policy_evaluation(policy1_converge)

Vdiff12 = V2_pi_2 - V1_pi_1 
Vdiff12_lower = V2_pi_1 - V1_pi_1
Vdiff12_upper = V2_pi_2 - V1_pi_2
Vdiff12_gap = Vdiff12_upper - Vdiff12_lower

Qdiff12 = Q2_pi_2 - Q1_pi_1
Qdiff12_lower = Q2_pi_1 - Q1_pi_1
Qdiff12_upper = Q2_pi_2 - Q1_pi_2
Qdiff12_gap = Qdiff12_upper - Qdiff12_lower

V1_pi_1, Q1_pi_1 = V1_converge, Q1_converge
V3_pi_3, Q3_pi_3 = V3_converge, Q3_converge
V1_pi_3, Q1_pi_3 = DPAgent_1_converge.policy_evaluation(policy3_converge)
V3_pi_1, Q3_pi_1 = DPAgent_3_converge.policy_evaluation(policy1_converge)

Vdiff13 = V3_pi_3 - V1_pi_1 
Vdiff13_lower = V3_pi_1 - V1_pi_1
Vdiff13_upper = V3_pi_3 - V1_pi_3
Vdiff13_gap = Vdiff13_upper - Vdiff13_lower

Qdiff13 = Q3_pi_3 - Q1_pi_1 
Qdiff13_lower = Q3_pi_1 - Q1_pi_1
Qdiff13_upper = Q3_pi_3 - Q1_pi_3
Qdiff13_gap = Qdiff13_upper - Qdiff13_lower

V1_pi_1, Q1_pi_1 = V1_converge, Q1_converge
V4_pi_4, Q4_pi_4 = V4_converge, Q4_converge
V1_pi_4, Q1_pi_4 = DPAgent_1_converge.policy_evaluation(policy4_converge)
V4_pi_1, Q4_pi_1 = DPAgent_4_converge.policy_evaluation(policy1_converge)

Vdiff14 = V4_pi_4 - V1_pi_1 
Vdiff14_lower = V4_pi_1 - V1_pi_1
Vdiff14_upper = V4_pi_4 - V1_pi_4
Vdiff14_gap = Vdiff14_upper - Vdiff14_lower

Qdiff14 = Q4_pi_4 - Q1_pi_1 
Qdiff14_lower = Q4_pi_1 - Q1_pi_1
Qdiff14_upper = Q4_pi_4 - Q1_pi_4
Qdiff14_gap = Qdiff14_upper - Qdiff14_lower

# # # VI Methods
# run_VI_exp(DPAgent_2_converge,V1_converge, Vdiff12_gap, Vdiff12, Vdiff12_upper, Vdiff12_lower)
# run_VI_exp(DPAgent_3_converge,V1_converge, Vdiff13_gap, Vdiff13, Vdiff13_upper, Vdiff13_lower)
# run_VI_exp(DPAgent_4_converge,V1_converge, Vdiff14_gap, Vdiff14, Vdiff14_upper, Vdiff14_lower)


# # PI Methods
# p, v, s = DPAgent_2_converge.policy_iteration(policy1_converge)


# # ## RTDP Methods

# eps_steps_V2 = run_RTDP_exp(env2_G, gamma=0.9, epsilon=0.2, max_step=100, num_eps = 50, num_runs=100, V_heur = [], init_policy=[])
# eps_steps_V2_ws = run_RTDP_exp(env2_G, gamma=0.9, epsilon=0.2, max_step=100, num_eps = 50, num_runs=100, V_heur = [], init_policy=policy1_converge)
# eps_steps_V2_ws_vdiff = run_RTDP_exp(env2_G, gamma=0.9, epsilon=0.2, max_step=100, num_eps = 50, num_runs=100, V_heur = Vdiff12, init_policy=policy1_converge)
# eps_steps_V2_ws_vgap = run_RTDP_exp(env2_G, gamma=0.9, epsilon=0.2, max_step=100, num_eps = 50, num_runs=100, V_heur = Vdiff12_gap, init_policy=policy1_converge)
# eps_steps_V2_ws_vupper = run_RTDP_exp(env2_G, gamma=0.9, epsilon=0.2, max_step=100, num_eps = 50, num_runs=100, V_heur = Vdiff12_upper, init_policy=policy1_converge)
# eps_steps_V2_ws_vlower = run_RTDP_exp(env2_G, gamma=0.9, epsilon=0.2, max_step=100, num_eps = 50, num_runs=100, V_heur = Vdiff12_lower, init_policy=policy1_converge)

# RTDP_V2_eps_dict = {}
# RTDP_V2_eps_dict['E-Greedy'] = eps_steps_V2
# RTDP_V2_eps_dict['E-Greedy + Warm Start'] = eps_steps_V2_ws
# RTDP_V2_eps_dict['E-Greedy + Warm Start + Vdiff'] = eps_steps_V2_ws_vdiff
# RTDP_V2_eps_dict['E-Greedy + Warm Start + Vdiff Gap'] = eps_steps_V2_ws_vgap
# RTDP_V2_eps_dict['E-Greedy + Warm Start + Vdiff Upper'] = eps_steps_V2_ws_vupper
# RTDP_V2_eps_dict['E-Greedy + Warm Start + Vdiff Lower'] = eps_steps_V2_ws_vlower

# plot_line_dict(RTDP_V2_eps_dict, './imgs/RTDP_V2_avg_eps', 'RTDP_V2')

# eps_steps_V3 = run_RTDP_exp(env3_G, gamma=0.9, epsilon=0.2, max_step=100, num_eps = 50, num_runs=100, V_heur = [], init_policy=[])
# eps_steps_V3_ws = run_RTDP_exp(env3_G, gamma=0.9, epsilon=0.2, max_step=100, num_eps = 50, num_runs=100, V_heur = [], init_policy=policy1_converge)
# eps_steps_V3_ws_vdiff = run_RTDP_exp(env3_G, gamma=0.9, epsilon=0.2, max_step=100, num_eps = 50, num_runs=100, V_heur = Vdiff13, init_policy=policy1_converge)
# eps_steps_V3_ws_vgap = run_RTDP_exp(env3_G, gamma=0.9, epsilon=0.2, max_step=100, num_eps = 50, num_runs=100, V_heur = Vdiff13_gap, init_policy=policy1_converge)
# eps_steps_V3_ws_vupper = run_RTDP_exp(env3_G, gamma=0.9, epsilon=0.2, max_step=100, num_eps = 50, num_runs=100, V_heur = Vdiff13_upper, init_policy=policy1_converge)
# eps_steps_V3_ws_vlower = run_RTDP_exp(env3_G, gamma=0.9, epsilon=0.2, max_step=100, num_eps = 50, num_runs=100, V_heur = Vdiff13_lower, init_policy=policy1_converge)

# RTDP_V3_eps_dict = {}
# RTDP_V3_eps_dict['E-Greedy'] = eps_steps_V3
# RTDP_V3_eps_dict['E-Greedy + Warm Start'] = eps_steps_V3_ws
# RTDP_V3_eps_dict['E-Greedy + Warm Start + Vdiff'] = eps_steps_V3_ws_vdiff
# RTDP_V3_eps_dict['E-Greedy + Warm Start + Vdiff Gap'] = eps_steps_V3_ws_vgap
# RTDP_V3_eps_dict['E-Greedy + Warm Start + Vdiff Upper'] = eps_steps_V3_ws_vupper
# RTDP_V3_eps_dict['E-Greedy + Warm Start + Vdiff Lower'] = eps_steps_V3_ws_vlower

# plot_line_dict(RTDP_V3_eps_dict, './imgs/RTDP_V3_avg_eps', 'RTDP_V3')

# eps_steps_V4 = run_RTDP_exp(env4_G, gamma=0.9, epsilon=0.2, max_step=100, num_eps = 50, num_runs=100, V_heur = [], init_policy=[])
# eps_steps_V4_ws = run_RTDP_exp(env4_G, gamma=0.9, epsilon=0.2, max_step=100, num_eps = 50, num_runs=100, V_heur = [], init_policy=policy1_converge)
# eps_steps_V4_ws_vdiff = run_RTDP_exp(env4_G, gamma=0.9, epsilon=0.2, max_step=100, num_eps = 50, num_runs=100, V_heur = Vdiff14, init_policy=policy1_converge)
# eps_steps_V4_ws_vgap = run_RTDP_exp(env4_G, gamma=0.9, epsilon=0.2, max_step=100, num_eps = 50, num_runs=100, V_heur = Vdiff14_gap, init_policy=policy1_converge)
# eps_steps_V4_ws_vupper = run_RTDP_exp(env4_G, gamma=0.9, epsilon=0.2, max_step=100, num_eps = 50, num_runs=100, V_heur = Vdiff14_upper, init_policy=policy1_converge)
# eps_steps_V4_ws_vlower = run_RTDP_exp(env4_G, gamma=0.9, epsilon=0.2, max_step=100, num_eps = 50, num_runs=100, V_heur = Vdiff14_lower, init_policy=policy1_converge)

# RTDP_V4_eps_dict = {}
# RTDP_V4_eps_dict['E-Greedy'] = eps_steps_V4
# RTDP_V4_eps_dict['E-Greedy + Warm Start'] = eps_steps_V4_ws
# RTDP_V4_eps_dict['E-Greedy + Warm Start + Vdiff'] = eps_steps_V4_ws_vdiff
# RTDP_V4_eps_dict['E-Greedy + Warm Start + Vdiff Gap'] = eps_steps_V4_ws_vgap
# RTDP_V4_eps_dict['E-Greedy + Warm Start + Vdiff Upper'] = eps_steps_V4_ws_vupper
# RTDP_V4_eps_dict['E-Greedy + Warm Start + Vdiff Lower'] = eps_steps_V4_ws_vlower

# plot_line_dict(RTDP_V4_eps_dict, './imgs/RTDP_V4_avg_eps', "RTDP_V4")

set_t = 1000

# QL Methods
eps_steps_V2 = run_QL_exp(env2_G, gamma=0.9, epsilon=0.2, max_step=100, num_eps = 50, num_runs=10, t=set_t, V_heur = [], init_q = [], init_policy=policy1_converge)
eps_steps_V2_ws = run_QL_exp(env2_G, gamma=0.9, epsilon=0.2, max_step=100, num_eps = 50, num_runs=10, t=set_t, V_heur = [], init_q = [], init_policy=policy1_converge)
eps_steps_V2_ws_vdiff = run_QL_exp(env2_G, gamma=0.9, epsilon=0.2, max_step=100, num_eps = 50, num_runs=10, t=set_t, V_heur = Vdiff12, init_q = [], init_policy=policy1_converge)
eps_steps_V2_ws_vgap = run_QL_exp(env2_G, gamma=0.9, epsilon=0.2, max_step=100, num_eps = 50, num_runs=10, t=set_t, V_heur = Vdiff12_gap, init_q = [], init_policy=policy1_converge)
eps_steps_V2_ws_vupper = run_QL_exp(env2_G, gamma=0.9, epsilon=0.2, max_step=100, num_eps = 50, num_runs=10, t=set_t, V_heur = Vdiff12_upper, init_q = [], init_policy=policy1_converge)
eps_steps_V2_ws_vlower = run_QL_exp(env2_G, gamma=0.9, epsilon=0.2, max_step=100, num_eps = 50, num_runs=10, t=set_t, V_heur = Vdiff12_lower, init_q = [], init_policy=policy1_converge)

QL_V2_eps_dict = {}
QL_V2_eps_dict['E-Greedy'] = eps_steps_V2
QL_V2_eps_dict['E-Greedy + Warm Start'] = eps_steps_V2_ws
QL_V2_eps_dict['E-Greedy + Warm Start + Vdiff'] = eps_steps_V2_ws_vdiff
QL_V2_eps_dict['E-Greedy + Warm Start + Vdiff Gap'] = eps_steps_V2_ws_vgap
QL_V2_eps_dict['E-Greedy + Warm Start + Vdiff Upper'] = eps_steps_V2_ws_vupper
QL_V2_eps_dict['E-Greedy + Warm Start + Vdiff Lower'] = eps_steps_V2_ws_vlower

plot_line_dict(QL_V2_eps_dict, './imgs/QL_V2_avg_eps', 'QL_V2')

# eps_steps_V3 = run_QL_exp(env3_G, gamma=0.9, epsilon=0.2, max_step=100, num_eps = 50, num_runs=100, V_heur = [], init_policy=[])
# eps_steps_V3_ws = run_QL_exp(env3_G, gamma=0.9, epsilon=0.2, max_step=100, num_eps = 50, num_runs=100, V_heur = [], init_policy=policy1_converge)
# eps_steps_V3_ws_vdiff = run_QL_exp(env3_G, gamma=0.9, epsilon=0.2, max_step=100, num_eps = 50, num_runs=100, V_heur = Vdiff13, init_policy=policy1_converge)
# eps_steps_V3_ws_vgap = run_QL_exp(env3_G, gamma=0.9, epsilon=0.2, max_step=100, num_eps = 50, num_runs=100, V_heur = Vdiff13_gap, init_policy=policy1_converge)
# eps_steps_V3_ws_vupper = run_QL_exp(env3_G, gamma=0.9, epsilon=0.2, max_step=100, num_eps = 50, num_runs=100, V_heur = Vdiff13_upper, init_policy=policy1_converge)
# eps_steps_V3_ws_vlower = run_QL_exp(env3_G, gamma=0.9, epsilon=0.2, max_step=100, num_eps = 50, num_runs=100, V_heur = Vdiff13_lower, init_policy=policy1_converge)

# QL_V3_eps_dict = {}
# QL_V3_eps_dict['E-Greedy'] = eps_steps_V3
# QL_V3_eps_dict['E-Greedy + Warm Start'] = eps_steps_V3_ws
# QL_V3_eps_dict['E-Greedy + Warm Start + Vdiff'] = eps_steps_V3_ws_vdiff
# QL_V3_eps_dict['E-Greedy + Warm Start + Vdiff Gap'] = eps_steps_V3_ws_vgap
# QL_V3_eps_dict['E-Greedy + Warm Start + Vdiff Upper'] = eps_steps_V3_ws_vupper
# QL_V3_eps_dict['E-Greedy + Warm Start + Vdiff Lower'] = eps_steps_V3_ws_vlower

# plot_line_dict(QL_V3_eps_dict, './imgs/QL_V3_avg_eps', 'QL_V3')


# eps_steps_V4 = run_QL_exp(env4_G, gamma=0.9, epsilon=0.2, max_step=100, num_eps = 50, num_runs=100, V_heur = [], init_policy=[])
# eps_steps_V4_ws = run_QL_exp(env4_G, gamma=0.9, epsilon=0.2, max_step=100, num_eps = 50, num_runs=100, V_heur = [], init_policy=policy1_converge)
# eps_steps_V4_ws_vdiff = run_QL_exp(env4_G, gamma=0.9, epsilon=0.2, max_step=100, num_eps = 50, num_runs=100, V_heur = Vdiff14, init_policy=policy1_converge)
# eps_steps_V4_ws_vgap = run_QL_exp(env4_G, gamma=0.9, epsilon=0.2, max_step=100, num_eps = 50, num_runs=100, V_heur = Vdiff14_gap, init_policy=policy1_converge)
# eps_steps_V4_ws_vupper = run_QL_exp(env4_G, gamma=0.9, epsilon=0.2, max_step=100, num_eps = 50, num_runs=100, V_heur = Vdiff14_upper, init_policy=policy1_converge)
# eps_steps_V4_ws_vlower = run_QL_exp(env4_G, gamma=0.9, epsilon=0.2, max_step=100, num_eps = 50, num_runs=100, V_heur = Vdiff14_lower, init_policy=policy1_converge)

# QL_V4_eps_dict = {}
# QL_V4_eps_dict['E-Greedy'] = eps_steps_V4
# QL_V4_eps_dict['E-Greedy + Warm Start'] = eps_steps_V4_ws
# QL_V4_eps_dict['E-Greedy + Warm Start + Vdiff'] = eps_steps_V4_ws_vdiff
# QL_V4_eps_dict['E-Greedy + Warm Start + Vdiff Gap'] = eps_steps_V4_ws_vgap
# QL_V4_eps_dict['E-Greedy + Warm Start + Vdiff Upper'] = eps_steps_V4_ws_vupper
# QL_V4_eps_dict['E-Greedy + Warm Start + Vdiff Lower'] = eps_steps_V4_ws_vlower

# plot_line_dict(QL_V4_eps_dict, './imgs/QL_V4_avg_eps', "QL_V4")



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







