from collections import defaultdict
import numpy as np
import copy

from Gridworld import GridWorldEnv
from utils import plot_matrix, plot_policy_matrix, plot_line_dict, plot_Qdiff_matrix
from pprint import pprint
from Models import DPAgent, QLearningAgent

# np.random.seed(0)

def run_QL_exp_warmstart(env, gamma=0.9, alpha=0.5, epsilon=0.1, max_step=20, num_eps = 50, num_runs = 10, explore='e-greedy', init_Q = []):
    total_dis_r = []
    for i in range(num_runs):
        agent = QLearningAgent(env, gamma=gamma, alpha=alpha, epsilon=epsilon)
        # if len(init_q) > 0:
        #     agent.warm_start_q(init_q)
        dis_r = agent.run_warmstart(init_Q=init_Q, max_step=max_step, num_eps=num_eps, explore=explore)
        total_dis_r.append(dis_r)
    avg_dis_r = np.average(total_dis_r, axis=0)
    return avg_dis_r



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

V1_pi_1, Q1_pi_1 = V1_converge, Q1_converge
V2_pi_2, Q2_pi_2 = V2_converge, Q2_converge
V1_pi_2_c, Q1_pi_2_c, _ = DPAgent_1.policy_evaluation(policy2_converge)
V2_pi_1_c, Q2_pi_1_c, _ = DPAgent_2.policy_evaluation(policy1_converge)

GAMMA = 0.9
ALPHA = 0.5
EPSILON = 0.1
TEMP = 0.01
NUM_RUNS = 10
NUM_EPS = 50
MAX_STEP = 20
QL_V2_eps_dict = {}

Vdiff12_lower_dict = {}
Qdiff12_lower_dict = {}
for i in range(0,101,5):
    Vdiff12_lower, Qdiff12_lower, n = DPAgent_diff_21.policy_evaluation(policy1_converge, max_iter=i)
    Vdiff12_lower_dict[i] = Vdiff12_lower
    Qdiff12_lower_dict[i] = Qdiff12_lower

for k,v in Qdiff12_lower_dict.items():
    Q2_pi_1 = v + Q1_pi_1
    eps_r_V2 = run_QL_exp_warmstart(env_2, gamma=0.9, alpha=ALPHA, epsilon=EPSILON , max_step=MAX_STEP, num_eps = NUM_EPS, num_runs=NUM_RUNS, init_Q = Q2_pi_1)
    QL_V2_eps_dict[k] = eps_r_V2

eps_r_V2_optimal = run_QL_exp_warmstart(env_2, gamma=GAMMA, alpha=0, epsilon=0.1 , max_step=MAX_STEP, num_eps = NUM_EPS, num_runs=NUM_RUNS, init_Q = Q2_pi_2)
QL_V2_eps_dict['optimal'] = eps_r_V2_optimal


for k,v in QL_V2_eps_dict.items():
    print(k, sum(v))

# raise ValueError
# Q2_pi_1 = Qdiff12_lower_list[0] + Q1_pi_1
# eps_r_V2 = run_QL_exp_warmstart(env_2, gamma=0.9, epsilon=EPSILON , max_step=100, num_eps = NUM_EPS, num_runs=NUM_RUNS, init_Q = Q2_pi_1)

# QL_V2_eps_dict = {}
# QL_V2_eps_dict['test'] = eps_r_V2

# plot_line_dict(QL_V2_eps_dict, './temp2', 'test')

# Vdiff12_lower, Qdiff12_lower, n = DPAgent_diff_21.policy_evaluation(policy1_converge, max_iter=100)
# V = Vdiff12_lower + V1_pi_1




# np.set_printoptions(precision=2)
# print(V1_pi_1.reshape(7,7)
# print(Vdiff12_lower.reshape(7,7))
# print(V.reshape(7,7))
# print(V2_pi_1.reshape(7,7))
# print(V2_converge.reshape(7,7))
# # V = V.reshape(7,7)
# np.set_printoptions(precision=2)
# print(V)

raise ValueError



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
V1_pi_4, Q1_pi_4 = DPAgent_1_converge.policy_evaluation(policy4_converge,)
V4_pi_1, Q4_pi_1 = DPAgent_4_converge.policy_evaluation(policy1_converge)

Vdiff14 = V4_pi_4 - V1_pi_1 
Vdiff14_lower = V4_pi_1 - V1_pi_1
Vdiff14_upper = V4_pi_4 - V1_pi_4
Vdiff14_gap = Vdiff14_upper - Vdiff14_lower

Qdiff14 = Q4_pi_4 - Q1_pi_1 
Qdiff14_lower = Q4_pi_1 - Q1_pi_1
Qdiff14_upper = Q4_pi_4 - Q1_pi_4
Qdiff14_gap = Qdiff14_upper - Qdiff14_lower

TEMP = 0.01
NUM_RUNS = 10
EPSILON = 0.1
NUM_EPS = 100
BOLTAZMANN = FALSE

# # QL Methods

# eps_steps_V2, eps_r_V2 = run_QL_exp(env2_G, gamma=0.9, epsilon=EPSILON , max_step=100, num_eps = NUM_EPS, num_runs=NUM_RUNS, temp=TEMP, V_heur = [], init_q = [])
# eps_steps_V2_ws, eps_r_V2_ws= run_QL_exp(env2_G, gamma=0.9, epsilon=EPSILON , max_step=100, num_eps = NUM_EPS, num_runs=NUM_RUNS, temp=TEMP, V_heur = [], init_q = Q1_converge)
# eps_steps_V2_ws_vdiff, eps_r_V2_ws_vidff = run_QL_exp(env2_G, gamma=0.9, epsilon=EPSILON , max_step=100, num_eps = NUM_EPS, num_runs=NUM_RUNS, temp=TEMP, V_heur = [], init_q = Q1_converge+Qdiff12_lower)

# # eps_steps_V2_ws_vdiff, eps_r_V2_ws_vidff = run_QL_exp(env2_G, gamma=0.9, epsilon=EPSILON , max_step=100, num_eps = NUM_EPS, num_runs=NUM_RUNS, temp=TEMP, V_heur = [Vdiff12], init_q = Q1_converge)

# # eps_steps_V2_ws_vgap, eps_r_V2_ws_vgap = run_QL_exp(env2_G, gamma=0.9, epsilon=EPSILON , max_step=100, num_eps = NUM_EPS, num_runs=NUM_RUNS, temp=TEMP, V_heur = Vdiff12_gap, init_q = Q1_converge)
# # eps_steps_V2_ws_vupper, eps_r_V2_ws_vupper = run_QL_exp(env2_G, gamma=0.9, epsilon=EPSILON , max_step=100, num_eps = NUM_EPS, num_runs=NUM_RUNS, temp=TEMP, V_heur = Vdiff12_upper, init_q = Q1_converge)
# # eps_steps_V2_ws_vlower, eps_r_V2_ws_vlower = run_QL_exp(env2_G, gamma=0.9, epsilon=EPSILON , max_step=100, num_eps = NUM_EPS, num_runs=NUM_RUNS, temp=TEMP, V_heur = Vdiff12_lower, init_q = Q1_converge)

# # QL_V2_eps_dict = {}
# # QL_V2_eps_dict['Boltzmann'] = eps_steps_V2
# # QL_V2_eps_dict['Boltzmann + Warm Start'] = eps_steps_V2_ws
# # QL_V2_eps_dict['Boltzmann + Warm Start + Vdiff'] = eps_steps_V2_ws_vdiff
# # QL_V2_eps_dict['Boltzmann + Warm Start + Vdiff Gap'] = eps_steps_V2_ws_vgap
# # QL_V2_eps_dict['Boltzmann + Warm Start + Vdiff Upper'] = eps_steps_V2_ws_vupper
# # QL_V2_eps_dict['Boltzmann+ Warm Start + Vdiff Lower'] = eps_steps_V2_ws_vlower

# # plot_line_dict(QL_V2_eps_dict, './imgs/QL_V2_avg_eps', 'QL_V2')


# QL_V2_eps_dict = {}
# QL_V2_eps_dict['e-greedy'] = eps_r_V2
# QL_V2_eps_dict['e-greed + Warm Start with Q1_pi1'] = eps_r_V2_ws
# QL_V2_eps_dict['e-greed + Warm Start with Q2_pi1'] = eps_r_V2_ws_vidff
# # QL_V2_eps_dict['Boltzmann + Warm Start + Vdiff Gap'] = eps_steps_V2_ws_vgap
# # QL_V2_eps_dict['Boltzmann + Warm Start + Vdiff Upper'] = eps_steps_V2_ws_vupper
# # QL_V2_eps_dict['Boltzmann+ Warm Start + Vdiff Lower'] = eps_steps_V2_ws_vlower

# plot_line_dict(QL_V2_eps_dict, './imgs/QL_V2_avg_dis_reward', 'QL_V2 Discounted Reward')

# ################################

# eps_steps_V4, eps_r_V4 = run_QL_exp(env4_G, gamma=0.9, epsilon=EPSILON , max_step=100, num_eps = NUM_EPS, num_runs=NUM_RUNS, temp=TEMP, V_heur = [], init_q = [])
# eps_steps_V4_ws, eps_r_V4_ws= run_QL_exp(env4_G, gamma=0.9, epsilon=EPSILON , max_step=100, num_eps = NUM_EPS, num_runs=NUM_RUNS, temp=TEMP, V_heur = [], init_q = Q1_converge)
# eps_steps_V4_ws_vdiff, eps_r_V4_ws_vidff = run_QL_exp(env4_G, gamma=0.9, epsilon=EPSILON , max_step=100, num_eps = NUM_EPS, num_runs=NUM_RUNS, temp=TEMP, V_heur = [], init_q = Q1_converge+Qdiff14_lower)

# QL_V4_eps_dict = {}
# QL_V4_eps_dict['e-greedy'] = eps_r_V4
# QL_V4_eps_dict['e-greed + Warm Start with Q1_pi1'] = eps_r_V4_ws
# QL_V4_eps_dict['e-greed + Warm Start with Q2_pi1'] = eps_r_V4_ws_vidff

# plot_line_dict(QL_V4_eps_dict, './imgs/QL_V4_avg_dis_reward', 'QL_V3 Discounted Reward')



# QL Methods

eps_steps_V2, eps_r_V2 = run_QL_exp(env2_G, gamma=0.9, epsilon=EPSILON , max_step=100, num_eps = NUM_EPS, num_runs=NUM_RUNS, temp=TEMP, V_heur = [], init_q = [])
eps_steps_V2_ws, eps_r_V2_ws= run_QL_exp(env2_G, gamma=0.9, epsilon=EPSILON , max_step=100, num_eps = NUM_EPS, num_runs=NUM_RUNS, temp=TEMP, V_heur = V1_pi_1, init_q = [])
eps_steps_V2_ws_vdiff, eps_r_V2_ws_vidff = run_QL_exp(env2_G, gamma=0.9, epsilon=EPSILON , max_step=100, num_eps = NUM_EPS, num_runs=NUM_RUNS, temp=TEMP, V_heur = Vdiff12_lower, init_q = [])
eps_steps_V2_ws_vdiff_plus, eps_r_V2_ws_vidff_plus = run_QL_exp(env2_G, gamma=0.9, epsilon=EPSILON , max_step=100, num_eps = NUM_EPS, num_runs=NUM_RUNS, temp=TEMP, V_heur = V2_pi_1, init_q = [])
eps_steps_V2_ws_vdiff_abs, eps_r_V2_ws_vidff_abs = run_QL_exp(env2_G, gamma=0.9, epsilon=EPSILON , max_step=100, num_eps = NUM_EPS, num_runs=NUM_RUNS, temp=TEMP, V_heur = abs(Vdiff12_lower), init_q = [])

# eps_steps_V2_ws_vdiff, eps_r_V2_ws_vidff = run_QL_exp(env2_G, gamma=0.9, epsilon=EPSILON , max_step=100, num_eps = NUM_EPS, num_runs=NUM_RUNS, temp=TEMP, V_heur = [Vdiff12], init_q = Q1_converge)

# eps_steps_V2_ws_vgap, eps_r_V2_ws_vgap = run_QL_exp(env2_G, gamma=0.9, epsilon=EPSILON , max_step=100, num_eps = NUM_EPS, num_runs=NUM_RUNS, temp=TEMP, V_heur = Vdiff12_gap, init_q = Q1_converge)
# eps_steps_V2_ws_vupper, eps_r_V2_ws_vupper = run_QL_exp(env2_G, gamma=0.9, epsilon=EPSILON , max_step=100, num_eps = NUM_EPS, num_runs=NUM_RUNS, temp=TEMP, V_heur = Vdiff12_upper, init_q = Q1_converge)
# eps_steps_V2_ws_vlower, eps_r_V2_ws_vlower = run_QL_exp(env2_G, gamma=0.9, epsilon=EPSILON , max_step=100, num_eps = NUM_EPS, num_runs=NUM_RUNS, temp=TEMP, V_heur = Vdiff12_lower, init_q = Q1_converge)

# QL_V2_eps_dict = {}
# QL_V2_eps_dict['Boltzmann'] = eps_steps_V2
# QL_V2_eps_dict['Boltzmann + Warm Start'] = eps_steps_V2_ws
# QL_V2_eps_dict['Boltzmann + Warm Start + Vdiff'] = eps_steps_V2_ws_vdiff
# QL_V2_eps_dict['Boltzmann + Warm Start + Vdiff Gap'] = eps_steps_V2_ws_vgap
# QL_V2_eps_dict['Boltzmann + Warm Start + Vdiff Upper'] = eps_steps_V2_ws_vupper
# QL_V2_eps_dict['Boltzmann+ Warm Start + Vdiff Lower'] = eps_steps_V2_ws_vlower

# plot_line_dict(QL_V2_eps_dict, './imgs/QL_V2_avg_eps', 'QL_V2')


QL_V2_eps_dict = {}
QL_V2_eps_dict['Boltz + 0'] = eps_r_V2
QL_V2_eps_dict['Boltz + V1_pi_1'] = eps_r_V2_ws
QL_V2_eps_dict['Boltz + Vdiff12_lower'] = eps_r_V2_ws_vidff
QL_V2_eps_dict['Boltz + V2_pi_1'] = eps_r_V2_ws_vidff_plus
QL_V2_eps_dict['Boltz + abs(Vdiff12_lower)'] = eps_r_V2_ws_vidff_abs
# QL_V2_eps_dict['Boltzmann + Warm Start + Vdiff Gap'] = eps_steps_V2_ws_vgap
# QL_V2_eps_dict['Boltzmann + Warm Start + Vdiff Upper'] = eps_steps_V2_ws_vupper
# QL_V2_eps_dict['Boltzmann+ Warm Start + Vdiff Lower'] = eps_steps_V2_ws_vlower

plot_line_dict(QL_V2_eps_dict, './imgs/Boltz_V2_avg_dis_reward2', 'Boltz_V2 Discounted Reward2')

# ################################

# eps_steps_V4, eps_r_V4 = run_QL_exp(env4_G, gamma=0.9, epsilon=EPSILON , max_step=100, num_eps = NUM_EPS, num_runs=NUM_RUNS, temp=TEMP, V_heur = [], init_q = [])
# eps_steps_V4_ws, eps_r_V4_ws= run_QL_exp(env4_G, gamma=0.9, epsilon=EPSILON , max_step=100, num_eps = NUM_EPS, num_runs=NUM_RUNS, temp=TEMP, V_heur = [], init_q = Q1_converge)
# eps_steps_V4_ws_vdiff, eps_r_V4_ws_vidff = run_QL_exp(env4_G, gamma=0.9, epsilon=EPSILON , max_step=100, num_eps = NUM_EPS, num_runs=NUM_RUNS, temp=TEMP, V_heur = [], init_q = Q1_converge+Qdiff14_lower)

# QL_V4_eps_dict = {}
# QL_V4_eps_dict['e-greedy'] = eps_r_V4
# QL_V4_eps_dict['e-greed + Warm Start with Q1_pi1'] = eps_r_V4_ws
# QL_V4_eps_dict['e-greed + Warm Start with Q2_pi1'] = eps_r_V4_ws_vidff

# plot_line_dict(QL_V4_eps_dict, './imgs/QL_V4_avg_dis_reward', 'QL_V3 Discounted Reward')


# eps_steps_V3, eps_r_V3 = run_QL_exp(env3_G, gamma=0.9, epsilon=EPSILON , max_step=100, num_eps = NUM_EPS, num_runs=NUM_RUNS, temp=TEMP, V_heur = [], init_q = [])
# eps_steps_V3_ws, eps_r_V3_ws= run_QL_exp(env3_G, gamma=0.9, epsilon=EPSILON , max_step=100, num_eps = NUM_EPS, num_runs=NUM_RUNS, temp=TEMP, V_heur = [], init_q = Q1_converge)
# eps_steps_V3_ws_vdiff, eps_r_V3_ws_vidff = run_QL_exp(env3_G, gamma=0.9, epsilon=EPSILON , max_step=100, num_eps = NUM_EPS, num_runs=NUM_RUNS, temp=TEMP, V_heur = Vdiff13, init_q = Q1_converge)
# eps_steps_V3_ws_vgap, eps_r_V3_ws_vgap = run_QL_exp(env3_G, gamma=0.9, epsilon=EPSILON , max_step=100, num_eps = NUM_EPS, num_runs=NUM_RUNS, temp=TEMP, V_heur = Vdiff13_gap, init_q = Q1_converge)
# eps_steps_V3_ws_vupper, eps_r_V3_ws_vupper = run_QL_exp(env3_G, gamma=0.9, epsilon=EPSILON , max_step=100, num_eps = NUM_EPS, num_runs=NUM_RUNS, temp=TEMP, V_heur = Vdiff13_upper, init_q = Q1_converge)
# eps_steps_V3_ws_vlower, eps_r_V3_ws_vlower = run_QL_exp(env3_G, gamma=0.9, epsilon=EPSILON , max_step=100, num_eps = NUM_EPS, num_runs=NUM_RUNS, temp=TEMP, V_heur = Vdiff13_lower, init_q = Q1_converge)

# QL_V3_eps_dict = {}
# QL_V3_eps_dict['Boltzmann'] = eps_steps_V3
# QL_V3_eps_dict['Boltzmann + Warm Start'] = eps_steps_V3_ws
# QL_V3_eps_dict['Boltzmann + Warm Start + Vdiff'] = eps_steps_V3_ws_vdiff
# QL_V3_eps_dict['Boltzmann + Warm Start + Vdiff Gap'] = eps_steps_V3_ws_vgap
# QL_V3_eps_dict['Boltzmann + Warm Start + Vdiff Upper'] = eps_steps_V3_ws_vupper
# QL_V3_eps_dict['Boltzmann+ Warm Start + Vdiff Lower'] = eps_steps_V3_ws_vlower

# plot_line_dict(QL_V3_eps_dict, './imgs/QL_V3_avg_eps', 'QL_V3')


# eps_steps_V4, eps_r_V4 = run_QL_exp(env4_G, gamma=0.9, epsilon=EPSILON , max_step=100, num_eps = NUM_EPS, num_runs=NUM_RUNS, temp=TEMP, V_heur = [], init_q = [])
# eps_steps_V4_ws, eps_r_V4_ws= run_QL_exp(env4_G, gamma=0.9, epsilon=EPSILON , max_step=100, num_eps = NUM_EPS, num_runs=NUM_RUNS, temp=TEMP, V_heur = [], init_q = Q1_converge)
# eps_steps_V4_ws_vdiff, eps_r_V4_ws_vidff = run_QL_exp(env4_G, gamma=0.9, epsilon=EPSILON , max_step=100, num_eps = NUM_EPS, num_runs=NUM_RUNS, temp=TEMP, V_heur = Vdiff14, init_q = Q1_converge)
# eps_steps_V4_ws_vgap, eps_r_V4_ws_vgap = run_QL_exp(env4_G, gamma=0.9, epsilon=EPSILON , max_step=100, num_eps = NUM_EPS, num_runs=NUM_RUNS, temp=TEMP, V_heur = Vdiff14_gap, init_q = Q1_converge)
# eps_steps_V4_ws_vupper, eps_r_V4_ws_vupper = run_QL_exp(env4_G, gamma=0.9, epsilon=EPSILON , max_step=100, num_eps = NUM_EPS, num_runs=NUM_RUNS, temp=TEMP, V_heur = Vdiff14_upper, init_q = Q1_converge)
# eps_steps_V4_ws_vlower, eps_r_V4_ws_vlower = run_QL_exp(env4_G, gamma=0.9, epsilon=EPSILON , max_step=100, num_eps = NUM_EPS, num_runs=NUM_RUNS, temp=TEMP, V_heur = Vdiff14_lower, init_q = Q1_converge)

# QL_V4_eps_dict = {}
# QL_V4_eps_dict['Boltzmann'] = eps_steps_V4
# QL_V4_eps_dict['Boltzmann + Warm Start'] = eps_steps_V4_ws
# QL_V4_eps_dict['Boltzmann + Warm Start + Vdiff'] = eps_steps_V4_ws_vdiff
# QL_V4_eps_dict['Boltzmann + Warm Start + Vdiff Gap'] = eps_steps_V4_ws_vgap
# QL_V4_eps_dict['Boltzmann + Warm Start + Vdiff Upper'] = eps_steps_V4_ws_vupper
# QL_V4_eps_dict['Boltzmann+ Warm Start + Vdiff Lower'] = eps_steps_V4_ws_vlower

# plot_line_dict(QL_V4_eps_dict, './imgs/QL_V4_avg_eps', 'QL_V4')

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







