from collections import defaultdict
import numpy as np
import copy
import json

from Gridworld import GridWorldEnv
from utils import plot_matrix, plot_policy_matrix, plot_line_dict, plot_Qdiff_matrix
from pprint import pprint
from Models import DPAgent, QLearningAgent

# np.random.seed(0)
np.set_printoptions(precision=2)

RANDOM_START = False
GAMMA = 0.9
THETA = 1e-6

env_1 = GridWorldEnv(is_slippery=False, map_name="7x7_S00G77", random_start=RANDOM_START)
env_12 = GridWorldEnv(is_slippery=False, map_name="7x7_S00G77T74", random_start=RANDOM_START)
env_13 = GridWorldEnv(is_slippery=False, map_name="7x7_S00G77T66", random_start=RANDOM_START)
env_2 = GridWorldEnv(is_slippery=False, map_name="7x7_S00G74G77", random_start=RANDOM_START)
env_3 = GridWorldEnv(is_slippery=False, map_name="7x7_S00G66G77", random_start=RANDOM_START)

env_diff_21 = GridWorldEnv(is_slippery=False, map_name="7x7_S00G77T74", random_start=RANDOM_START, reward_matrix=env_2.reward_matrix-env_12.reward_matrix)
env_diff_31 = GridWorldEnv(is_slippery=False, map_name="7x7_S00G77T66", random_start=RANDOM_START, reward_matrix=env_3.reward_matrix-env_13.reward_matrix)

DPAgent_1 = DPAgent(env_1, gamma=GAMMA, theta=THETA)
DPAgent_12 = DPAgent(env_12, gamma=GAMMA, theta=THETA)
DPAgent_13 = DPAgent(env_13, gamma=GAMMA, theta=THETA)
DPAgent_2 = DPAgent(env_2, gamma=GAMMA, theta=THETA)
DPAgent_3 = DPAgent(env_3, gamma=GAMMA, theta=THETA)

DPAgent_diff_21 = DPAgent(env_diff_21, gamma=GAMMA, theta=THETA)
DPAgent_diff_31 = DPAgent(env_diff_31, gamma=GAMMA, theta=THETA)

policy1_converge, V1_converge, Q1_converge, steps1_converge, iter1_converge = DPAgent_1.value_iteration()
policy12_converge, V12_converge, Q12_converge, steps12_converge, iter12_converge = DPAgent_12.value_iteration()
policy13_converge, V13_converge, Q13_converge, steps13_converge, iter13_converge = DPAgent_13.value_iteration()
policy2_converge, V2_converge, Q2_converge, steps2_converge, iter2_converge = DPAgent_2.value_iteration()
policy3_converge, V3_converge, Q3_converge, steps3_converge, iter3_converge = DPAgent_3.value_iteration()

### use env_1 policy
V12_pi_12, Q12_pi_12, _ = DPAgent_12.policy_evaluation(policy1_converge)
V2_pi_2, Q2_pi_2 = V2_converge, Q2_converge
V12_pi_2_c, Q12_pi_2_c, _ = DPAgent_12.policy_evaluation(policy2_converge)
V2_pi_12_c, Q2_pi_12_c, _ = DPAgent_2.policy_evaluation(policy1_converge)

V13_pi_13, Q13_pi_13, _ = DPAgent_13.policy_evaluation(policy1_converge)
V3_pi_3, Q3_pi_3 = V3_converge, Q3_converge
V13_pi_3_c, Q13_pi_3_c, _ = DPAgent_13.policy_evaluation(policy3_converge)
V3_pi_13_c, Q3_pi_13_c, _ = DPAgent_3.policy_evaluation(policy1_converge)

### Use optimal policy 
# V12_pi_12, Q12_pi_12 = V12_converge, Q12_converge
# V2_pi_2, Q2_pi_2 = V2_converge, Q2_converge
# V12_pi_2_c, Q12_pi_2_c, _ = DPAgent_12.policy_evaluation(policy2_converge)
# V2_pi_12_c, Q2_pi_12_c, _ = DPAgent_2.policy_evaluation(policy12_converge)

# V13_pi_13, Q13_pi_13 = V13_converge, Q13_converge
# V3_pi_3, Q3_pi_3 = V3_converge, Q3_converge
# V13_pi_3_c, Q13_pi_3_c, _ = DPAgent_13.policy_evaluation(policy3_converge)
# V3_pi_13_c, Q3_pi_13_c, _ = DPAgent_3.policy_evaluation(policy13_converge)

# plot_policy_matrix(policy12_converge, DPAgent_12.S, goal_coords=[(6,6), (6,3)],title='Goal at 77', save_path='./temp')
# raise ValueError

def run_QL_exp_lowerbound(env, gamma=0.9, alpha=0.5, epsilon=0.1, max_step=20, \
                          num_eps = 50, num_runs = 10, explore='e-greedy', update_q_lower=False, Q_init = [], Q_lower = []):
    total_dis_r = []
    for i in range(num_runs):
        agent = QLearningAgent(env, gamma=gamma, alpha=alpha, epsilon=epsilon)
        dis_r = agent.run_lowerbound(Q_init=Q_init, Q_lower=Q_lower, max_step=max_step, num_eps=num_eps, explore=explore, update_q_lower=update_q_lower)
        total_dis_r.append(dis_r)
    avg_dis_r = np.average(total_dis_r, axis=0)
    return avg_dis_r

def run_lowerbound_exp(env, DPAgent, source_policy, init_q, source_q, 
                       converge_q, optimal_q, iter, iter_size, update_q_lower, save_path):
    ## run 12 EXP
    Vdiff_lower_dict = {}
    Qdiff_lower_dict = {}
    for i in range(0,21,2):
        Vdiff_lower, Qdiff_lower, n = DPAgent.policy_evaluation(source_policy, max_iter=i)
        Vdiff_lower_dict[i] = Vdiff_lower
        Qdiff_lower_dict[i] = Qdiff_lower

    for k,v in Qdiff_lower_dict.items():
        Q2_pi_1 = v + source_q
        eps_r_V2 = run_QL_exp_lowerbound(env, gamma=GAMMA, alpha=ALPHA, epsilon=EPSILON , max_step=MAX_STEP, \
                                         num_eps = NUM_EPS, num_runs=NUM_RUNS, update_q_lower=update_q_lower, Q_init = init_q, Q_lower = Q2_pi_1,)
        QL_V2_eps_dict[k] = list(eps_r_V2)

    eps_r_V2_optimal = run_QL_exp_lowerbound(env, gamma=GAMMA, alpha=ALPHA, epsilon=EPSILON , max_step=MAX_STEP, \
                                             num_eps = NUM_EPS, num_runs=NUM_RUNS, update_q_lower=update_q_lower, Q_init = init_q, Q_lower = converge_q)
    QL_V2_eps_dict['converge'] = list(eps_r_V2_optimal)

    zero_start = np.zeros([env.nS, env.nA])
    eps_r_V2_zero = run_QL_exp_lowerbound(env, gamma=GAMMA, alpha=ALPHA, epsilon=EPSILON , max_step=MAX_STEP, \
                                             num_eps = NUM_EPS, num_runs=NUM_RUNS, update_q_lower=update_q_lower, Q_init = zero_start, Q_lower = zero_start)
    QL_V2_eps_dict['zero'] = list(eps_r_V2_zero)

    eps_r_V2_optimal = run_QL_exp_lowerbound(env, gamma=GAMMA, alpha=0, epsilon=EPSILON , max_step=MAX_STEP, \
                                             num_eps = NUM_EPS, num_runs=NUM_RUNS, update_q_lower=update_q_lower, Q_init = optimal_q, Q_lower = zero_start)
    QL_V2_eps_dict['optimal'] = list(eps_r_V2_optimal)

    with open(save_path, "w") as outfile:
        json.dump(QL_V2_eps_dict, outfile)

    for k,v in QL_V2_eps_dict.items():
        print(k, sum(v))

    return


GAMMA = 0.9
ALPHA = 0.5
EPSILON = 0.1
TEMP = 0.01
NUM_RUNS = 50
NUM_EPS = 50
MAX_STEP = 20
QL_V2_eps_dict = {}


run_lowerbound_exp(env=env_2, DPAgent=DPAgent_diff_21, source_policy=policy1_converge, source_q=Q12_pi_12, init_q = [], \
                              converge_q=Q2_pi_12_c, optimal_q=Q2_converge, iter=100, iter_size=25, update_q_lower=False, save_path='./results/QL_lower_diff12.json')

run_lowerbound_exp(env=env_2, DPAgent=DPAgent_diff_21, source_policy=policy1_converge, source_q=Q13_pi_13, init_q = Q12_converge, \
                              converge_q=Q2_pi_12_c, optimal_q=Q2_converge, iter=100, iter_size=25, update_q_lower=False, save_path='./results/QL_lower_diff12.json')

# run_lowerbound_exp(env=env_3, DPAgent=DPAgent_diff_31, source_policy=policy1_converge, source_q=Q1_converge, \
#                               converge_q=Q3_pi_1_c, optimal_q=Q3_converge, iter=100, iter_size=5, update_q_lower=False, save_path='./results/QL_lower_diff13.json')
# run_lowerbound_exp(env=env_4, DPAgent=DPAgent_diff_41, source_policy=policy1_converge, source_q=Q1_converge,\
#                               converge_q=Q4_pi_1_c, optimal_q=Q4_converge, iter=100, iter_size=5, update_q_lower=False, save_path='./results/QL_lower_diff14.json')

# run_lowerbound_exp(env=env_2, DPAgent=DPAgent_diff_21, source_policy=policy1_converge, source_q=Q1_converge, converge_q=Q2_pi_1_c, optimal_q=Q2_converge, \
#                               iter=100, iter_size=5, update_q_lower=True, save_path='./results/QL_lower_diff12_updateQLower.json')
# run_lowerbound_exp(env=env_3, DPAgent=DPAgent_diff_31, source_policy=policy1_converge, source_q=Q1_converge, converge_q=Q3_pi_1_c, optimal_q=Q3_converge, \
#                               iter=100, iter_size=5, update_q_lower=True,save_path='./results/QL_lower_diff13_updateQLower.json')
# run_lowerbound_exp(env=env_4, DPAgent=DPAgent_diff_41, source_policy=policy1_converge, source_q=Q1_converge,converge_q=Q4_pi_1_c, optimal_q=Q4_converge, \
#                               iter=100, iter_size=5, update_q_lower=True,save_path='./results/QL_lower_diff14_updateQLower.json')
