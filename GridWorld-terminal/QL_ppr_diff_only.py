from collections import defaultdict
import numpy as np
import copy
import json

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

V1_pi_1, Q1_pi_1 = V1_converge, Q1_converge
V2_pi_2, Q2_pi_2 = V2_converge, Q2_converge
V1_pi_2_c, Q1_pi_2_c, _ = DPAgent_1.policy_evaluation(policy2_converge)
V2_pi_1_c, Q2_pi_1_c, _ = DPAgent_2.policy_evaluation(policy1_converge)

V1_pi_1, Q1_pi_1 = V1_converge, Q1_converge
V3_pi_3, Q3_pi_3 = V3_converge, Q3_converge
V1_pi_3_c, Q1_pi_3_c, _ = DPAgent_1.policy_evaluation(policy3_converge)
V3_pi_1_c, Q3_pi_1_c, _ = DPAgent_3.policy_evaluation(policy1_converge)

V1_pi_1, Q1_pi_1 = V1_converge, Q1_converge
V4_pi_4, Q4_pi_4 = V4_converge, Q4_converge
V1_pi_4_c, Q1_pi_4_c, _ = DPAgent_1.policy_evaluation(policy4_converge)
V4_pi_1_c, Q4_pi_1_c, _ = DPAgent_4.policy_evaluation(policy1_converge)


def run_QL_exp_ppr(env, gamma=0.9, alpha=0.5, epsilon=0.1, psi=0.5, upsilon=0.95, max_step=20, \
                          num_eps = 50, num_runs = 10, explore='ppr', Q_heur = [], Q_optimal = []):
    total_dis_r = []
    for i in range(num_runs):
        agent = QLearningAgent(env, gamma=gamma, alpha=alpha, epsilon=epsilon)
        if len(Q_optimal) != 0:
            agent.warm_start_q(Q_optimal)
        dis_r = agent.run_ppr(Q_heur=Q_heur, max_step=max_step, num_eps=num_eps, psi=psi, upsilon=upsilon, \
                              explore=explore)
        total_dis_r.append(dis_r)
    avg_dis_r = np.average(total_dis_r, axis=0)
    return avg_dis_r

def run_ppr_exp(env, DPAgent, source_policy, source_q, psi, upsilon,
                    converge_q, optimal_q, iter, iter_size, save_path):
    ## run 12 EXP
    Vdiff_lower_dict = {}
    Qdiff_lower_dict = {}
    for i in range(0,101,5):
        Vdiff_lower, Qdiff_lower, n = DPAgent.policy_evaluation(source_policy, max_iter=i)
        Vdiff_lower_dict[i] = Vdiff_lower
        Qdiff_lower_dict[i] = Qdiff_lower

    for k,v in Qdiff_lower_dict.items():
        Q2_pi_1 = v ## + source_q ## del if only idff
        eps_r_V2 = run_QL_exp_ppr(env, gamma=GAMMA, alpha=ALPHA, epsilon=EPSILON, psi=psi, upsilon=upsilon, max_step=MAX_STEP, \
                                         num_eps = NUM_EPS, num_runs=NUM_RUNS, Q_heur = Q2_pi_1,)
        QL_V2_eps_dict[k] = list(eps_r_V2)

    eps_r_V2_optimal = run_QL_exp_ppr(env, gamma=GAMMA, alpha=ALPHA, epsilon=EPSILON, psi=psi, upsilon=upsilon, max_step=MAX_STEP, \
                                             num_eps = NUM_EPS, num_runs=NUM_RUNS, Q_heur = converge_q)
    QL_V2_eps_dict['converge'] = list(eps_r_V2_optimal)

    zero_start = np.zeros([env.nS, env.nA])
    eps_r_V2_zero = run_QL_exp_ppr(env, gamma=GAMMA, alpha=ALPHA, epsilon=EPSILON, psi=psi, upsilon=upsilon, max_step=MAX_STEP, \
                                             num_eps = NUM_EPS, num_runs=NUM_RUNS, Q_heur = zero_start)
    QL_V2_eps_dict['zero'] = list(eps_r_V2_zero)

    eps_r_V2_optimal = run_QL_exp_ppr(env, gamma=GAMMA, alpha=0, epsilon=EPSILON, psi=psi, upsilon=upsilon, max_step=MAX_STEP, \
                                             num_eps = NUM_EPS, num_runs=NUM_RUNS, Q_heur = zero_start, Q_optimal=optimal_q)
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
NUM_RUNS = 10
NUM_EPS = 50
MAX_STEP = 50
PPR_PSI = 1
PPR_UPSILON = 0.95
QL_V2_eps_dict = {}

run_ppr_exp(env=env_2, DPAgent=DPAgent_diff_21, source_policy=policy1_converge, source_q=Q1_converge, psi=PPR_PSI, upsilon=PPR_UPSILON, \
                              converge_q=Q2_pi_1_c, optimal_q=Q2_converge, iter=100, iter_size=5, save_path='./results/QL_ppr_diff12.json')
run_ppr_exp(env=env_3, DPAgent=DPAgent_diff_31, source_policy=policy1_converge, source_q=Q1_converge, psi=PPR_PSI, upsilon=PPR_UPSILON, \
                              converge_q=Q3_pi_1_c, optimal_q=Q3_converge, iter=100, iter_size=5, save_path='./results/QL_ppr_diff13.json')
run_ppr_exp(env=env_4, DPAgent=DPAgent_diff_41, source_policy=policy1_converge, source_q=Q1_converge, psi=PPR_PSI, upsilon=PPR_UPSILON, \
                              converge_q=Q4_pi_1_c, optimal_q=Q4_converge, iter=100, iter_size=5, save_path='./results/QL_ppr_diff14.json')

