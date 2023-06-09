import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import Tuple
from collections import defaultdict
import copy

def array_index_to_matplot_coords(i: int, j: int, n_cols: int) -> Tuple[int, int]:
    """Converts an array index to a matplot coordinate"""
    x = j
    y = n_cols - i - 1
    return x, y


def plot_matrix(
    env,
    M: np.array,
    goal_coords: list = [],
    img_width: int = 5,
    img_height: int = 5,
    title: str = None,
    annotate_goal: bool = True,
    save_path: '' = str,
    ) -> None:
    """
    Plots a matrix as an image.
    """
    height, width = env.nrow, env.ncol
    M = M.reshape(height, width)

    fig = plt.figure(figsize=(img_width, img_width))
    ax = fig.add_subplot(111, aspect='equal')

    max_v = np.max(M)
    min_v = np.min(M)

    for y in range(height):
        for x in range(width):
            # By default, the (0, 0) coordinate in matplotlib is the bottom left corner,
            # so we need to invert the y coordinate to plot the matrix correctly
            matplot_x, matplot_y = array_index_to_matplot_coords(x, y, height)

            # If there is a tuple of (x, y) in the goal_coords list, we color the cell gray
            if (x, y) in goal_coords:
                ax.add_patch(matplotlib.patches.Rectangle((matplot_x - 0.5, matplot_y - 0.5), 1, 1, facecolor='gray'))
                if annotate_goal:
                    ax.annotate(str(round(M[x][y],2)), xy=(matplot_x, matplot_y), ha='center', va='center')
            else:
                if M[x][y] >= 0:
                    ax.add_patch(matplotlib.patches.Rectangle((matplot_x - 0.5, matplot_y - 0.5), 1, 1, facecolor='green', alpha=max(0.0, M[x][y]/max_v)))
                else:
                    ax.add_patch(matplotlib.patches.Rectangle((matplot_x - 0.5, matplot_y - 0.5), 1, 1, facecolor='red', alpha=max(0.0, M[x][y]/min_v)))
                ax.annotate(str(round(M[x][y],2)), xy=(matplot_x, matplot_y), ha='center', va='center')

    offset = .5
    ax.set_xlim(-offset, width - offset)
    ax.set_ylim(-offset, height - offset)

    ax.hlines(y=np.arange(height+1)- offset, colors='black', xmin=-offset, xmax=width-offset)
    ax.vlines(x=np.arange(width+1) - offset, colors='black', ymin=-offset, ymax=height-offset)

    plt.title(title)

    # plt.show()
    plt.savefig(save_path, bbox_inches = 'tight')


def plot_policy_matrix(P: dict, S:np.array, goal_coords: list = [], img_width: int = 5, img_height: int = 5, title: str = None, save_path: '' = str) -> None:
    """
    Plots the policy matrix out of the dictionary provided; The dictionary values are used to draw the arrows
    """
    height, width = S.shape

    fig = plt.figure(figsize=(img_width, img_width))
    ax = fig.add_subplot(111, aspect='equal')
    for y in range(height):
        for x in range(width):
            matplot_x, matplot_y = array_index_to_matplot_coords(x, y, height)

            # If there is a tuple of (x, y) in the goal_coords list, we color the cell gray
            if (x, y) in goal_coords:
                ax.add_patch(matplotlib.patches.Rectangle((matplot_x - 0.5, matplot_y - 0.5), 1, 1, facecolor='gray'))

            else:
                # Adding the arrows to the plot
                if P[S[x, y]][3] > 0: # up
                    plt.arrow(matplot_x, matplot_y, 0, 0.3, head_width = 0.05, head_length = 0.05)
                if P[S[x, y]][1] > 0: # down
                    plt.arrow(matplot_x, matplot_y, 0, -0.3, head_width = 0.05, head_length = 0.05)
                if P[S[x, y]][0] > 0: # left
                    plt.arrow(matplot_x, matplot_y, -0.3, 0, head_width = 0.05, head_length = 0.05)
                if P[S[x, y]][2] > 0: # right
                    plt.arrow(matplot_x, matplot_y, 0.3, 0, head_width = 0.05, head_length = 0.05)
                if P[S[x, y]][4] > 0: # stay
                    ax.add_patch(plt.Circle((matplot_x, matplot_y), 0.1, fill=False))


    offset = .5
    ax.set_xlim(-offset, width - offset)
    ax.set_ylim(-offset, height - offset)

    ax.hlines(y=np.arange(height+1)- offset, colors='black', xmin=-offset, xmax=width-offset)
    ax.vlines(x=np.arange(width+1) - offset, colors='black', ymin=-offset, ymax=height-offset)

    plt.title(title)

    plt.savefig(save_path, bbox_inches = 'tight')

def plot_line_dict(line_dict, save_path, title):
    for k, v in line_dict.items():
        plt.plot(v, label=k)
    plt.legend()
    plt.title('{} Num Eps vs Average Eps Length'.format(title))
    plt.savefig(save_path, bbox_inches = 'tight')
    plt.close()

# def policy_evaluation(env, policy, gamma=1, theta=1e-8):
#     V = np.zeros(env.nS)
#     while True:
#         delta = 0
#         for s in range(env.nS):
#             Vs = 0
#             for a, action_prob in enumerate(policy[s]):
#                 for prob, next_state, reward, done in env.P[s][a]:
#                     Vs += action_prob * prob * (reward + gamma * V[next_state])
#             delta = max(delta, np.abs(V[s]-Vs))
#             V[s] = Vs
#         if delta < theta:
#             break
#     return V

# def q_from_v(env, V, s, gamma=1):
#     q = np.zeros(env.nA)
#     for a in range(env.nA):
#         for prob, next_state, reward, done in env.P[s][a]:
#             q[a] += prob * (reward + gamma * V[next_state])
#     return q

# def policy_improvement(env, V, gamma=1):
#     policy = np.zeros([env.nS, env.nA]) / env.nA
#     for s in range(env.nS):
#         q = q_from_v(env, V, s, gamma)

#         # OPTION 1: construct a deterministic policy
#         # policy[s][np.argmax(q)] = 1

#         # OPTION 2: construct a stochastic policy that puts equal probability on maximizing actions
#         best_a = np.argwhere(q==np.max(q)).flatten()
#         policy[s] = np.sum([np.eye(env.nA)[i] for i in best_a], axis=0)/len(best_a)

#     return policy

# def policy_iteration(env, gamma=1, theta=1e-8):
#     policy = np.ones([env.nS, env.nA]) / env.nA
#     while True:
#         V = policy_evaluation(env, policy, gamma, theta)
#         new_policy = policy_improvement(env, V)

#         # OPTION 1: stop if the policy is unchanged after an improvement step
#         if (new_policy == policy).all():
#             break

#         # OPTION 2: stop if the value function estimates for successive policies has converged
#         # if np.max(abs(policy_evaluation(env, policy) - policy_evaluation(env, new_policy))) < theta*1e2:
#         #    break;

#         policy = copy.copy(new_policy)
#     return policy, V

# def value_iteration(env, gamma=1, theta=1e-8):
#     V = np.zeros(env.nS)
#     for i in range(10):
#     # while True:
#         delta = 0
#         for s in range(env.nS):
#             v = V[s]
#             V[s] = max(q_from_v(env, V, s, gamma))
#             delta = max(delta,abs(V[s]-v))
#         if delta < theta:
#             break
#     policy = policy_improvement(env, V, gamma)
#     return policy, V
