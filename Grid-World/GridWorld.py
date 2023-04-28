import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import Tuple

from utils import *


def get_next_state(a: str, s: int, S: np.array): 
    """ 
    Function that returns the next state's coordinates given an action and a state 
    """
    # Getting the current indexes 
    s_index = np.where(S == s)
    s_row = s_index[0][0]
    s_col = s_index[1][0]

    # Defining the indexes of the next state
    next_row = s_row 
    next_col = s_col

    if a == 'up':
        next_row = s_row - 1
        next_col = s_col
    elif a == 'down':
        next_row = s_row + 1
        next_col = s_col
    elif a == 'left':
        next_row = s_row
        next_col = s_col - 1
    elif a == 'right':
        next_row = s_row
        next_col = s_col + 1

    return next_row, next_col

def bellman_value(
    s: int, 
    S: np.array, 
    P: dict, 
    G: np.array, 
    V: np.array, 
    gamma: float = 0.9
    ) -> Tuple: 
    """
    Calculates the Belman equation value for the given state
    """
    # Extracting all the available actions for the given state
    actions = P[s]

    # Placeholder to hold the sum 
    sum = 0
    for action in actions: 
        # Extracting the probability of the given action 
        prob = actions[action]

        # Getting the next states indexes
        next_row, next_col = get_next_state(action, s, S)

        # Extracting the expected reward 
        reward = G[next_row, next_col]

        # Extracting the value of the next state
        value_prime = V[next_row, next_col]

        # Adding to the sum 
        sum += prob * (reward + gamma * value_prime)

    return sum

def get_max_return(s: int, S: np.array, P: dict, G: np.array, V: np.array, gamma: float = 0.9) -> Tuple:
    """
    Returns the best action and the Bellman's value for the given state
    """
    # Extracting all the available actions for the given state
    actions = P[s]

    # Placeholder to hold the best action and the max return 
    best_action = None
    max_return = -np.inf

    for action in actions: 
        # Getting the probability of the action 
        prob = actions[action]

        # Getting the next states indexes
        next_row, next_col = get_next_state(action, s, S)

        # Extracting the expected reward 
        reward = G[next_row, next_col]

        print(reward)

        # Extracting the value of the next state
        value_prime = V[next_row, next_col]

        # Calculating the return 
        _return = prob * (reward + gamma * value_prime)

        # Checking if the return is greater than the current max return
        if _return > max_return:
            best_action = action
            max_return = _return

    return best_action, max_return

def update_value(s, S, P, G, V, gamma) -> float:
    """
    Updates the value function for the given state
    """
    # Getting the indexes of s in S 
    s_index = np.where(S == s)
    s_row = s_index[0][0]
    s_col = s_index[1][0]

    # Getting the best action and the Bellman's value 
    _, max_return = get_max_return(s, S, P, G, V, gamma)

    # Rounding up the bellman value
    max_return = np.round(max_return, 2)

    # Updating the value function with a rounded value
    V[s_row, s_col] = max_return

    return max_return

def value_iteration(
    S: np.array, 
    P: np.array, 
    G: np.array, 
    V: np.array, 
    gamma: float = 0.9, 
    epsilon: float = 0.000001,
    n_iter: int = None 
    ) -> None: 
    """
    Function that performs the value iteration algorithm

    The function updates the V matrix inplace 
    """
    # Iteration tracker 
    iteration = 0

    # print(n_iter)

    # Iterating until the difference between the value functions is less than epsilon 
    iterate = True
    while iterate:
        # Placeholder for the maximum difference between the value functions 
        delta = 0
        
        # Updating the iteration tracker
        iteration += 1 
        # Iterating over the states 
        for s in S.flatten():
            # Getting the indexes of s in S 
            s_index = np.where(S == s)
            s_row = s_index[0][0]
            s_col = s_index[1][0]

            # Saving the current value for the state
            v_init = V[s_row, s_col].copy()

            # Updating the value function
            v_new = update_value(s, S, P, G, V, gamma)

            print(v_init, v_new)
            raise ValueError

            # Updating the delta 
            delta = np.max([delta, np.abs(v_new - v_init)])

            if (delta < epsilon) and (n_iter is None): 
                iterate = False
                break

        if (n_iter is not None) and (iteration >= n_iter):
            iterate = False

    # Printing the iteration tracker
    print(f"Converged in {iteration} iterations")

    return None


def update_policy(S, P, V): 
    """
    Function that updates the policy given the value function 
    """
    # Iterating over the states 
    for s in S.flatten(): 
        # Listing all the actions 
        actions = P[s]

        # For each available action, getting the Bellman's value
        values = {}
        for action in actions.keys():
            # Getting the next state indexes
            next_row, next_col = get_next_state(action, s, S)

            # Saving the value function of that nex t state
            values[action] = V[next_row, next_col]
        
        # Extracting the maximum key value of the values dictionary 
        max_value = max(values.values())        

        # Leaving the keys that are equal to the maximum value
        best_actions = [key for key in values if values[key] == max_value]

        # Getting the length of the dictionary 
        length = len(values)

        # Creating the final dictionary with all the best actions in it 
        p_star = {}
        for action in best_actions:
            p_star[action] = 1/length

        # Updating the policy 
        P[s] = p_star

def init_gridworld(n: int, step_reward: float) -> Tuple: 
    # Creating the reward matrix 
    G = np.zeros((n, n)) 
    G[G == 0] = step_reward

    # Initiating the empty value array 
    V = np.zeros((n, n))

    # Creating the state array
    S = np.arange(0, n * n).reshape(n, n)

    # Initializing the policy
    P = init_policy(S)

    return S, P, G, V

# def add_random_goal(G: np.array, goal_reward) -> np.array: 
#     # Extracting the shape of the matrix 
#     n = G.shape[0]
    
#     # Getting random coords
#     x = np.random.randint(n)
#     y = np.random.randint(n)

#     # Adding the goal value inplace
#     G[x, y] = goal_reward

#     # Returning the goal coordinates
#     return x, y

def add_goal(G: np.array, goal_reward, x: int, y: int) -> np.array: 
    # Adding the goal value inplace
    G[x, y] = goal_reward


# Defining the number of blocks of a n x n grid 
n = 5

# Defining the value for the hole and the goal
goal = 10
step = -1

# Initiating an empty dataframe of size n x n
G = np.ones((n,n))

# Defining the coordinates of the goal
goal_coords = [(n-1, n-1)]
#goal_coords = [(1, 2)]
# Adding the goal values to the center and the cornersn_iter
S = np.arange(0, n*n).reshape(n, n)

# plot_matrix(S, goal_coords, title='State space')

P =  init_policy(S)

# plot_policy_matrix(P, S, goal_coords, title='Policy matrix')

V = np.zeros((n, n))


# plot_matrix(V, goal_coords, title='Value function', annotate_goal=False)

# plot_policy_value_matrix(P, S, V, goal_coords, title='Policy value matrix')

value_iteration(S, P, G, V, epsilon=0.0001, gamma=0.8)

# plot_matrix(V, goal_coords, annotate_goal=False, title='Value function')

update_policy(S, P, V)



plot_policy_value_matrix(P, S, V, goal_coords, img_width=11, img_height=11, annotate_goal=False)

