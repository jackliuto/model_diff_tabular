# Things added:
# 1. Added 5th Nothing action (not curretly use)
# 2. modified from Gym's frozen lake 

from contextlib import closing
from io import StringIO
from os import path
from typing import List, Optional

import numpy as np

from gym import Env, logger, spaces, utils
from gym.envs.toy_text.utils import categorical_sample
from gym.error import DependencyNotInstalled

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {
    "2x2": ["SF", "GF"],
    "3x3": ["SFF", "FFF", "FFG"],
    "4x4": ["SFFF", "FFFF", "FFFF", "FFFG"],

    "7x7_S00G77": [
        "SFFFFFF",
        "FFFFFFF",
        "FFFFFFF",
        "FFFFFFF",
        "FFFFFFF",
        "FFFFFFF",
        "FFFFFFG",
    ],

    "7x7_S00G77T74": [
        "SFFFFFF",
        "FFFFFFF",
        "FFFFFFF",
        "FFFFFFF",
        "FFFFFFF",
        "FFFFFFF",
        "FFFTFFG",
    ],

    "7x7_S00G77T66": [
        "SFFFFFF",
        "FFFFFFF",
        "FFFFFFF",
        "FFFFFFF",
        "FFFFFFF",
        "FFFFFTF",
        "FFFFFFG",
    ],

    "7x7_S00G74G77": [
        "SFFFFFF",
        "FFFFFFF",
        "FFFFFFF",
        "FFFFFFF",
        "FFFFFFF",
        "FFFFFTF",
        "FFFGFFG",
    ],

    "7x7_S00G66G77": [
        "SFFFFFF",
        "FFFFFFF",
        "FFFFFFF",
        "FFFFFFF",
        "FFFFFFF",
        "FFFFFGF",
        "FFFFFFG",

    ],

}


class GridWorldEnv(Env):

    def __init__(
        self,
        render_mode: Optional[str] = None,
        desc=None,
        map_name="4x4",
        is_slippery=True,
        goal_reward = 1,
        step_cost = 0,
        terminal_states = "GTH",
        slip_prob = 0.2,
        random_start = False,
        reward_matrix = None,

    ):
        if desc is None and map_name is None:
            desc = generate_random_map()
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype="c")
        self.nrow, self.ncol = nrow, ncol = desc.shape

        self.goal_reward = goal_reward
        self.step_cost = step_cost

        def initialize_reward_matrx(desc):
            reward_matrix = np.zeros(desc.shape)
            for i in range(reward_matrix.shape[0]):
                for j in range(reward_matrix.shape[1]):
                    letter = desc[i,j]
                    if letter == b"G":
                        reward_matrix[i][j] = goal_reward
                    elif letter == b"T":
                        reward_matrix[i][j] = 0
                    elif letter == b"W":
                        reward_matrix[i][j] = 0
                    else:
                        reward_matrix[i][j]= step_cost
            return reward_matrix
        
        if type(reward_matrix) == type(None):
            self.reward_matrix = initialize_reward_matrx(self.desc)
        else:
            self.reward_matrix = reward_matrix

        self.terminal_states = terminal_states
        
        nA = 4
        nS = nrow * ncol

        self.nA = nA
        self.nS = nS


        if random_start:
            self.initial_state_distrib = np.array(desc == b"S").astype("float64").ravel() + np.array(desc == b"F").astype("float64").ravel()
        else:
            self.initial_state_distrib = np.array(desc == b"S").astype("float64").ravel()
        self.initial_state_distrib /= self.initial_state_distrib.sum()

        self.P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row * ncol + col

        def inc(row, col, a):
            if self.desc[row][col] == b"W":
                return (row, col)
            if a == LEFT:
                newcol = max(col - 1, 0)
                if self.desc[row, newcol] != b"W":
                    col = newcol
            elif a == DOWN:
                newrow = min(row + 1, nrow - 1)
                if self.desc[newrow, col] != b"W":
                    row = newrow
            elif a == RIGHT:
                newcol = min(col + 1, ncol - 1)
                if self.desc[row, newcol] != b"W":
                    col = newcol
            elif a == UP:
                newrow = max(row - 1, 0)
                if self.desc[newrow, col] != b"W":
                    row = newrow
            # elif a == NOTHING:
            #     pass
            return (row, col)

        def update_probability_matrix(row, col, action):
            oldstate = to_s(row, col)
            oldletter = desc[row, col]
            newrow, newcol = inc(row, col, action)
            newstate = to_s(newrow, newcol)
            # change reward to current state get reward,but not new state get reward
            newletter = desc[newrow, newcol]
            # newletter = desc[row, col]
            terminated = bytes(newletter) in str.encode(self.terminal_states)
            # if newletter == b"G":
            #     reward = self.goal_reward
            # elif newletter == b"W":
            #     reward = 0
            # else:
            #     reward = self.step_cost
            # # reward = float(newletter == b"G")
            if bytes(oldletter) in str.encode(self.terminal_states):
                reward = 0
            else:
                reward = self.reward_matrix[newrow, newcol]
            return newstate, reward, terminated

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(self.nA):
                    li = self.P[s][a]
                    letter = desc[row, col]
                    if letter in str.encode(self.terminal_states):
                        li.append((1.0, s, 0, True)) ## terminal state have
                        # li.append((1.0, *update_probability_matrix(row, col, a))) ## none-terminal states 
                    else:
                        if is_slippery:
                            # for b in [(a - 1) % 4, a, (a + 1) % 4]:
                            for b in [0,1,2,3]:
                                if b == a:
                                    ap = 1.0 - slip_prob
                                else:
                                    ap = slip_prob/3.0
                                li.append(
                                    (slip_prob, *update_probability_matrix(row, col, b))
                                )
                                # li.append(
                                #     (1.0 / 3.0, *update_probability_matrix(row, col, b))
                                # )
                        else:
                            li.append((1.0, *update_probability_matrix(row, col, a)))

        self.observation_space = spaces.Discrete(nS)
        self.action_space = spaces.Discrete(nA)

        self.render_mode = render_mode

        # pygame utils
        self.window_size = (min(64 * ncol, 512), min(64 * nrow, 512))
        self.cell_size = (
            self.window_size[0] // self.ncol,
            self.window_size[1] // self.nrow,
        )
        self.window_surface = None
        self.clock = None
        self.hole_img = None
        self.cracked_hole_img = None
        self.ice_img = None
        self.elf_images = None
        self.goal_img = None
        self.start_img = None


    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, t = transitions[i]
        self.s = s
        self.lastaction = a

        if self.render_mode == "human":
            self.render()
        return (int(s), r, t, False, {"prob": p})

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.s = categorical_sample(self.initial_state_distrib, self.np_random)
        self.lastaction = None

        if self.render_mode == "human":
            self.render()
        return int(self.s), {"prob": 1}


    @staticmethod
    def _center_small_rect(big_rect, small_dims):
        offset_w = (big_rect[2] - small_dims[0]) / 2
        offset_h = (big_rect[3] - small_dims[1]) / 2
        return (
            big_rect[0] + offset_w,
            big_rect[1] + offset_h,
        )

    def _render_text(self):
        desc = self.desc.tolist()
        outfile = StringIO()

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = [[c.decode("utf-8") for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write(f"  ({['Left', 'Down', 'Right', 'Up', 'Nothing'][self.lastaction]})\n")
        else:
            outfile.write("\n")
        outfile.write("\n".join("".join(line) for line in desc) + "\n")

        with closing(outfile):
            return outfile.getvalue()

    def close(self):
        if self.window_surface is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()


