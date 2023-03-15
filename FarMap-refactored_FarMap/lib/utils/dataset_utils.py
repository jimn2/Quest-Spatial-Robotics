import os
import tqdm
import random
import numpy as np

import torch
from torch import nn
import torch.backends.cudnn as cudnn

import sys
from lib.utils.obs_utils import get_obs
from lib.utils.utils import get_map, get_avg_surprisal


def generate_dataset_from_generator(maze_id, size=(5,5), cone_view=True, blocking=True, ratio=1, num_envs=1, mode='train'):
    maze, _ = get_map(maze_id, num_envs, generation=True)
    rows, cols = (maze.sum(0) == 0).nonzero()
    H, W = maze.shape[1:]
    observations = {}
    ground_truths = {}
    keys = []
    for h, w in tqdm.tqdm(zip(rows, cols), total=len(rows)):
        for d in range(4):
            h = int(h)
            w = int(w)
            obs = get_obs(h, w, d, maze, size=size, padding_value=-1, cone_view=cone_view, blocking=blocking, ratio=ratio)
            num_empty = ((obs==0).sum(0) == len(obs))
            # no reachable location for predicting surprisal
            if num_empty.sum() == 1 and mode =='train':
                continue
            obs = obs.reshape([num_envs, 3] + list(obs.shape[1:]))
            surprisal = obs
            observations[(h,w,d)] = torch.from_numpy(obs.copy()).float()
            ground_truths[(h,w,d)] = torch.zeros((4, obs.shape[1], obs.shape[2]))
            keys.append([h, w, d])
    keys = np.asarray(keys, dtype=int)
    return keys, observations, ground_truths, maze


def generate_dataset(trajs, scores, maze_id, size=(5,5), cone_view=True, blocking=True, ratio=1, num_envs=1, maze=None, generation=False, scale=3, rotate=False):
    if maze is None:
        maze, _ = get_map(maze_id, num_envs, scale, generation, maze_path)
    mazes = []
    rows, cols = (maze.sum(0) == 0).nonzero()
    H, W = maze.shape[1:]
    score_map, used_map = get_avg_surprisal(trajs, scores, maze)
    observations = {}
    keys = []
    ground_truths = {}
    for h, w in tqdm.tqdm(zip(rows, cols), total=len(rows)):
        for d in range(4):
            h = int(h)
            w = int(w)
            obs = get_obs(h, w, d, maze, size=size, padding_value=-1, cone_view=cone_view, blocking=blocking, ratio=ratio)
            mask = get_obs(h, w, d, used_map, size=size, padding_value=0, cone_view=False, blocking=False, ratio=ratio)
            mask[mask ==ord('X')] = 0
            mask = np.logical_and(mask > 0, obs.sum(0) == 0)
            if mask.sum() == 4:
                continue
            obs = obs.reshape([num_envs, 3] + list(obs.shape[1:]))
            surprisal = get_obs(h, w, d, score_map, size=size, padding_value=-1, cone_view=False, blocking=False, ratio=ratio)
            surprisal[np.logical_not(mask)] = -1
            observations[(h,w,d)] = torch.from_numpy(obs.copy()).float()
            # currently, surprisal ground_truth's head direction is absolute direction not relational.

            surprisal = torch.from_numpy(surprisal.copy()).float()
            order = [0, 1, 2, 3]
            if rotate:
                if d == 1:
                    order = [1, 0, 3, 2]
                elif d == 2:
                    order = [3, 2, 0, 1]
                elif d == 3:
                    order = [3, 2, 0, 1]

            ground_truths[(h,w,d)] = surprisal[order]
            keys.append([h, w, d])
    keys = np.asarray(keys, dtype=int)
    return keys, observations, ground_truths, maze


def generate_seq_dataset(trajs, scores, maze_id, size=(5,5), cone_view=True, blocking=True, ratio=1, num_envs=1, maze=None, generation=False, scale=3, rotate=False):
    """
        Generate Sequential Dataset for SPS-LSTM

        key: np array [h, w, d]
        observations: np array [N, 3, size[0], size[1]]
        ground_truths: np array [M, 3, size[0], size[1]], unknown entry has 
    """
    if maze is None:
        maze, _ = get_map(maze_id, num_envs, scale, generation, maze_path)
    mazes = []
    rows, cols = (maze.sum(0) == 0).nonzero()
    H, W = maze.shape[1:]

    score_map, used_map = get_avg_surprisal(trajs, scores, maze)
    observations = {}
    keys = []
    ground_truths = {}
    for h, w in tqdm.tqdm(zip(rows, cols), total=len(rows)):
        for d in range(4):
            h = int(h)
            w = int(w)
            obs = get_obs(h, w, d, maze, size=size, padding_value=-1, cone_view=cone_view, blocking=blocking, ratio=ratio)
            mask = get_obs(h, w, d, used_map, size=size, padding_value=0, cone_view=False, blocking=False, ratio=ratio)
            mask[mask ==ord('X')] = 0
            mask = np.logical_and(mask > 0, obs.sum(0) == 0)
            if mask.sum() == 4:
                continue
            obs = obs.reshape([num_envs, 3] + list(obs.shape[1:]))
            observations[(h,w,d)] = torch.from_numpy(obs.copy()).float()
    for h, w, d in trajs:
        surprisal = get_obs(h, w, d, score_map, size=size, padding_value=-1, cone_view=False, blocking=False, ratio=ratio)
        surprisal[np.logical_not(mask)] = -1
        surprisal = torch.from_numpy(surprisal.copy()).float()
        ground_truths[(h,w,d)] = surprisal
        keys.append([h, w, d])
    keys = np.asarray(keys, dtype=int)
    return keys, observations, ground_truths, maze




if __name__ == '__main__':
    env_id = 14
    num_envs = 1
    scale = 3
    dir_name = f'SPS/DATASET/Maze{env_id}action_x3size15_15step500000R25'
    maze, _ = get_map(env_id, num_envs, scale, generation=True, maze_path='dataset.pkl')
    traj_data, action, score = np.load(os.path.join(dir_name, 'data.npy'), allow_pickle=True)
    data = generate_dataset(traj_data, score, 5, (15, 15), ratio=0.25, num_envs=num_envs, maze=maze, generation=True, rotate=True)
