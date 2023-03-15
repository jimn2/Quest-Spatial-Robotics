import os
import cv2
import numpy as np
import torch

def crop_cone_view(board, curr=(4,2), ratio=1):
    C, H, W = board.shape
    xs = np.arange(W//2)  + 1
    grad1 = -2 * (H * ratio) / W
    b1 = H * ratio
    ys1 = grad1 * xs + b1
    h, w = curr
    block_char = 'X'
    ascii_val = ord(block_char)
    board[:, h+1:] = ascii_val
    for i in range(W//2):
        val = min(-1, -int(ys1[i])-(H-h-1))
        board[:, val:, i] = ascii_val
        board[:, val:, -i-1] = ascii_val
    board = repair(board, board, ascii_val)
    return board


def _repair(env, h, w, idx, mode='cross'):
    """
        Fix some errors in observation due to discrete space
    """
    if env[h, w] != 0:
        return env
    env[h, w] = idx
    H, W = env.shape
    if mode == 'square':
        for dh in [-1, 0, 1]:
            for dw in [-1, 0, 1]:
                if dh == 0 and dw == 0:
                    continue
                if h + dh < 0 or h + dh == H or w + dw < 0 or w + dw == W:
                    continue
                if env[h+dh, w+dw] == 0:
                    env = _repair(env, h+dh, w+dw, idx)
                if env[h+dh, w+dw] == -1:
                    env[h+dh, w+dw] = 100
    elif mode == 'cross':
#        for dh, dw in zip([-1, +1, 0, 0], [0, 0, -1, +1]):
        for dh, dw in zip([-1, 0, 0], [0, -1, +1]):
            if h + dh < 0 or h + dh == H or w + dw < 0 or w + dw == W:
                continue
            if env[h+dh, w+dw] == 0:
                env = _repair(env, h+dh, w+dw, idx)
            elif env[h+dh, w+dw] == -1: # wall
                env[h+dh, w+dw] = 100
    return env


def repair(board, original_board, ascii_val):
#    mask = -(original_board.sum(0) > 0).astype(int)
    mask = -(board.sum(0) > 0).astype(int)
    H, W = mask.shape
    mask = _repair(mask, H-1, W//2, 1, 'cross')
    # occluded area
    mask = mask <= 0 # 0: empty but visible via slit between wall in a diagonal manner, -1: occulded by wall
    board[:,mask] = ascii_val

    return board


def update_blocking(board, original_board, obstacles=[61, 37]):
    C, H, W = board.shape
    block_char = 'X'
    ascii_val = ord(block_char)
    w = W // 2
#    loc_obstacles = np.sum([((board==code).sum(0) == 3) for code in obstacles], axis=0) > 0
    loc_blocked = (board==ascii_val).sum(0) == board.shape[0]
    loc_empty = (board==0).sum(0) == board.shape[0]
    loc_obstacles = np.logical_and(np.logical_not(loc_empty), np.logical_not(loc_blocked))
    row, col = loc_obstacles.nonzero()
    row = row[::-1]
    col = col[::-1]
    for r, c in zip(row, col):
        if r == 0:
            continue
        if c == W//2:
            board[:, :r, c] = ascii_val
            continue
        if c < W//2:
            xs = np.arange(c+1)
            grad1 = (H  - r) / (W/2 - (c+1))
            grad2 = (H  - (r+1)) / (W/2 - c)
            ys1 = H - 1 - grad1 * (W//2 - xs)
            ys2 = H - 1 - grad2 * (W//2 - xs)
        elif c > W//2:
            xs = np.arange(start=c, stop=W)
            grad1 = (H - r) / (W/2 - c)
            grad2 = (H - (r+1)) / (W/2 - (c+1))
            ys1 = grad1 * (xs-W//2) + H - 1
            ys2 = grad2 * (xs-W//2) + H - 1
        ys1 = np.ceil(ys1.clip(min=0, max=r))
        ys2 = np.ceil(ys2.clip(min=0, max=r))

        for x, y1, y2 in zip(xs, ys1, ys2):
            board[:, int(y1):int(y2), x] = ascii_val
    board = repair(board, original_board, ascii_val)
    return board

def get_obs(h, w, d, maze, size=(5,5), padding_value=-1, cone_view=True, blocking=True, ratio=1):
    H, W = size
    dh1s = [-H+1, +0, -(W//2), -(W//2)]
    dh2s = [+1, H, W-(W//2), W-(W//2)]
    dw1s = [-(W//2), -(W//2), -H+1, +0]
    dw2s = [W-(W//2), W-(W//2), +1, H]

    rotation_factor = [0, 2, 3, 1]
    dh1, dh2 = dh1s[d], dh2s[d]
    dw1, dw2 = dw1s[d], dw2s[d]
    rot = rotation_factor[d]
    obs = padding_obs(h+dh1, h+dh2, w+dw1, w+dw2, maze)
    obs = np.rot90(obs, rotation_factor[d], axes=(-2, -1))
    original_obs = obs.copy()
    if blocking:
        obs = update_blocking(obs, original_obs)
    if cone_view:
        obs = crop_cone_view(obs, (size[0]-1, size[1]//2), ratio=ratio)
    return obs


def padding_obs(h1, h2, w1, w2, maze, padding_value=ord('X')):
    shape = maze.shape
    H, W = shape[-2:]
    if len(shape) == 3:
        C = shape[0]
        obs = np.ones((C, h2-h1, w2-w1)) * padding_value
    else:
        obs = np.ones((h2-h1, w2-w1)) * padding_value

    st_h = -h1 if h1<0 else 0
    st_w = -w1 if w1<0 else 0
    ed_h = H-h2 if h2>H else h2-h1
    ed_w = W-w2 if w2>W else w2-w1

    h1 = max(h1, 0)
    w1 = max(w1, 0)
    h2 = min(h2, H)
    w2 = min(w2, W)

    if len(shape) == 3:
        try:
            obs[:, st_h:ed_h, st_w:ed_w] = maze[:, h1:h2, w1:w2]
        except:
            breakpoint()
    else:
        obs[st_h:ed_h, st_w:ed_w] = maze[h1:h2, w1:w2]
#    obs[:, 4, 2] = 255
    return obs


def get_switch_cost(occupancy):
    H, W = occupancy.shape
    current = (H-1, W//2, 0)
    if type(occupancy) is np.ndarray:
        cost =  300 * np.ones((4, H, W))
    else:
        cost = 300 * torch.ones((4,H,W), device=occupancy.device)
    cost[0, H-1, W//2] = 0

    queue = [current]
    while len(queue) > 0:
        h, w, d = queue[0]
        c = cost[d, h, w]
        queue = queue[1:]
        for dh, dw, dd in [[-1, 0, 0], [1, 0, 1], [0, -1, 2], [0, 1, 3]]:
            add = 0 if d == dd else 1
            if h+dh == H or h+dh < 0 or w+dw == W or w+dw < 0:
                continue
            if occupancy[h+dh, w+dw]:
                new_c = c  + add
                if cost[dd, h+dh, w+dw] > new_c:
                    cost[dd, h+dh, w+dw] = new_c
                    queue.append([h+dh, w+dw, dd])
    
    cost[cost==300] = 0
    return cost
