import os
import cv2
import random
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from lib.utils.obs_utils import get_obs


def draw_triangle(img, loc_x, loc_y, d, H, W, size=1, color=(0, 0, 255)):
    """
        Draw agent / fracture point / recall point on the image.
    """
    if d == 2:
        tri = [[loc_x, loc_y + size/2],
                 [loc_x+size, loc_y],
                 [loc_x+size, loc_y+size]]
    elif d ==3:
        tri = [[loc_x, loc_y],
                [loc_x, loc_y+size],
                [loc_x+size, loc_y + size/2]]
    elif d == 0:
        tri = [[loc_x+size/2, loc_y],
               [loc_x, loc_y+size],
                [loc_x+size, loc_y+size]]
    elif d == 1:
        tri = [[loc_x, loc_y],
                [loc_x+size, loc_y],
                [loc_x+size/2, loc_y+size]]
    tri = np.asarray(tri)
    tri = tri.astype(int)
    img = cv2.drawContours(cv2.UMat(img), [tri], 0, color, -1)
    return cv2.UMat.get(img)


def visualize_obs(observations, predictions, ground_truths, locations=None, env_ids=None, mazes=None, save_path=None):
    """
        Visualize a sequence of observations.
    """
    # obs: (B, C, H, W)
    observations = (observations * 255).cpu().numpy().astype(np.uint8).transpose(0,2,3,1)
    observations[observations==ord('X') ] = 255
    masks = (observations.sum(-1) == 0)

    predictions = (predictions.detach().cpu().numpy()*255)
    ground_truths = (ground_truths.cpu().numpy()*255)
    predictions[predictions < 0] = 0
    ground_truths[ground_truths<0] = 0

    predictions = predictions.astype(np.uint8)
    ground_truths = ground_truths.astype(np.uint8)
    _, H, W, _ = observations.shape
    vis_maps = []

    v_line = np.zeros((2*H+1, 1, 3)).astype(np.uint8)
    v_line[:,:,1] = 255
    h_line = np.zeros((1, W, 3)).astype(np.uint8)
    h_line[:,:,1] = 255
    for i, (obs, mask, pred, gt) in enumerate(zip(observations, masks, predictions, ground_truths)):
        neg_mask = np.logical_not(mask)
        scores = []
        for p, g in zip(pred, gt):
            p = cv2.applyColorMap(p, cv2.COLORMAP_JET)
            p[neg_mask] = obs[neg_mask]
            g = cv2.applyColorMap(g, cv2.COLORMAP_JET)
            g[neg_mask] = obs[neg_mask]
            score = np.concatenate((g, h_line, p), axis=0)
            scores.append(score)
        
        if locations is not None and mazes is not None:
            h, w, d = locations[i]
            if env_ids is None:
                maze = mazes
            elif type(mazes) is list:
                env_id = env_ids[i]
                maze = mazes[env_id]
            else:
                env_id = env_ids[i]
                maze = mazes[3*env_id:3*(env_id+1)]
            full_obs = get_obs(h, w, d, maze, size=(H,W), padding_value=0, cone_view=False, blocking=False)
            full_obs = full_obs.transpose(1,2,0)
            obs = np.concatenate((full_obs, h_line, obs), axis=0)
        else:
            obs = np.concatenate((obs, h_line, obs), axis=0)
        
        vis_map = [np.concatenate((e, v_line), axis=1) for e in ([obs] + scores)]
        vis_map = np.concatenate(vis_map, axis=1)
        line = np.zeros((1, vis_map.shape[1], 3), dtype=np.uint8)
        line[:,:,2] = 255
        vis_map = np.concatenate((vis_map, line), 0)
        vis_maps.append(vis_map)
    vis_map = visualize_tiling(vis_maps)

    H, W = vis_map.shape[:2]
    scale = int(800/W)
    if scale > 1:
        vis_map = cv2.resize(vis_map, (scale*W, scale*H), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(save_path, vis_map)

    return [vis_map]


def visualize(trajs, rel_actions, scores, maze, save_dir='', norm=False):
    plt.plot(list(range(len(scores))), scores)
    plt.savefig(os.path.join(save_dir, 'explore.png'))

    scores = np.asarray(scores)

    distance = np.absolute(trajs[1:] - trajs[:-1])
    distance = distance[:,0] + distance[:,1]
    distance = distance[:,np.newaxis]

    vis_set, inverse = np.unique(np.concatenate((trajs[:-1], trajs[1:], distance), axis=1)[:,3:], axis=0, return_inverse=True)
    if norm:
        scores = (scores - scores.min()) / (scores.max() - scores.min())
    avg_scores = np.stack([scores[inverse==i].mean() for i in range(len(vis_set))])
    
    vis_mazes_all = []
    unique_dist = np.unique(distance)
    for dist in unique_dist:
        vis_mazes = []
        masks = []
        for direction in [0, 1, 2, 3]:
            vis_maze = (maze > 0).astype(float)[:1]
            mask = vis_maze.copy()
            for (h, w, d, dis), s in zip(vis_set, avg_scores):
                if direction == d and dis == dist:
                    vis_maze[:,h, w] = 1-s  #> 0.5
                    mask[:,h,w] = 0.5
            vis_mazes.append(vis_maze)
            masks.append(mask)

        vis_maze1 = np.concatenate(vis_mazes[:2], axis=-1)
        vis_maze2 = np.concatenate(vis_mazes[2:], axis=-1)
        vis_maze = np.concatenate((vis_maze1, vis_maze2), axis=-2)
        masks1 = np.concatenate(masks[:2], axis=-1)
        masks2 = np.concatenate(masks[2:], axis=-1)
        mask = np.concatenate((masks1, masks2), axis=-2)
        mask = mask.transpose(1,2,0).repeat(3, axis=-1)
        vis_maze *= 255
        vis_maze = vis_maze.transpose(1, 2, 0)
        vis_maze = vis_maze.astype(np.uint8)
        vis_maze = cv2.applyColorMap(vis_maze, cv2.COLORMAP_JET)
        vis_maze[mask==1] = 255
        vis_maze[mask==0] = 0
        H, W, C = vis_maze.shape
        scale = int(800/H)
        vis_mazes_all.append(vis_maze)
        vis_maze = cv2.resize(vis_maze, (scale*W, scale*H), interpolation=cv2.INTER_NEAREST)
        if norm:
            cv2.imwrite(os.path.join(save_dir, 'dist_norm_score_{}.png'.format(dist)), vis_maze)
        else:
            cv2.imwrite(os.path.join(save_dir, 'dist{}.png'.format(dist)), vis_maze)
        print('dist', dist)

    L = len(unique_dist)
    N = int(np.ceil(L**0.5))
    M = int(np.ceil(L/N))
    vis_maze = []
    w = 0
    for n in range(N):
        vis_maze_chunk = []
        for m in range(M):
            i = n * M + m
            if i < L:
                temp = vis_mazes_all[i]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 0.250
                temp = cv2.putText(temp, 'd = {}'.format(unique_dist[i]), (int(W*0.4),H//2),
                        font, fontScale, color=(255,0,0), thickness=1)
                vis_maze_chunk.append(temp)
            else:
                vis_maze_chunk.append(np.zeros(vis_mazes_all[0].shape))
            vis_maze_chunk.append(np.zeros((H, 1, C)))
        vis_maze_chunk = np.concatenate(vis_maze_chunk, axis=1)
        vis_maze_chunk = np.concatenate((vis_maze_chunk, np.zeros((1,vis_maze_chunk.shape[1], 3))), axis=0)

        vis_maze.append(vis_maze_chunk)
    vis_maze = np.concatenate(vis_maze, 0)
    H, W, C = vis_maze.shape
    scale = int(800/H)
    if scale > 1:
        vis_maze = cv2.resize(vis_maze, (scale*W, scale*H), interpolation=cv2.INTER_NEAREST)
    if norm:
        cv2.imwrite(os.path.join(save_dir, 'vis_maze_norm_score.png'), vis_maze)
    else:
        cv2.imwrite(os.path.join(save_dir, 'vis_maze.png'), vis_maze)


def visualize_tiling(vis_maps):
    """
        Tiling to the image in case of the size exceeds the maze.
    """
    L = len(vis_maps)
    N = int(np.ceil(L**0.5))
    M = int(np.ceil(L/N))
    vis_map = []
    w = 0
    H, W, C = vis_maps[0].shape
    w_line = np.zeros((H, 1, C), dtype=np.uint8)
    w_line[:,:,2] = 255

    for n in range(N):
        vis_map_chunk = []
        for m in range(M):
            i = n * M + m
            if i < L:
                vis_map_chunk.append(vis_maps[i])
            else:
                vis_map_chunk.append(np.zeros(vis_maps[0].shape))
            vis_map_chunk.append(w_line)
        vis_map_chunk = np.concatenate(vis_map_chunk, axis=1)
        h_line = np.zeros((1, vis_map_chunk.shape[1], C), dtype=np.uint8)
        h_line[:,:,2] = 255
        vis_map_chunk = np.concatenate((vis_map_chunk, h_line), axis=0)
        vis_map.append(vis_map_chunk)
    vis_map = np.concatenate(vis_map, 0)
    return vis_map

def visualize_score(scores):
    # convert confidence score to surprisal
    scores = 1-np.asarray(scores)
    out = cv2.applyColorMap(np.uint8(255 * scores), cv2.COLORMAP_JET)
    out = out.transpose(1,0,2)
    return out


def visualize_discovery(target_maps, env, trajs, subgoals, remaps=[], recalls=[], scores=[], save_dir='./', name='', second=30, is_binary=True, start_idx=-1):
    """
        Generate Video about how agent moves to the environment.
    """
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        
    if not os.path.exists(os.path.join(save_dir, name)):
        os.mkdir(os.path.join(save_dir, name))

    C, H, W = env.shape
    env = env.transpose(1,2,0)
    scale = max(1, int(400 / W))
    vis_maps = []
    white_mask = (np.ones(env.shape) * 255)
    red_mask = np.zeros(env.shape)
    red_mask[:,:,2] = 255
    prev_discovered = np.zeros((H,W, 1)) + 0.25
    if start_idx == -1:
        os.system(f'rm -rf {save_dir}/{name}/*.png')

    if len(scores) > 0:
        colored_score = visualize_score(scores)

    for i, (target, traj, subgoal) in enumerate(zip(target_maps, trajs, subgoals)):
        if i < start_idx:
            continue
        if is_binary:
            target = target > 0
            discovered = target * 0.75 + 0.25  ##  * 200 + 55
            discovered = discovered[:,:,np.newaxis]
            new = discovered - prev_discovered
            vis_map = env * (discovered - new * 0.5) + white_mask * (1-discovered) + red_mask * (new * 0.5)
        else:
            mask = target == 0
            target = cv2.applyColorMap(np.uint8(255 * (1-target)), cv2.COLORMAP_JET)
            target[mask] = 0
            mask = mask[:,:,np.newaxis]
            vis_map = white_mask * mask * 0.5 + env * 0.5  + target * 0.5 
            discovered = None

        vis_map = vis_map.astype(np.uint8)
        vis_map = cv2.resize(vis_map, (scale*W, scale*H), interpolation=cv2.INTER_NEAREST)
        for l in remaps:
            if l+1 < i:
                remap = trajs[l+1]
                vis_map = draw_triangle(vis_map, scale*remap[1], scale*remap[0], remap[2], scale*H, scale*W, size=scale, color=(255,255,0))
        for l in recalls:
            if l+1 < i:
                recall = trajs[l+1]
                vis_map = draw_triangle(vis_map, scale*recall[1], scale*recall[0], recall[2], scale*H, scale*W, size=scale, color=(0,255,255))
        vis_map = draw_triangle(vis_map, scale*subgoal[1], scale*subgoal[0], subgoal[2], scale*H, scale*W, size=scale, color=(0,255,0))
        vis_map = draw_triangle(vis_map, scale*traj[1], scale*traj[0], traj[2], scale*H, scale*W, size=scale)

        
        if len(scores) > 0:
            target = colored_score.copy()
            target[:,i+1:] = 0
            colored_bar = cv2.resize(target, (vis_map.shape[1], int(vis_map.shape[0] * 0.1)))
            vis_map = np.concatenate((vis_map, colored_bar), axis=0)
        cv2.imwrite(os.path.join(save_dir, name, '{:03}.png'.format(i)), vis_map)
        vis_maps.append(vis_map)
        prev_discovered = discovered
    
    if start_idx == -1:
        ratio = min(60, len(trajs) / second)
        os.system(f"ffmpeg -y -r {ratio} -start_number 0 -i '{save_dir}/{name}/%3d.png'  -c:v libopenh264 -pix_fmt yuv420p -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2'  -vcodec libx264  '{save_dir}/video_{name}.mp4'")
        print(f"ffmpeg -y -r {ratio} -start_number 0 -i '{save_dir}/{name}/%3d.png'  -c:v libopenh264 -pix_fmt yuv420p -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2'  -vcodec libx264  '{save_dir}/video_{name}.mp4'")

