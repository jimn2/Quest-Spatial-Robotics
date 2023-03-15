import os
import time
import tqdm
import pickle
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import Normalize, Compose, Resize, ToTensor

from lib.utils import *


class SPSDataset(Dataset):
    def __init__(self, path=[], size=(-1,-1), norm=True, score_norm=False, action_norm=False, avg_duplicate=False, discrete_score=True, mode='all', env_generation=False, ratio=0.1):
        super(SPSDataset, self).__init__()
        self.norm = norm
        self.size = size
        self.avg_duplicate = avg_duplicate
        self.mode = mode
        self.score_norm = score_norm

        self.load_data(path)

        self.mean = torch.tensor([0,0,0])
        self.std = torch.tensor([1,1,1])
        if self.norm:
            self.observations = self.observations / 255
        
        if score_norm:
            self.scores = (self.scores - self.scores.min())/(self.scores.max() - self.scores.min())

        if discrete_score:
            self.scores = (100 * self.scores).long()
            self.scores[self.scores > 99] = 99

        if action_norm:
            self.actions[:,0] = 1 + self.actions[:,0] * 2 / (size[0]-1)
            self.actions[:,1] = self.actions[:,1] / (size[1] // 2)
            self.actions[:,2] = self.actions[:,2] /  3
        else:
            self.actions = self.actions.long()
            actions = F.one_hot(self.actions[:,0]+14, 35)
            actions += F.one_hot(self.actions[:,1]+16+7, 35)
            actions += F.one_hot(35-self.actions[:,2]-1, 35)
            self.actions = actions.float()

        N = int(len(self.scores) * ratio)
        if self.mode == 'val':
            self.scores = self.scores[-N:]
            self.actions = self.actions[-N:]
            self.trajs = self.trajs[-N:]
            self.observations = self.observations[-N:]
        elif self.mode == 'train':
            self.scores = self.scores[:-N]
            self.actions = self.actions[:-N]
            self.trajs = self.trajs[:-N]
            self.observations = self.observations[:-N]
        self.mean = torch.tensor([0,0,0])
        self.std = torch.tensor([1,1,1])


    def load_data(self, dir_names):
        if type(dir_names) is str:
            dir_names = [dir_names]
        trajs = []
        actions = []
        scores = []
        observations = []
        for dir_name in dir_names:
            with open(os.path.join(dir_name, 'unique_train_data.pkl'), 'rb') as f:
                traj_data, obs, action, score = pickle.load(f)
            obs = obs[:,::-1]
            trajs.append(traj_data)
            observations.append(obs)
            actions.append(action)
            scores.append(score)
        self.trajs = torch.from_numpy(np.concatenate(trajs))
        self.actions = torch.from_numpy(np.concatenate(actions)).float()
        self.observations = torch.from_numpy(np.concatenate(observations)).float()
        self.scores = torch.from_numpy(np.concatenate(scores)).float()

    
    def analyze_diversity(self):
        data = torch.cat((self.observations.flatten(1).float(), self.actions.float()), -1)
        if self.avg_duplicate or True:
            data = data.numpy()
            unique, indices = np.unique(data, axis=0, return_index=True)
            self.scores = self.scores[indices]
            self.observations = self.observations[indices]
            self.trajs = self.trajs[indices]
            self.actions = self.actions[indices]

            with open(os.path.join('SPS', 'unique_train_data.pkl'), 'wb') as f:
                pickle.dump((self.trajs.numpy(), self.observations.numpy(), self.actions.numpy(), self.scores.numpy()), f)

        else:
            unique = data.unique(dim=0)
        l1 = len(data)
        l2 = len(unique)
        print('Unique: {}/{} = {:.3}'.format(l2, l1, l2/l1))


    def __getitem__(self, idx):
        state = self.observations[idx]
        traj = self.trajs[idx]
        action = self.actions[idx]
        scores = self.scores[idx]
        if self.size[0] > 0:
            state = F.interpolate(state.unsqueeze(0), size=self.size, mode='nearest')[0]
        return state, action, scores, traj


class SPSDataset_v2(Dataset):
    def __init__(self, path=[], size=(-1,-1), norm=True, score_norm=True, action_norm=False, avg_duplicate=False, pred_highest=True, discrete_score=False, mode='all', ratio=0.1, num_envs=1, env_generation=False, rotate=True):
        super(SPSDataset_v2, self).__init__()
        self.norm = norm
        self.size = size
        self.avg_duplicate = avg_duplicate
        self.pred_highest = pred_highest
        self.mode = mode
        self.rotate = rotate


        self.num_envs = num_envs
        self.mean = torch.tensor([0,0,0])
        self.std = torch.tensor([1,1,1])
        
        if env_generation:
            locations, observations, ground_truths, maze = generate_dataset_from_generator(0, (15, 15), ratio=0.25, num_envs=num_envs)
        else:
            locations, observations, ground_truths, maze = self.load_data(path, score_norm)
        env_ids = []
        for i, location in enumerate(locations):
            env_ids += [i for _ in location]
        locations = np.concatenate(locations)

        self.env_ids = np.asarray(env_ids)
        self.locations = locations
        self.observations = observations
        self.scores = ground_truths
        self.maze = maze
        self.ratio = ratio

    def shuffle(self, order):
        self.locations = self.locations[order]
        self.env_ids = self.env_ids[order]
        N = int(len(self.locations) * self.ratio)
        if self.mode == 'train':
            self.locations = self.locations[:-N]
            self.env_ids = self.env_ids[:-N]
        if self.mode == 'val':
            self.locations = self.locations[-N:]
            self.env_ids = self.env_ids[-N:]

    def __len__(self):
        return len(self.locations) * self.num_envs


    def load_data(self, dir_names, score_norm):
        if type(dir_names) is str:
            dir_names = [dir_names]
        trajs = []
        actions = []
        scores = []
        observations = []
        locations = []
        ground_truths = []
        mazes = []
        print(dir_names)
        for dir_name in dir_names:
            file_name = os.path.join(dir_name, f'sps_data_{self.num_envs}.pkl')
            if self.rotate:
                file_name = file_name.replace('sps_data', 'sps_rotate_data')
            if score_norm:
                file_name = file_name.replace('_data_', '_data_norm_score')
            print(file_name)
            if os.path.exists(file_name):
                with open(file_name, 'rb') as f:
                    location, observation, ground_truth, maze = pickle.load(f)
            else:
                print("LOAD")
                env_id, scale = dir_name.split('Maze')[1].split('_x')
                env_id = int(env_id.replace('action','').replace('loc',''))
                scale = int(scale.split('size')[0])
                maze, _ = get_map(env_id, self.num_envs, scale, generation=True, maze_path='dataset.pkl')

                traj_data, action, score = np.load(os.path.join(dir_name, 'data.npy'), allow_pickle=True)
                if score_norm:
                    score = (score - score.min())/(score.max() - score.min())
                data = generate_dataset(traj_data, score, 5, (15, 15), ratio=0.25, num_envs=self.num_envs, maze=maze, generation=True, rotate=True)
                location, observation, ground_truth, maze = data
                with open(file_name, 'wb') as f:
                    pickle.dump(data, f)

            mazes.append(maze)
            locations.append(location)
            ground_truths.append(ground_truth)
            observations.append(observation)

        return locations, observations, ground_truths, mazes

    def __getitem__(self, idx):
        env_id = self.env_ids[idx]
        loc = self.locations[idx]
        env = idx // len(self.locations)
        state = self.observations[env_id][tuple(loc)][env]
        scores = self.scores[env_id][tuple(loc)]
        if self.size[0] > 0:
            state = F.interpolate(state.unsqueeze(0), size=self.size, mode='nearest')[0]
            scores = F.interpolate(scores.unsqueeze(0), size=self.size, mode='bilinear', align_corners=False)[0]
        if self.norm:
            state = state / 255
        if self.pred_highest:
            scores = scores.view(-1).argmax()
        return state, [], scores, loc, env_id
