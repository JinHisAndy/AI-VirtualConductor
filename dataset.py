import os
import tqdm
import numpy as np
import torch.utils.data as Data
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader

from utils_pose import show_pose

from utils_audio import feature_extraction


class TestDataset(Data.Dataset):
    def __init__(self, test_samles_dir):
        self.test_samles_dir = test_samles_dir
        self.name_list = os.listdir(test_samles_dir)

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        name = self.name_list[index]
        try:
            feature = np.load('test/cache/' + name + '.npy')
            print('\tusing cached feature: cache/' + name + '.npy')
        except:
            print('\textracting features for', name, '...')
            feature = feature_extraction(self.test_samles_dir + name)
            np.save('test/cache/' + name + '.npy', feature)
            print('\tfeature cached in: test/cache/' + name + '.npy')
        return feature, name


class ConductorDataset(Data.Dataset):
    def __init__(self, sample_length, dataset_dir, sample_limit=None, mode='high'):
        self.dataset_dir = dataset_dir
        self.sample_length = sample_length
        self.name_list = os.listdir(dataset_dir)
        self.sample_idx = []
        self.data = dict()
        pbar = tqdm.tqdm(range(len(self.name_list)))
        for i in pbar:
            if sample_limit and i > sample_limit:
                break
            name = self.name_list[i]
            if mode == 'low':
                pose_seq = np.load(self.dataset_dir + name + '\\' + 'low-pass-results-norm.npy')[200:-200, :]
            elif mode == 'high':
                pose_seq = np.load(self.dataset_dir + name + '\\' + 'high-pass-results.npy')[200:-200, :]
            else:
                raise Exception('invalid mode!')
            audio_feature = np.load(self.dataset_dir + name + '\\' + 'audiofeature-30fps-light.npy')[:, 200:-200]

            min_length = min(np.shape(audio_feature)[1], np.shape(pose_seq)[0])
            sample_num = int(min_length / self.sample_length)

            pbar.set_description(
                'loading training sample {}, total frames: {}, splited to {} clips, name: {}'.format(i, min_length,
                                                                                                     sample_num, name))

            # load dataset to RAM
            self.data[name] = {'pose': pose_seq.reshape([-1, 20]).astype(np.float32),
                               'audio': audio_feature.transpose().astype(np.float32)}

            for j in range(sample_num):
                self.sample_idx.append([i, j * self.sample_length, (j + 1) * self.sample_length])

    def __len__(self):
        return len(self.sample_idx)

    def __getitem__(self, index):
        idx, start, end = self.sample_idx[index]
        name = self.name_list[idx]
        audio_feature = self.data[name]['audio']
        pose_seq = self.data[name]['pose']

        audio_feature[:, 0] /= 4000
        audio_feature[:, 1] /= 4000
        audio_feature[:, 5] /= 300
        # audio_feature[390:410, :] /= 10
        # audio_feature[410:538, :] /= 2
        audio_feature[:, 538:622] /= 2.5

        return audio_feature[start:end, :], pose_seq[start:end, :] * 5


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


if __name__ == '__main__':

    file_list = os.listdir('test/testset')
    for i in range(len(file_list)):
        file = file_list[i]
        new_file = file.replace(' ', '_').replace('-', '_').replace(',', '_')
        for n in range(2, 5):
            new_file = new_file.replace('_' * n, '')
        print(i, '\t', file)
        print('\t', new_file)
        try:
            os.rename('test/testset/' + file, 'test/testset/' + new_file)
        except FileNotFoundError:
            continue
