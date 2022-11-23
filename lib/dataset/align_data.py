import os
import sys
import h5py
import matplotlib.pyplot as plt

import cv2
from skimage import io, transform
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset


class AlignDataSet(Dataset):
    def __init__(self, dataset_dir, is_val):
        super(AlignDataSet, self).__init__()
        self.ext = '.nii.gz.h5'
        if is_val:
            with open(os.path.join(dataset_dir, "test.txt"), 'r') as f:
                self.data_list = f.readlines()
        else:
            with open(os.path.join(dataset_dir, "train.txt"), 'r') as f:
                self.data_list = f.readlines()
        self.dataset_path = os.path.join(dataset_dir, "data")
        print(self.data_list)
        self.dataset_size = len(self.data_list)
    
    def __len__(self):
        return self.dataset_size

    def get_data_path(self, root, index_name):
        pass

    def load_file(self, data_path):
        h5_file = h5py.File(data_path, 'r')
        image = np.asarray(h5_file['image'])
        landmarks = np.asarray(h5_file['landmarks'])
        transformation = np.asarray(h5_file['transformation'])
        center = np.asarray(h5_file['center'])
        source = np.asarray(h5_file['source'])
        h5_file.close()
        return image, landmarks, transformation, center, source
        # h5_file.close()
        # return image, landmarks, transformation

    def preprocess(self, image, max_value=255, min_value=0):
        pass

    def __getitem__(self, item):
        data_file_name = self.data_list[item].rstrip('\n') + self.ext
        image, landmarks, transformation, center, source = self.load_file(os.path.join(self.dataset_path, data_file_name))
        image = np.expand_dims(image, axis=0)
        
        return image, landmarks, transformation, center, source, data_file_name


class AlignDataSetWithoutFile(Dataset):
    def __init__(self, dataset_dir):
        super(AlignDataSetWithoutFile, self).__init__()
        self.ext = '.nii.gz.h5'
        self.dataset_path = os.path.join(dataset_dir)
        self.data_list = os.listdir(self.dataset_path)
        print(self.data_list)
        self.dataset_size = len(self.data_list)
    
    def __len__(self):
        return self.dataset_size

    def get_data_path(self, root, index_name):
        pass

    def load_file(self, data_path):
        h5_file = h5py.File(data_path, 'r')
        image = np.asarray(h5_file['image'])
        landmarks = np.asarray(h5_file['landmarks'])
        transformation = np.asarray(h5_file['transformation'])
        center = np.asarray(h5_file['center'])
        source = np.asarray(h5_file['source'])
        h5_file.close()
        return image, landmarks, transformation, center, source
        # h5_file.close()
        # return image, landmarks, transformation

    def preprocess(self, image, max_value=255, min_value=0):
        pass

    def __getitem__(self, item):
        data_file_name = self.data_list[item]
        image, landmarks, transformation, center, source = self.load_file(os.path.join(self.dataset_path, data_file_name))
        image = np.expand_dims(image, axis=0)
        
        return image, landmarks, transformation, center, source, data_file_name


class AlignDataSetLung(Dataset):
    def __init__(self, dataset_dir, is_val):
        super(AlignDataSetLung, self).__init__()
        self.ext = '.nii.gz.h5'
        if is_val:
            with open(os.path.join(dataset_dir, "test.txt"), 'r') as f:
                self.data_list = f.readlines()
        else:
            with open(os.path.join(dataset_dir, "train.txt"), 'r') as f:
                self.data_list = f.readlines()
        self.dataset_path = os.path.join(dataset_dir, "data")
        print(self.data_list)
        self.dataset_size = len(self.data_list)
    
    def __len__(self):
        return self.dataset_size

    def get_data_path(self, root, index_name):
        pass

    def load_file(self, data_path):
        h5_file = h5py.File(data_path, 'r')
        image = np.asarray(h5_file['image'])
        landmarks = np.asarray(h5_file['landmarks'])
        landmarks_lung = np.asarray(h5_file['landmarks_lung'])
        transformation = np.asarray(h5_file['transformation'])
        center = np.asarray(h5_file['center'])
        source = np.asarray(h5_file['source'])
        h5_file.close()
        return image, landmarks, landmarks_lung, transformation, center, source
        # h5_file.close()
        # return image, landmarks, transformation

    def preprocess(self, image, max_value=255, min_value=0):
        pass

    def __getitem__(self, item):
        data_file_name = self.data_list[item].rstrip('\n') + self.ext
        image, landmarks, landmarks_lung, transformation, center, source = self.load_file(os.path.join(self.dataset_path, data_file_name))
        image = np.expand_dims(image, axis=0)
        
        return image, landmarks, landmarks_lung, transformation, center, source, data_file_name