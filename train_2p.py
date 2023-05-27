import os
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader, random_split, Subset

from lib.net.scnet_integration_2p import SCNetIntegration2P
from lib.dataset.align_data import AlignDataSet


def get_args():
    parser = argparse.ArgumentParser(description='Train the SCNet on images and target landmarks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', dest='epochs', metavar='E', type=int, default=200, help='Number of epochs')
    parser.add_argument('-b', '--batch-size', dest='batchsize', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('-l', '--learning-rate', dest='lr', metavar='LR', type=float, default=1e-8, help='Learning rate')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=0.2, help='Percent of the data that is used as validation (0-1)')
    parser.add_argument('-sv', '--solver', dest='solver', type=str, default="2pe2e", help='Solver layer for pose estimation')
    parser.add_argument('-d', '--dataset-dir', dest='dataset_dir', type=str, default="/home/leko/SCN-pytorch-jbhi/data/ap_aug_crop", 
                        help='Path of dataset for training and validation')
    parser.add_argument('-m', '--model-dir', dest='model_dir', type=str, default="checkpoints/ap_aug_crop_sigma2_refine", 
                        help='Path of trained model for saving')

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = SCNetIntegration2P(device=device, input_channel=1, num_labels=25, learning_rate=args.lr, solver=args.solver)
    pretrained_dict = torch.load("/media/leko/LLL/SCN-pytorch-jbhi/checkpoints/ap_aug_crop_sigma2/regular_100.pth", map_location=device)
    model_dict=net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}  # 不必要的键去除掉
    model_dict.update(pretrained_dict)  # 覆盖现有的字典里的条目
    net.load_state_dict(model_dict)
    net.to(device=device)

    train = AlignDataSet(args.dataset_dir, is_val=False)
    val = AlignDataSet(args.dataset_dir, is_val=True)

    train_loader = DataLoader(train, batch_size=args.batchsize, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=args.batchsize, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    for epoch in range(args.epochs + 1):

        print(f"Start {epoch}th epoch!")
        net.train()
        batch_count = 0
        for batch in train_loader:
            batch_count += 1
            print("Start {}th epoch, {}th batch!".format(epoch, batch_count))
            image = batch[0].to(device=device, dtype=torch.float32)
            landmarks = batch[1].to(device=device, dtype=torch.float32)
            transformation = batch[2].to(device=device, dtype=torch.float32)
            net.set_input(image, landmarks, transformation)
            valid_flag = net.optimize_parameters()
            if not valid_flag:
                continue

        if epoch % 10 == 0:
            net.eval()
            batch_count_test = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch_count_test += 1
                    print("Test {}th epoch, {}th batch!".format(epoch, batch_count_test))
                    image = batch[0].to(device=device, dtype=torch.float32)
                    landmarks = batch[1].to(device=device, dtype=torch.float32)

                    net.set_input(image, landmarks, transformation)
                    net.forward_test()
            
            if not os.path.exists(args.model_dir):
                os.mkdir(args.model_dir)
            torch.save(net.state_dict(), os.path.join(args.model_dir, "regular_{}.pth".format(epoch)))
