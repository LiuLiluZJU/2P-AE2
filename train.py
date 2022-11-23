import os
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader, random_split, Subset
from tensorboardX import SummaryWriter

from lib.net.scnet_integration import SCNetIntegration
from lib.dataset.align_data import AlignDataSet


def get_args():
    parser = argparse.ArgumentParser(description='Train the SCNet on images and target landmarks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', dest='epochs', metavar='E', type=int, default=50, help='Number of epochs')
    parser.add_argument('-de', '--decay-epochs', dest='decay_epochs', metavar='DE', type=int, default=50, help='Number of decay epochs')
    parser.add_argument('-b', '--batch-size', dest='batchsize', metavar='B', type=int, default=4, help='Batch size')
    parser.add_argument('-l', '--learning-rate', dest='lr', metavar='LR', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=0.2, help='Percent of the data that is used as validation (0-1)')
    parser.add_argument('-d', '--dataset-dir', dest='dataset_dir', type=str, default="/home/leko/SCN-pytorch-jbhi/data/ap_aug_crop", 
                        help='Path of dataset for training and validation')
    parser.add_argument('-m', '--model-dir', dest='model_dir', type=str, default="checkpoints/ap_aug_crop_sigma2", 
                        help='Path of trained model for saving')

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = SCNetIntegration(args=args, device=device, input_channel=1, num_labels=25, learning_rate=args.lr)
    net.to(device=device)

    train = AlignDataSet(args.dataset_dir, is_val=False)
    val = AlignDataSet(args.dataset_dir, is_val=True)

    train_loader = DataLoader(train, batch_size=args.batchsize, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=args.batchsize, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    writer = SummaryWriter()

    for epoch in range(args.epochs + args.decay_epochs + 1):

        print(f"Start {epoch}th epoch!")
        net.train()
        batch_count = 0
        total_loss = 0
        total_loss_net = 0
        total_loss_sigmas = 0
        for batch in train_loader:
            batch_count += 1
            print("Start {}th epoch, {}th batch!".format(epoch, batch_count))
            image = batch[0].to(device=device, dtype=torch.float32)
            landmarks = batch[1].to(device=device, dtype=torch.float32)
            data_name = batch[5]

            net.set_input(image, landmarks)
            loss, loss_net, loss_sigmas = net.optimize_parameters()
            total_loss += loss
            total_loss_net += loss_net
            total_loss_sigmas += loss_sigmas
        writer.add_scalars('loss', {'total': (total_loss / batch_count).item()}, global_step=epoch)
        writer.add_scalars('loss net', {'net': (total_loss_net / batch_count).item()}, global_step=epoch)
        writer.add_scalars('loss sigmas', {'sigmas': (total_loss_sigmas / batch_count).item()}, global_step=epoch)

        # if epoch % 1 == 0 and epoch > 0:
        net.eval()
        batch_count_test = 0
        total_loss_test = 0
        total_loss_net_test = 0
        total_loss_sigmas_test = 0
        total_mPDE = 0
        with torch.no_grad():
            for batch in val_loader:
                batch_count_test += 1
                print("Test {}th epoch, {}th batch!".format(epoch, batch_count_test))
                image = batch[0].to(device=device, dtype=torch.float32)
                landmarks = batch[1].to(device=device, dtype=torch.float32)
                data_name = batch[5]

                net.set_input(image, landmarks)
                loss_test, loss_net_test, loss_sigmas_test, mPDE = net.forward_test()
                total_loss_test += loss_test
                total_loss_net_test += loss_net_test
                total_loss_sigmas_test += loss_sigmas_test
                total_mPDE += mPDE
            writer.add_scalars('loss_test', {'total': (total_loss_test / batch_count_test).item()}, global_step=epoch)
            writer.add_scalars('loss net test', {'net': (total_loss_net_test / batch_count_test).item()}, global_step=epoch)
            writer.add_scalars('loss sigmas test', {'sigmas': (total_loss_sigmas_test / batch_count_test).item()}, global_step=epoch)
            writer.add_scalars('mPDE', {'mPDE': (total_mPDE / batch_count_test).item()}, global_step=epoch)
        
        if epoch % 10 == 0 and epoch > 0:
            if not os.path.exists(args.model_dir):
                os.mkdir(args.model_dir)
            torch.save(net.state_dict(), os.path.join(args.model_dir, "regular_{}.pth".format(epoch)))
        
        net.update_learning_rate(epoch)
