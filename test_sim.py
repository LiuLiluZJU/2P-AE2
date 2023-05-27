import os
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader, random_split, Subset
from scipy.spatial.transform import Rotation
import matplotlib
import matplotlib.pyplot as plt
import time

from lib.net.scnet_integration_2p import SCNetIntegration2P
from lib.dataset.align_data import AlignDataSet, AlignDataSetLung, AlignDataSetWithoutFile


def get_args():
    parser = argparse.ArgumentParser(description='Train the SCNet on images and target landmarks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', dest='epochs', metavar='E', type=int, default=150, help='Number of epochs')
    parser.add_argument('-b', '--batch-size', dest='batchsize', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('-l', '--learning-rate', dest='lr', metavar='LR', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=0.2, help='Percent of the data that is used as validation (0-1)')
    parser.add_argument('-sv', '--solver', dest='solver', type=str, default="2pe2e", help='Solver layer for pose estimation, e.g. p3pransac, p3pransac4dof, pnp, 2pe2e, 2pransac')
    parser.add_argument('-t', '--threshold', dest='threshold', type=float, default=4, help='Inlier threshold of the EARE')
    parser.add_argument('-d', '--dataset-dir', dest='dataset_dir', type=str, default="/home/leko/SCN-pytorch-jbhi/data/ap_aug_crop_lung/", 
                        help='Path of dataset for training and validation')

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = SCNetIntegration2P(device=device, input_channel=1, num_labels=25, learning_rate=args.lr, solver=args.solver, threshold=args.threshold)
    pretrained_dict = torch.load("/media/leko/LLL/SCN-pytorch-jbhi/checkpoints/ap_aug_crop_sigma2/regular_100.pth", map_location=device)
    # pretrained_dict = torch.load("/media/leko/LLL/SCN-pytorch-jbhi/checkpoints/ap_aug_crop_sigma2_refine/regular_200.pth", map_location=device)
    model_dict=net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}  # 不必要的键去除掉
    model_dict.update(pretrained_dict)  # 覆盖现有的字典里的条目
    net.load_state_dict(model_dict)
    net.to(device=device)

    val = AlignDataSetLung(args.dataset_dir, is_val=True)
    val_loader = DataLoader(val, batch_size=args.batchsize, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    net.eval()
    batch_count_test = 0
    total_time = 0
    mPDE_list = []
    mTRE_list = []
    ax_error = []
    ay_error = []
    az_error = []
    X_error = []
    Y_error = []
    Z_error = []
    data_name_list = []
    dist_2d_list = []
    for batch in val_loader:
        batch_count_test += 1
        print("Test {}th batch!".format(batch_count_test))
        tic = time.time()
        image = batch[0].to(device=device, dtype=torch.float32)
        landmarks = batch[1].to(device=device, dtype=torch.float32)
        landmarks_lung = batch[2].to(device=device, dtype=torch.float32)
        transformation = batch[3].to(device=device, dtype=torch.float32)
        center = batch[4]
        source = batch[5]
        data_name = batch[6]
        
        data_name_list.append(data_name[0])

        net.set_input(image, landmarks, transformation)
        if args.solver in ['p3pransac', 'p3pransac4dof', 'pnp']:
            pose_pred, angle_pnp = net.forward()
        elif args.solver in ['2pe2e', '2pransac']:
            pose_pred = net.forward()
        else:
            raise NotImplementedError()
        toc = time.time()
        total_time = total_time + toc - tic
        print("time:", toc - tic)
        if not pose_pred:
            raise NotImplementedError()
            continue

        Tr = transformation[0].detach().cpu().numpy()
        r = Rotation.from_matrix(Tr[:3, :3])
        angle = r.as_euler('xyz')
        print("data name:", data_name)
        print("trans diff:", np.abs(Tr[:3, 3] - pose_pred[0].detach().cpu().numpy()[0]))
        print("angle diff:", np.abs(angle[2] - np.float(pose_pred[1].detach().cpu().numpy())))

        # Calculate TRE
        if args.solver in ['p3pransac', 'p3pransac4dof', 'pnp']:
            r_pred = Rotation.from_euler('xyz', [angle_pnp[0], angle_pnp[1], np.float(pose_pred[1].detach().cpu().numpy())])
        elif args.solver in ['2pe2e', '2pransac']:
            r_pred = Rotation.from_euler('xyz', [np.pi / 2, 0, np.float(pose_pred[1].detach().cpu().numpy())])
        else:
            raise NotImplementedError()
        R_pred = r_pred.as_matrix()
        Tr_pred = np.identity(4)
        Tr_pred[:3, :3] = R_pred
        Tr_pred[:3, 3] = pose_pred[0].detach().cpu().numpy()[0][:3]
        landmarks_3d = landmarks_lung[0, :, 3:].detach().cpu().numpy()
        TRE = np.mean(np.linalg.norm((np.dot(Tr[:3, :3], landmarks_3d.T).T + Tr[:3, 3] - np.dot(Tr_pred[:3, :3], landmarks_3d.T).T - Tr_pred[:3, 3]), axis=1))
        mTRE_list.append(TRE)

        # Calculate specific DoF error
        angle_pred = r_pred.as_euler('xyz')
        ax_error.append(np.abs(angle[0] - angle_pred[0]))
        ay_error.append(np.abs(angle[1] - angle_pred[1]))
        az_error.append(np.abs(angle[2] - angle_pred[2]))
        X_error.append(np.abs(Tr[0, 3] - Tr_pred[0, 3]))
        Y_error.append(np.abs(Tr[1, 3] - Tr_pred[1, 3]))
        Z_error.append(np.abs(Tr[2, 3] - Tr_pred[2, 3]))

        # Calculate PDE
        # intrinsic = np.array([[1.20e+03, 0.00e+00, 1.28e+02],
        #                             [0.00e+00, 1.20e+03, 1.28e+02],
        #                             [0.00e+00, 0.00e+00, 1.00e+00]])
        intrinsic = np.array([[804.18848168,   0.        , 128.        ],
                                    [  0.        , 804.18848168, 128.        ],
                                    [  0.        ,   0.        ,   1.        ]])
        landmarks_2d = landmarks_lung[0, :, (1, 2)].detach().cpu().numpy()[:, :2]
        landmarks_3d_cam = np.dot(Tr_pred[:3, :3], landmarks_3d.T).T + Tr_pred[:3, 3]
        landmarks_2d_pred = (1 / landmarks_3d_cam[:, 2] * np.dot(intrinsic, landmarks_3d_cam.T)).T[:, :2]
        landmarks_2d_pred[np.logical_not(landmarks_lung[0, :, 0].detach().cpu().numpy().astype(np.bool))] = 0
        PDE = np.mean(np.linalg.norm(landmarks_2d - landmarks_2d_pred, axis=1))
        mPDE_list.append(PDE)

    print("mean projection error:", np.mean(mPDE_list, axis=0))
    print("mean TRE:", np.mean(mTRE_list))
    X_error.sort()
    Y_error.sort()
    Z_error.sort()
    ax_error.sort()
    ay_error.sort()
    az_error.sort()
    mPDE_list.sort()
    mTRE_list.sort()
    print("X_error 50th 75th 95th: {:.2f} / {:.2f} / {:.2f}".format(X_error[int(len(X_error) * 0.5)], X_error[int(len(X_error) * 0.75)], X_error[int(len(X_error) * 0.95)]))
    print("Y_error 50th 75th 95th: {:.2f} / {:.2f} / {:.2f}".format(Y_error[int(len(Y_error) * 0.5)], Y_error[int(len(Y_error) * 0.75)], Y_error[int(len(Y_error) * 0.95)]))
    print("Z_error 50th 75th 95th: {:.2f} / {:.2f} / {:.2f}".format(Z_error[int(len(Z_error) * 0.5)], Z_error[int(len(Z_error) * 0.75)], Z_error[int(len(Z_error) * 0.95)]))
    print("ax_error 50th 75th 95th: {:.2f} / {:.2f} / {:.2f}".format(np.rad2deg(ax_error[int(len(ax_error) * 0.5)]), np.rad2deg(ax_error[int(len(ax_error) * 0.75)]), np.rad2deg(ax_error[int(len(ax_error) * 0.95)])))
    print("ay_error 50th 75th 95th: {:.2f} / {:.2f} / {:.2f}".format(np.rad2deg(ay_error[int(len(ay_error) * 0.5)]), np.rad2deg(ay_error[int(len(ay_error) * 0.75)]), np.rad2deg(ay_error[int(len(ay_error) * 0.95)])))
    print("az_error 50th 75th 95th: {:.2f} / {:.2f} / {:.2f}".format(np.rad2deg(az_error[int(len(az_error) * 0.5)]), np.rad2deg(az_error[int(len(az_error) * 0.75)]), np.rad2deg(az_error[int(len(az_error) * 0.95)])))
    print("mTRE 50th 75th 95th: {:.2f} / {:.2f} / {:.2f}".format(mTRE_list[int(len(mTRE_list) * 0.5)], mTRE_list[int(len(mTRE_list) * 0.75)], mTRE_list[int(len(mTRE_list) * 0.95)]))
    print("mPDE 50th 75th 95th: {:.2f} / {:.2f} / {:.2f}".format(mPDE_list[int(len(mPDE_list) * 0.5)], mPDE_list[int(len(mPDE_list) * 0.75)], mPDE_list[int(len(mPDE_list) * 0.95)]))
    print("avg time:", total_time / batch_count_test)

    print("alpha mean std: {:.2f} {:.2f}".format(np.rad2deg(np.mean(ax_error)), np.rad2deg(np.std(ax_error))))
    print("beta mean std: {:.2f} {:.2f}".format(np.rad2deg(np.mean(ay_error)), np.rad2deg(np.std(ay_error))))
    print("gamma mean std: {:.2f} {:.2f}".format(np.rad2deg(np.mean(az_error)), np.rad2deg(np.std(az_error))))
    print("x mean std: {:.2f} {:.2f}".format(np.mean(X_error), np.std(X_error)))
    print("y mean std: {:.2f} {:.2f}".format(np.mean(Y_error), np.std(Y_error)))
    print("z mean std: {:.2f} {:.2f}".format(np.mean(Z_error), np.std(Z_error)))
    print("mtre_lung mean std: {:.2f} {:.2f}".format(np.mean(mTRE_list), np.std(mTRE_list)))
    print("mpde_lung mean std: {:.2f} {:.2f}".format(np.mean(mPDE_list), np.std(mPDE_list)))
    print("gfr:", len(np.array(mTRE_list)[np.array(mTRE_list) > 30]) / len(mTRE_list))
