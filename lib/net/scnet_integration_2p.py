import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.transform import Rotation
from lib.net.scnet_model import SCNet
from lib.net.solver_bank import sweeney_2p_layer, sweeney_2p_ransac_layer, PnP_layer, P3PRansac_layer, P3PRansac_4DOF_layer
import matplotlib.pyplot as plt
import time
from torchviz import make_dot
from tensorboardX import SummaryWriter


count = 0
count_test = 0

class SCNetIntegration2P(nn.Module):
    def __init__(self, device, input_channel, num_labels, spatial_downsample=8, learning_rate=1e-4, solver='2pe2e', threshold=4):
        super(SCNetIntegration2P, self).__init__()
        self.device = device
        self.scnet = SCNet(input_channel, num_labels, spatial_downsample)
        self.solver = solver
        if solver == '2pe2e':
            self.minimal_slover_layer = sweeney_2p_layer(device, threshold)
        elif solver == '2pransac':
            self.minimal_slover_layer = sweeney_2p_ransac_layer(device, threshold)
        elif solver == 'pnp':
            self.minimal_slover_layer = PnP_layer(device)
        elif solver == 'p3pransac':
            self.minimal_slover_layer = P3PRansac_layer(device)
        elif solver == 'p3pransac4dof':
            self.minimal_slover_layer = P3PRansac_4DOF_layer(device)
        else:
            raise NotImplementedError()
        # self.sigmas = nn.Parameter(torch.randn(num_labels))
        # self.sigmas = nn.Parameter(torch.ones(num_labels) * 8)
        self.sigmas = nn.Parameter(torch.ones(num_labels) * 2)
        # self.sigmas = nn.Parameter(torch.ones(num_labels) * 10)
        self.sigma_scale = 1000.0
        self.sigma_regularization = 100.0
        self.optimizer = torch.optim.SGD([
            {'params': self.scnet.parameters()},
            {'params': self.sigmas},
            # {'params': self.minimal_slover_layer.parameters(), 'lr': 5e-2}
            ], lr=learning_rate, weight_decay=1e-8)
        # self.optimizer = torch.optim.Adam([
        #     {'params': self.scnet.parameters()},
        #     {'params': self.sigmas}
        #     ], lr=learning_rate, weight_decay=1e-8)
        # self.writer = SummaryWriter()

    def set_input(self, input_image, target_landmarks, transformation, noise_gamma=0, noise_beta=0, landmark_mask=None):
        self.input_image = input_image
        self.target_landmarks = target_landmarks
        self.transformation = transformation
        self.landmark_mask = landmark_mask
        self.image_size = input_image.shape[2:]
        self.heatmap_size = self.image_size
        self.noise_gamma = noise_gamma
        self.noise_beta = noise_beta
    
    def generate_heatmap_target(self, heatmap_size, landmarks, sigmas, scale=1.0, normalize=False):
        landmarks = landmarks.cpu()
        sigmas = sigmas.cpu()
        landmarks_shape = list(landmarks.shape)
        sigmas_shape = list(sigmas.shape)
        batch_size = landmarks_shape[0]
        num_landmarks = landmarks_shape[1]
        # print(landmarks_shape)
        # print(heatmap_size)
        dim = landmarks_shape[2] - 1
        assert len(heatmap_size) == dim, 'Dimensions do not match.'
        assert sigmas_shape[1] == num_landmarks, 'Number of sigmas does not match.'
        heatmap_axis = 1
        landmarks_reshaped = torch.reshape(landmarks[..., 1:], [batch_size, num_landmarks] + [1] * dim + [dim])
        is_valid_reshaped = torch.reshape(landmarks[..., 0], [batch_size, num_landmarks] + [1] * dim)
        sigmas_reshaped = torch.reshape(sigmas, [1, num_landmarks] + [1] * dim)
        aranges = [torch.arange(s) for s in heatmap_size]
        grid = torch.meshgrid(*aranges)
        grid_stacked = torch.stack(grid, dim=dim)  # shape:(H, W, 2)
        grid_stacked = torch.flip(grid_stacked, dims=[dim])
        grid_stacked = grid_stacked.float()
        grid_stacked = torch.stack([grid_stacked] * batch_size, dim=0)
        grid_stacked = torch.stack([grid_stacked] * num_landmarks, dim=heatmap_axis)
        if normalize:
            scale /= torch.pow(np.sqrt(2 * np.pi) * sigmas_reshaped, dim)
        # print("scale:", scale)
        squared_distances = torch.squeeze(torch.mean(torch.pow(grid_stacked - landmarks_reshaped, 2.0), dim=-1), dim=-1)
        heatmap = scale * torch.exp(-squared_distances / (2 * torch.pow(sigmas_reshaped, 2)))  # Gaussain function
        # print("max, min:", torch.max(heatmap), torch.min(heatmap))
        heatmap_or_zeros = torch.where((is_valid_reshaped + torch.zeros_like(heatmap)) > 0, heatmap, torch.zeros_like(heatmap))
        # Show heatmaps
        # for b in range(2):
        #     for c in range(25):
        #         tmp = heatmap_or_zeros[b, c, :, :]
        #         tmp = tmp.detach().numpy()
        #         plt.imshow(tmp)
        #         plt.show()
            
        return heatmap_or_zeros

    def loss_function(self, pred, target, mask=None):
        batch_size = pred.shape[0]
        # print("mse loss:", F.mse_loss(pred, target, reduction='sum') / batch_size)
        if mask is not None:
            return F.mse_loss(pred * mask, target * mask, reduction='sum') / batch_size
        else:
            return F.mse_loss(pred, target, reduction='sum') / batch_size

    def loss_function_sigmas(self, sigmas, valid_landmarks):
        # print("sigmas loss:", F.mse_loss(sigmas * valid_landmarks, torch.zeros_like(sigmas).to(self.device), reduction='sum'))
        return self.sigma_regularization * F.mse_loss(sigmas * valid_landmarks, torch.zeros_like(sigmas).to(self.device), reduction='sum')

    def forward(self):
        # self.prediction, self.local_prediction, self.spatial_prediction, self.sigmas_uncertainty = self.scnet(self.input_image)
        # self.sigmas = 2 + self.sigmas_uncertainty
        # print(self.sigmas)
        # target_heatmaps = self.generate_heatmap_target(self.heatmap_size, self.target_landmarks[:, :, :3], self.sigmas, scale=self.sigma_scale, normalize=True)
        self.prediction, self.local_prediction, self.spatial_prediction = self.scnet(self.input_image)
        target_heatmaps = self.generate_heatmap_target(self.heatmap_size, self.target_landmarks[:, :, :3], self.sigmas.repeat(self.prediction.shape[0], 1), scale=self.sigma_scale, normalize=True)
        self.target_heatmaps = target_heatmaps.to(device=self.device)
        # print("max:", torch.max(self.prediction), "max:", torch.max(target_heatmaps))
        # print("min:", torch.min(self.prediction), "min:", torch.min(target_heatmaps))

        global count
        count += 1
        tmp = self.prediction.cpu().detach().numpy()
        # print(np.argmax(tmp))
        tmp = tmp[0, :, :, :]
        tmp = np.expand_dims(np.mean(tmp, axis=0), axis=0)
        tmp = np.concatenate((tmp, tmp, tmp), axis=0)
        # self.writer.add_image('prediction', tmp, global_step=count)
        # plt.imshow(tmp)
        # plt.savefig("/home/leko/SCN-pytorch/tmp/{}.png".format(count))
        # plt.clf()
        # count += 1
        tmp = target_heatmaps.detach().numpy()
        tmp = tmp[0, :, :, :]
        tmp = np.expand_dims(np.mean(tmp, axis=0), axis=0)
        tmp = np.concatenate((tmp, tmp, tmp), axis=0)
        # self.writer.add_image('groundtruth', tmp, global_step=count)
        # plt.imshow(tmp)
        # plt.savefig("/home/leko/SCN-pytorch/tmp/{}.png".format(count))
        # plt.clf()
        # count += 1

        # 2P solver
        if self.solver in ['2pe2e', '2pransac']:
            self.pose_pred = self.minimal_slover_layer(self.prediction, self.target_landmarks, self.transformation, self.noise_gamma, self.noise_beta)
            if not self.pose_pred:
                return []
            r = Rotation.from_matrix(self.transformation[:, :3, :3].detach().cpu().numpy())
            angle_target = r.as_euler('xyz')
            self.az_target = torch.from_numpy(angle_target[:, 2]).to(self.device, torch.float32)
            self.t_target = self.transformation[:, :3, 3]
            # print("T_pred:", self.pose_pred[1:])
            return self.pose_pred[1:]

        # PnP solver
        elif self.solver in ['p3pransac', 'p3pransac4dof', 'pnp']:
            if self.solver == 'p3pransac4dof':
                self.pose_pred = self.minimal_slover_layer(self.prediction, self.target_landmarks, self.transformation)
            else:
                self.pose_pred = self.minimal_slover_layer(self.prediction, self.target_landmarks)
            r = Rotation.from_matrix(self.pose_pred[:3, :3])
            angle_pred = r.as_euler('xyz')
            self.az_pred = torch.tensor([angle_pred[2]]).to(self.device, torch.float32)
            self.t_pred = torch.from_numpy(np.expand_dims(self.pose_pred[:3, 3], axis=0)).to(self.device, torch.float32)
            # print(np.rad2deg(angle_pred), self.t_pred)
            self.pose_pred = (self.t_pred, self.az_pred)
            return self.pose_pred, angle_pred
        else:
            raise NotImplementedError()
        
    
    def forward_test(self):
        # self.prediction, self.local_prediction, self.spatial_prediction, self.sigmas_uncertainty = self.scnet(self.input_image)
        # self.sigmas = 2 + self.sigmas_uncertainty
        # target_heatmaps = self.generate_heatmap_target(self.heatmap_size, self.target_landmarks[:, :, :3], self.sigmas, scale=self.sigma_scale, normalize=True)
        self.prediction, self.local_prediction, self.spatial_prediction = self.scnet(self.input_image)
        target_heatmaps = self.generate_heatmap_target(self.heatmap_size, self.target_landmarks[:, :, :3], self.sigmas.repeat(self.prediction.shape[0], 1), scale=self.sigma_scale, normalize=True)
        self.target_heatmaps = target_heatmaps.to(device=self.device)

        global count_test
        count_test += 1
        prediction_tmp = self.prediction.detach().cpu().numpy()
        tmp = prediction_tmp[0, :, :, :]
        tmp = np.mean(tmp, axis=0)
        # plt.imshow(tmp)
        # plt.savefig("/home/leko/SCN-pytorch/tmp_test/{}.png".format(count_test))
        # plt.clf()
        count_test += 1
        target_heatmaps_tmp = target_heatmaps.detach().numpy()
        tmp = target_heatmaps_tmp[0, :, :, :]
        tmp = np.mean(tmp, axis=0)
        # plt.imshow(tmp)
        # plt.savefig("/home/leko/SCN-pytorch/tmp_test/{}.png".format(count_test))
        # plt.clf()

        self.loss_net = self.loss_function(target=self.target_heatmaps, pred=self.prediction, mask=self.landmark_mask)
        self.loss_sigmas = self.loss_function_sigmas(self.sigmas, self.target_landmarks[0, :, 0])
        # self.loss_reg = get_reg_loss(self.reg_constant)
        # self.loss = self.loss_net + self.loss_reg + self.loss_sigmas
        self.loss = self.loss_net + self.loss_sigmas
        # self.writer.add_scalar('loss_test', self.loss.item(), global_step=count_test)
        # self.writer.add_scalars('loss_test', {'total': self.loss.item(), 'net': self.loss_net.item(), 'sigmas': self.loss_sigmas.item()}, global_step=count_test)

        # PDE
        prediction_tmp = self.prediction.detach().cpu().numpy()
        target_landmarks_tmp = self.target_landmarks.detach().cpu().numpy()
        PDE_list = []
        max_confidence_list = []
        for b in range(prediction_tmp.shape[0]):
            for c in range(prediction_tmp.shape[1]):
                pred_heatmap = prediction_tmp[b, c, :, :]
                # plt.imshow(pred_heatmap, cmap='gray')
                # plt.savefig("/home/leko/SCN-pytorch/tmp_show/{}_pred.png".format(count_test))
                # plt.clf()
                # target_heatmap = target_heatmaps[b, c, :, :]
                # plt.imshow(target_heatmap, cmap='gray')
                # plt.savefig("/home/leko/SCN-pytorch/tmp_show/{}_target.png".format(count_test))
                # plt.clf()
                # print(pred_heatmap.shape)
                max_confidence_list.append(np.max(pred_heatmap))
                is_valid = target_landmarks_tmp[b, c, 0]
                if is_valid > 0:
                    pred_landmark = np.array([np.argmax(pred_heatmap) % pred_heatmap.shape[1], np.floor(np.argmax(pred_heatmap) / pred_heatmap.shape[1])])
                    gt_landmark = target_landmarks_tmp[b, c, 1:3]
                    # print(pred_landmark)
                    # print(gt_landmark)
                    PDE_list.append(np.linalg.norm(pred_landmark - gt_landmark))
        print("max_confidence_list:", max_confidence_list)
        print("target_list:", target_landmarks_tmp[:, :, 0])
        max_confidence_list = np.array(max_confidence_list)
        hamming_distance = np.where((max_confidence_list > 1) ^ (target_landmarks_tmp[:, :, 0].ravel() > 0.5) == True)[0].shape[0]
        mPDE = np.mean(PDE_list)
        print("mPDE:", mPDE)
        print("hamming_distance:", hamming_distance)
        # self.writer.add_scalar('mPDE', mPDE, global_step=count_test)
        # self.writer.add_scalar('hamming_distance', hamming_distance, global_step=count_test)
        return self.prediction.detach().cpu().numpy(), self.target_landmarks.detach().cpu().numpy(), mPDE

    def backward_basic(self):
        self.loss_net = self.loss_function(target=self.target_heatmaps, pred=self.prediction, mask=self.landmark_mask)
        self.loss_sigmas = self.loss_function_sigmas(self.sigmas, self.target_landmarks[:, :, 0])
        # self.loss_reg = get_reg_loss(self.reg_constant)
        # self.loss = self.loss_net + self.loss_reg + self.loss_sigmas
        # self.loss_pose = torch.max(F.mse_loss(self.pose_pred[2], self.az_target, reduction='sum'), F.mse_loss(self.pose_pred[1][:, :2], self.t_target[:, :2], reduction='sum'))
        self.loss_pose = F.mse_loss(self.pose_pred[2], self.az_target, reduction='sum') + F.mse_loss(self.pose_pred[1][:, :2], self.t_target[:, :2], reduction='sum') + \
            1e-3 * F.mse_loss(self.pose_pred[1][:, 2], self.t_target[:, 2], reduction='sum')
        # self.loss_pose = F.mse_loss(self.pose_pred[2], self.az_target, reduction='sum') + F.mse_loss(self.pose_pred[1][:, :2], self.t_target[:, :2], reduction='sum')
        # self.loss = self.loss_net + self.loss_sigmas * 0.1 + self.loss_pose
        self.loss = self.loss_net * 1e-4 + self.loss_sigmas * 1e-4 + self.loss_pose
        # self.loss = self.loss_pose
        # self.loss = self.loss_pose + self.loss_net * 0.1
        print("pose loss:", self.loss_pose)
        print("net loss:", self.loss_net)
        print("sigmas loss:", self.loss_sigmas * 0.1)
        print("total loss:", self.loss)
        global count
        # self.writer.add_scalar('loss', self.loss.item(), global_step=count)
        # self.writer.add_scalars('loss', {'total': self.loss.item(), 'net': self.loss_net.item(), 'sigmas': self.loss_sigmas.item() * 0.1}, global_step=count) 
        # g = make_dot(self.loss)
        # g.view()
        if torch.max(torch.isnan(self.sigmas)):
            k = 1
            raise ArithmeticError()
        # print(self.sigmas)
        # print(self.loss)
        self.loss.backward()

    def optimize_parameters(self):
        # forward
        self()
        if not self.pose_pred:
            return False
        self.optimizer.zero_grad()
        self.backward_basic()
        nn.utils.clip_grad_value_(self.scnet.parameters(), 0.1)
        nn.utils.clip_grad_value_(self.sigmas, 0.1)
        self.optimizer.step()
        return True
