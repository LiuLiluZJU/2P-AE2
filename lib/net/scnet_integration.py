import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import numpy as np
from lib.net.scnet_model import SCNet
import matplotlib.pyplot as plt
from torchviz import make_dot
from tensorboardX import SummaryWriter


count = 0
count_test = 0

class SCNetIntegration(nn.Module):
    def __init__(self, args, device, input_channel, num_labels, spatial_downsample=8, learning_rate=1e-4):
        super(SCNetIntegration, self).__init__()
        self.device = device
        self.scnet = SCNet(input_channel, num_labels, spatial_downsample)
        # self.sigmas = nn.Parameter(torch.randn(num_labels))
        self.sigmas = nn.Parameter(torch.ones(num_labels) * 2)
        # self.sigmas = nn.Parameter(torch.ones(num_labels) * 10)
        self.sigma_scale = 1000.0
        self.sigma_regularization = 100.0
        self.optimizer = torch.optim.Adam([
            {'params': self.scnet.parameters()},
            {'params': self.sigmas}
            ], lr=learning_rate, weight_decay=1e-8)
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 - args.epochs) / float(args.decay_epochs + 1)
            return lr_l
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_rule)
        # self.writer = SummaryWriter()

    def set_input(self, input_image, target_landmarks, landmark_mask=None):
        self.input_image = input_image
        self.target_landmarks = target_landmarks
        self.landmark_mask = landmark_mask
        self.image_size = input_image.shape[2:]
        self.heatmap_size = self.image_size
    
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
        sigmas_reshaped = torch.reshape(sigmas, [batch_size, num_landmarks] + [1] * dim)
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
        # print("mse loss:", F.mse_loss(pred, target, reduction='mean') / batch_size)
        if mask is not None:
            return F.mse_loss(pred * mask, target * mask, reduction='mean') / batch_size
        else:
            return F.mse_loss(pred, target, reduction='mean') / batch_size

    def loss_function_sigmas(self, sigmas, valid_landmarks):
        # print("sigmas loss:", F.mse_loss(sigmas * valid_landmarks, torch.zeros_like(sigmas).to(self.device), reduction='mean'))
        return self.sigma_regularization * F.mse_loss(sigmas * valid_landmarks, torch.zeros_like(sigmas).to(self.device), reduction='mean')

    def forward(self):
        # self.prediction, self.local_prediction, self.spatial_prediction, self.sigmas_uncertainty = self.scnet(self.input_image)
        # self.sigmas = 10 + self.sigmas_uncertainty
        # target_heatmaps = self.generate_heatmap_target(self.heatmap_size, self.target_landmarks[:, :, :3], self.sigmas, scale=self.sigma_scale, normalize=True)
        self.prediction, self.local_prediction, self.spatial_prediction = self.scnet(self.input_image)
        target_heatmaps = self.generate_heatmap_target(self.heatmap_size, self.target_landmarks[:, :, :3], self.sigmas.repeat(self.prediction.shape[0], 1), scale=self.sigma_scale, normalize=True)
        # self.target_heatmaps = target_heatmaps.to(device=self.device)
        self.target_heatmaps = target_heatmaps.to(device=self.device).detach()
        # print("max:", torch.max(self.prediction), "max:", torch.max(target_heatmaps))
        # print("min:", torch.min(self.prediction), "min:", torch.min(target_heatmaps))

        global count
        count += 1
        tmp_pred = self.prediction.cpu().detach().numpy()
        tmp_pred = tmp_pred[0, :, :, :]
        tmp_pred = np.expand_dims(np.mean(tmp_pred, axis=0), axis=0)
        tmp_pred = np.concatenate((tmp_pred, tmp_pred, tmp_pred), axis=0)
        # self.writer.add_image('prediction', tmp, global_step=count)
        # plt.imshow(tmp)
        # plt.savefig("/home/leko/SCN-pytorch/tmp/{}.png".format(count))
        # plt.clf()
        # count += 1
        tmp_gt = target_heatmaps.detach().numpy()
        tmp_gt = tmp_gt[0, :, :, :]
        tmp_gt = np.expand_dims(np.mean(tmp_gt, axis=0), axis=0)
        tmp_gt = np.concatenate((tmp_gt, tmp_gt, tmp_gt), axis=0)
        # self.writer.add_image('prediction-groundtruth', np.concatenate([tmp_pred, tmp_gt], axis=2), global_step=count)
        # plt.imshow(tmp)
        # plt.savefig("/home/leko/SCN-pytorch/tmp/{}.png".format(count))
        # plt.clf()
        # count += 1
    
    def forward_test(self):
        # self.prediction, self.local_prediction, self.spatial_prediction, self.sigmas_uncertainty = self.scnet(self.input_image)
        # self.sigmas = 10 + self.sigmas_uncertainty
        # target_heatmaps = self.generate_heatmap_target(self.heatmap_size, self.target_landmarks[:, :, :3], self.sigmas, scale=self.sigma_scale, normalize=True)
        self.prediction, self.local_prediction, self.spatial_prediction = self.scnet(self.input_image)
        target_heatmaps = self.generate_heatmap_target(self.heatmap_size, self.target_landmarks[:, :, :3], self.sigmas.repeat(self.prediction.shape[0], 1), scale=self.sigma_scale, normalize=True)
        self.target_heatmaps = target_heatmaps.to(device=self.device)

        global count_test
        count_test += 1
        prediction_tmp = self.prediction.detach().cpu().numpy()
        tmp = prediction_tmp[0, :, :, :]
        tmp = np.mean(tmp, axis=0)
        plt.imshow(tmp)
        plt.savefig("/home/leko/SCN-pytorch-jbhi/tmp_test/{}.png".format(count_test))
        plt.clf()
        count_test += 1
        target_heatmaps_tmp = target_heatmaps.detach().numpy()
        tmp = target_heatmaps_tmp[0, :, :, :]
        tmp = np.mean(tmp, axis=0)
        plt.imshow(tmp)
        plt.savefig("/home/leko/SCN-pytorch-jbhi/tmp_test/{}.png".format(count_test))
        plt.clf()

        self.loss_net = self.loss_function(target=self.target_heatmaps, pred=self.prediction, mask=self.landmark_mask).cpu().data
        self.loss_sigmas = self.loss_function_sigmas(self.sigmas, self.target_landmarks[:, :, 0]).cpu().data
        # self.loss_sigmas = self.loss_function_sigmas(self.sigmas, self.target_landmarks[0, :, 0]).cpu().data
        print(self.sigmas)
        # self.loss_reg = get_reg_loss(self.reg_constant)
        # self.loss = self.loss_net + self.loss_reg + self.loss_sigmas
        self.loss = self.loss_net + self.loss_sigmas * 0.1
        # self.writer.add_scalar('loss_test', self.loss.item(), global_step=count_test)
        # self.writer.add_scalars('loss_test', {'total': self.loss.item()}, global_step=count)
        # self.writer.add_scalars('loss_test net sigmas', {'net': self.loss_net.item(), 'sigmas': self.loss_sigmas.item() * 0.1}, global_step=count_test)

        # PDE
        prediction_tmp = self.prediction.detach().cpu().numpy()
        target_landmarks_tmp = self.target_landmarks.detach().cpu().numpy()
        PDE_list = []
        max_confidence_list = []
        for b in range(prediction_tmp.shape[0]):
            for c in range(prediction_tmp.shape[1]):
                pred_heatmap = prediction_tmp[b, c, :, :]
                # print(pred_heatmap.shape)
                max_confidence_list.append(np.max(pred_heatmap))
                is_valid = target_landmarks_tmp[b, c, 0]
                if is_valid > 0:
                    pred_landmark = np.array([np.argmax(pred_heatmap) % pred_heatmap.shape[1], np.floor(np.argmax(pred_heatmap) / pred_heatmap.shape[1])])
                    gt_landmark = target_landmarks_tmp[b, c, 1 : 3]
                    # print(pred_landmark)
                    # print(gt_landmark)
                    PDE_list.append(np.linalg.norm(pred_landmark - gt_landmark))
        # print("max_confidence_list:", max_confidence_list)
        # print("target_list:", target_landmarks_tmp[:, :, 0])
        max_confidence_list = np.array(max_confidence_list)
        hamming_distance = np.where((max_confidence_list > 1) ^ (target_landmarks_tmp[:, :, 0].ravel() > 0.5) == True)[0].shape[0]
        mPDE = np.mean(PDE_list)
        # print("mPDE:", mPDE)
        # print("hamming_distance:", hamming_distance)
        # self.writer.add_scalar('mPDE', mPDE, global_step=count_test)
        # self.writer.add_scalar('hamming_distance', hamming_distance, global_step=count_test)

        return self.loss, self.loss_net, self.loss_sigmas, mPDE
        # return self.prediction.data.cpu().numpy(), self.target_heatmaps.data.cpu().numpy()

    def backward_basic(self):
        self.loss_net = self.loss_function(target=self.target_heatmaps, pred=self.prediction, mask=self.landmark_mask)
        self.loss_sigmas = self.loss_function_sigmas(self.sigmas, self.target_landmarks[:, :, 0])
        # self.loss_sigmas = self.loss_function_sigmas(self.sigmas, self.target_landmarks[0, :, 0])
        # self.loss_reg = get_reg_loss(self.reg_constant)
        # self.loss = self.loss_net + self.loss_reg + self.loss_sigmas
        # self.loss = self.loss_net + self.loss_sigmas * 0.1
        self.loss = self.loss_net
        # self.loss = torch.mean((torch.exp(-self.sigmas_uncertainty) * self.loss_net + self.sigmas_uncertainty) * 0.5) + self.loss_sigmas * 0.1
        # global count
        # # self.writer.add_scalar('loss', self.loss.item(), global_step=count)
        # self.writer.add_scalars('loss', {'total': self.loss.item()}, global_step=count)
        # self.writer.add_scalars('loss net', {'net': self.loss_net.item()}, global_step=count)
        # self.writer.add_scalars('loss sigmas', {'sigmas': self.loss_sigmas.item()}, global_step=count)
        # g = make_dot(self.loss)
        # g.view()
        if torch.max(torch.isnan(self.sigmas)):
            k = 1
            raise ArithmeticError()
        # print(self.sigmas)
        # print(self.loss)
        loss, loss_net, loss_sigmas = (self.loss, self.loss_net, self.loss_sigmas)
        self.loss.backward()

        return loss, loss_net, loss_sigmas
    
    def update_learning_rate(self, total_step):
        self.scheduler.step(total_step)
        lr = self.optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def optimize_parameters(self):
        # forward
        self()
        self.optimizer.zero_grad()
        loss, loss_net, loss_sigmas = self.backward_basic()
        nn.utils.clip_grad_value_(self.scnet.parameters(), 0.1)
        nn.utils.clip_grad_value_(self.sigmas, 0.1)
        self.optimizer.step()

        return loss, loss_net, loss_sigmas
