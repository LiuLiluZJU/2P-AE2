""" Minimal solution solver layers """

import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia import SpatialSoftArgmax2d
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
import cv2
import math
import random
import numpy as np
# np.random.seed(2)
np.random.seed(0)
# random.seed(8)
random.seed(13)


class PnP_layer(nn.Module):
    def __init__(self, device):
        super(PnP_layer, self).__init__()
        self.softArgmax = SpatialSoftArgmax2d(temperature=10, normalized_coordinates=False)
        self.device = device
    
    def forward(self, pred_heatmaps, target_landmarks):
        heatmaps_shape = pred_heatmaps.shape
        landmarks_mask = target_landmarks[:, :, 0]
        landmarks_2d_gt = target_landmarks[:, :, 1 : 3]
        landmarks_3d = target_landmarks[:, :, 3:]
        landmarks_2d = self.softArgmax(pred_heatmaps)
        landmarks_2d = landmarks_2d[landmarks_mask.bool()]
        landmarks_3d = landmarks_3d[landmarks_mask.bool()]
        landmarks_2d_gt = landmarks_2d_gt[landmarks_mask.bool()]
        landmarks_3d_np = landmarks_3d.cpu().data.numpy().astype(np.float)
        landmarks_3d_np = np.expand_dims(landmarks_3d_np, axis=1)
        landmarks_2d_np = landmarks_2d.cpu().data.numpy().astype(np.float)
        landmarks_2d_np = np.expand_dims(landmarks_2d_np, axis=1)
        landmarks_2d_gt_np = landmarks_2d_gt.cpu().data.numpy().astype(np.float)
        intrinsic = np.array([[804.18848168,   0.        , 128.        ],
                                    [  0.        , 804.18848168, 128.        ],
                                    [  0.        ,   0.        ,   1.        ]])
        # intrinsic = np.array([[909.09090909,   0.        , 128.        ],
        #                             [  0.        , 909.09090909, 128.        ],
        #                             [  0.        ,   0.        ,   1.        ]])
        dist = np.zeros((5, 1))
        found, r, t = cv2.solvePnP(landmarks_3d_np, landmarks_2d_np, intrinsic, dist) #计算雷达相机外参,r-旋转向量，t-平移向量
        # found, r, t, _ = cv2.solvePnPRansac(landmarks_3d_np, landmarks_2d_np, intrinsic, dist, useExtrinsicGuess=True) #计算雷达相机外参,r-旋转向量，t-平移向量
        R = cv2.Rodrigues(r)[0] #旋转向量转旋转矩阵
        d2, _ = cv2.projectPoints(landmarks_3d_np, r, t, intrinsic, dist)#重投影验证W
        Tr_pred = np.identity(4)
        Tr_pred[:3, :3] = R
        Tr_pred[:3, 3] = t.T
        return Tr_pred


class P3PRansac_layer(nn.Module):
    def __init__(self, device, times=10):
        super(P3PRansac_layer, self).__init__()
        self.softArgmax = SpatialSoftArgmax2d(temperature=10, normalized_coordinates=False)
        self.times = times
        self.device = device
    
    def forward(self, pred_heatmaps, target_landmarks):
        heatmaps_shape = pred_heatmaps.shape
        landmarks_mask = target_landmarks[:, :, 0]
        landmarks_2d_gt = target_landmarks[:, :, 1 : 3]
        landmarks_3d = target_landmarks[:, :, 3:]
        landmarks_2d = self.softArgmax(pred_heatmaps)
        landmarks_2d = landmarks_2d[landmarks_mask.bool()]
        landmarks_3d = landmarks_3d[landmarks_mask.bool()]
        landmarks_2d_gt = landmarks_2d_gt[landmarks_mask.bool()]
        landmarks_3d_np = landmarks_3d.cpu().data.numpy().astype(np.float)
        landmarks_3d_np = np.expand_dims(landmarks_3d_np, axis=1)
        landmarks_2d_np = landmarks_2d.cpu().data.numpy().astype(np.float)
        landmarks_2d_np = np.expand_dims(landmarks_2d_np, axis=1)
        landmarks_2d_gt_np = landmarks_2d_gt.cpu().data.numpy().astype(np.float)
        intrinsic = np.array([[804.18848168,   0.        , 128.        ],
                                    [  0.        , 804.18848168, 128.        ],
                                    [  0.        ,   0.        ,   1.        ]])
        # intrinsic = np.array([[909.09090909,   0.        , 128.        ],
        #                             [  0.        , 909.09090909, 128.        ],
        #                             [  0.        ,   0.        ,   1.        ]])
        dist = np.zeros((5, 1))
        landmark_num = landmarks_3d_np.shape[0]
        inlier_num_max = -np.inf
        Tr_ret = None

        for i in range(self.times):
            random_indices = np.random.choice(landmark_num, 3, replace=False)
            found, r, t = cv2.solveP3P(landmarks_3d_np[random_indices], landmarks_2d_np[random_indices], intrinsic, dist, flags=cv2.SOLVEPNP_P3P) #计算雷达相机外参,r-旋转向量，t-平移向量
            # found, r, t, _ = cv2.solvePnPRansac(landmarks_3d_np, landmarks_2d_np, intrinsic, dist, useExtrinsicGuess=True) #计算雷达相机外参,r-旋转向量，t-平移向量
            if found:
                R = cv2.Rodrigues(r[0])[0] #旋转向量转旋转矩阵
                d2, _ = cv2.projectPoints(landmarks_3d_np, r[0], t[0], intrinsic, dist)#重投影验证
                # inlier_num = np.sum(np.linalg.norm(landmarks_2d_np - d2, axis=-1) < 7.3)
                inlier_num = np.sum(np.linalg.norm(landmarks_2d_np - d2, axis=-1) < 4)
                print(inlier_num)
                Tr_pred = np.identity(4)
                Tr_pred[:3, :3] = R
                Tr_pred[:3, 3] = t[0].T

                if inlier_num > inlier_num_max:
                    inlier_num_max = inlier_num
                    Tr_ret = Tr_pred

        return Tr_ret


class P3PRansac_4DOF_layer(nn.Module):
    def __init__(self, device, times=10):
        super(P3PRansac_4DOF_layer, self).__init__()
        self.softArgmax = SpatialSoftArgmax2d(temperature=10, normalized_coordinates=False)
        self.times = times
        self.device = device
    
    def forward(self, pred_heatmaps, target_landmarks, transformation_gt):
        heatmaps_shape = pred_heatmaps.shape
        landmarks_mask = target_landmarks[:, :, 0]
        landmarks_2d_gt = target_landmarks[:, :, 1 : 3]
        landmarks_3d = target_landmarks[:, :, 3:]
        landmarks_2d = self.softArgmax(pred_heatmaps)
        landmarks_2d = landmarks_2d[landmarks_mask.bool()]
        landmarks_3d = landmarks_3d[landmarks_mask.bool()]
        landmarks_2d_gt = landmarks_2d_gt[landmarks_mask.bool()]
        landmarks_3d_np = landmarks_3d.cpu().data.numpy().astype(np.float)
        landmarks_3d_np = np.expand_dims(landmarks_3d_np, axis=1)
        landmarks_2d_np = landmarks_2d.cpu().data.numpy().astype(np.float)
        landmarks_2d_np = np.expand_dims(landmarks_2d_np, axis=1)
        landmarks_2d_gt_np = landmarks_2d_gt.cpu().data.numpy().astype(np.float)
        transformation_gt_np = transformation_gt[0].detach().cpu().numpy()
        r = Rotation.from_matrix(transformation_gt_np[:3, :3])
        angle_gt = r.as_euler('xyz')
        intrinsic = np.array([[804.18848168,   0.        , 128.        ],
                                    [  0.        , 804.18848168, 128.        ],
                                    [  0.        ,   0.        ,   1.        ]])
        # intrinsic = np.array([[909.09090909,   0.        , 128.        ],
        #                             [  0.        , 909.09090909, 128.        ],
        #                             [  0.        ,   0.        ,   1.        ]])
        dist = np.zeros((5, 1))
        landmark_num = landmarks_3d_np.shape[0]
        inlier_num_max = -np.inf
        inlier_num_mask_ret = None
        Tr_ret = None

        for i in range(self.times):
            random_indices = np.random.choice(landmark_num, 3, replace=False)
            found, r, t = cv2.solveP3P(landmarks_3d_np[random_indices], landmarks_2d_np[random_indices], intrinsic, dist, flags=cv2.SOLVEPNP_P3P) #计算雷达相机外参,r-旋转向量，t-平移向量
            # found, r, t, _ = cv2.solvePnPRansac(landmarks_3d_np, landmarks_2d_np, intrinsic, dist, useExtrinsicGuess=True) #计算雷达相机外参,r-旋转向量，t-平移向量
            if found:
                R = cv2.Rodrigues(r[0])[0] #旋转向量转旋转矩阵
                d2, _ = cv2.projectPoints(landmarks_3d_np, r[0], t[0], intrinsic, dist)#重投影验证
                # inlier_num = np.sum(np.linalg.norm(landmarks_2d_np - d2, axis=-1) < 7.3)
                inlier_num = np.sum(np.linalg.norm(landmarks_2d_np - d2, axis=-1) < 4)
                inlier_num_mask = np.linalg.norm(landmarks_2d_np - d2, axis=-1) < 4
                print(inlier_num)
                Tr_pred = np.identity(4)
                Tr_pred[:3, :3] = R
                Tr_pred[:3, 3] = t[0].T

                if inlier_num > inlier_num_max:
                    inlier_num_max = inlier_num
                    inlier_num_mask_ret = inlier_num_mask.squeeze(1)
                    Tr_ret = Tr_pred

        # Optimization
        def fun_OptimizePose(pose, points_2d, points_3d, intrinsic):
            ax = math.pi / 2
            ay = 0
            az = pose[0] / 180 * math.pi
            tx = pose[1]
            ty = pose[2]
            tz = pose[3]
            t = np.array([tx, ty, tz])

            rz = Rotation.from_euler('xyz', [ax, ay, az])
            R = rz.as_matrix()
            points_3d_Rt = np.dot(R, points_3d.T).T + t
            points_2d_pred = 1 / np.expand_dims(points_3d_Rt[:, 2], axis=-1) * np.dot(intrinsic, points_3d_Rt.T).T
            points_2d_error = np.mean(np.linalg.norm(points_2d_pred[:, :2] - points_2d, axis=-1))
            # cost_value = points_2d_error + 10 * (np.abs(ax - np.pi / 2) + np.abs(ay))
            cost_value = points_2d_error
            # print(pose)
            # print(cost_value)
            return cost_value

        r = Rotation.from_matrix(Tr_ret[:3, :3])
        angle_ret = r.as_euler('xyz')
        t_ret = Tr_ret[:3, 3]
        # if angle_ret[0] < 0:
        #     angle_ret[2] = 0
        #     angle_ret[1] = 0
        #     angle_ret[0] = math.pi / 2
        pose0 = np.array([angle_ret[2] / math.pi * 180, t_ret[0], t_ret[1], t_ret[2]])
        res_1 = least_squares(fun_OptimizePose, pose0, args=(np.squeeze(landmarks_2d_np[inlier_num_mask_ret]), np.squeeze(landmarks_3d_np[inlier_num_mask_ret]), intrinsic))
        # res_1 = least_squares(fun_OptimizePose, pose0, args=(landmarks_2d, landmarks_3d, intrinsic))
        angle_opt = res_1.x[0]
        t_opt = res_1.x[1:] * np.array([1, 1, 1])

        r = Rotation.from_euler('xyz', [math.pi / 2, 0, angle_opt / 180 * math.pi])
        R_opt = r.as_matrix()
        Tr_ret = np.identity(4)
        Tr_ret[:3, :3] = R_opt
        Tr_ret[:3, 3] = t_opt.T

        return Tr_ret


class sweeney_2p_ransac_layer(nn.Module):
    def __init__(self, device, times=10, threshold=4):
        super(sweeney_2p_ransac_layer, self).__init__()
        self.softArgmax = SpatialSoftArgmax2d(temperature=10, normalized_coordinates=False)
        # self.linear = nn.Linear(256, 256)
        # self.softmax_w = nn.Parameter(torch.tensor(10.))
        self.softmax_w = nn.Parameter(torch.tensor(1.))
        self.softmax_b = nn.Parameter(torch.tensor(0.))
        self.register_parameter('softmax_w', self.softmax_w)
        self.register_parameter('softmax_b', self.softmax_b)
        self.threshold = threshold
        self.times = times
        self.device = device

    def AddPoseSolution(self, axis, model_point_1, model_point_2, image_ray_1, image_ray_2, ray_length_1, ray_length_2):
        point_in_image_space_1 = ray_length_1.unsqueeze(-1) * image_ray_1
        point_in_image_space_2 = ray_length_2.unsqueeze(-1) * image_ray_2
        image_points_diff = point_in_image_space_1 - point_in_image_space_2
        model_points_diff = model_point_1 - model_point_2
        norm_model_points_diff = model_points_diff / torch.norm(model_points_diff, dim=-1, keepdim=True)
        axis = axis.repeat([norm_model_points_diff.shape[0], norm_model_points_diff.shape[1], 1])
        basis_vector_2 = torch.cross(axis, norm_model_points_diff, dim=-1)
        norm_axis = axis / torch.norm(axis, dim=-1, keepdim=True)
        basis_vector_1 = torch.cross(basis_vector_2, norm_axis, dim=-1)
        res={}

        dp_1 = torch.sum(basis_vector_1 * image_points_diff, dim=-1)
        dp_2 = torch.sum(basis_vector_2 * image_points_diff, dim=-1)
        angle = torch.atan2(dp_2, dp_1)

        sin_tensor = torch.sin(angle)
        cos_tensor = torch.cos(angle)
        ones_tensor = torch.ones(angle.shape).to(self.device, dtype=torch.float32)
        zeros_tensor = torch.zeros(angle.shape).to(self.device, dtype=torch.float32)
        R = torch.stack([cos_tensor, -sin_tensor, zeros_tensor, sin_tensor, cos_tensor, zeros_tensor, zeros_tensor, zeros_tensor, ones_tensor], dim=-1)
        # R = torch.tensor([[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
        R = R.reshape((R.shape[0], R.shape[1], 3, 3))
        t = point_in_image_space_1 - torch.matmul(R.reshape((R.shape[0] * R.shape[1], R.shape[2], R.shape[3])), \
            model_point_1.reshape((model_point_1.shape[0] * model_point_1.shape[1], model_point_1.shape[2], 1))).reshape(model_point_1.shape)
        
        R_mean = torch.mean(R, dim=(0, 1))
        t_mean = torch.mean(t, dim=(0, 1))
        if torch.max(torch.isnan(R_mean)):
            raise NotImplementedError()
        return R, t, angle

    def TwoPointPoseCore(self, axis, model_point_1, model_point_2, image_ray_1, image_ray_2, landmarks_3d, landmarks_2d):
        ray_1_axis_dp = torch.matmul(image_ray_1, axis)
        ray_2_axis_dp = torch.matmul(image_ray_2, axis)
        model_diff_axis_dp = torch.matmul(model_point_1 - model_point_2, axis)
        m = model_diff_axis_dp / ray_1_axis_dp
        n = ray_2_axis_dp / ray_1_axis_dp
        # ray_dp = np.dot(image_ray_1, image_ray_2)
        ray_dp = torch.sum(image_ray_1 * image_ray_2, dim=-1)
        a = n * (n - 2.0 * ray_dp) + 1.0
        # if a < 0:
        #     raise NotImplementedError()
        b = 2.0 * m * (n - ray_dp)
        c = m * m - torch.norm(model_point_1 - model_point_2, dim=-1) ** 2
        if torch.max(b ** 2 - 4 * a * c < 0):
            return []
        myroot = -b + torch.sqrt(b ** 2 - 4 * a * c) / (2 * a)
        if torch.max(torch.isnan(myroot)):
            raise NotImplementedError()
        if torch.max(torch.isinf(myroot)):
            raise NotImplementedError()
        ray_distance = m + n * myroot
        res = self.AddPoseSolution(axis, model_point_1, model_point_2, image_ray_1, image_ray_2, ray_distance, myroot)
        return res

    def compute_2p_sweeney(self, point_3d_1, point_3d_2, bearing_1, bearing_2, axis, landmarks_3d, landmarks_2d):
        model_point_1 = point_3d_1
        model_point_2 = point_3d_2
        image_ray_1 = bearing_1
        image_ray_2 = bearing_2
        res = self.TwoPointPoseCore(axis, model_point_1, model_point_2, image_ray_1, image_ray_2, landmarks_3d, landmarks_2d)
        return res

    def forward(self, pred_heatmaps, target_landmarks, transformation_gt, noise_gamma=0, noise_beta=0):
        heatmaps_shape = pred_heatmaps.shape
        landmarks_mask = target_landmarks[:, :, 0]
        landmarks_2d_gt = target_landmarks[:, :, 1 : 3]
        landmarks_3d = target_landmarks[:, :, 3:]
        # pred_heatmaps = self.softmax_w * pred_heatmaps + self.softmax_b
        print("softmax_w, softmax_b:", self.softmax_w, self.softmax_b)
        # pred_heatmaps = F.softmax(pred_heatmaps.view(heatmaps_shape[0], heatmaps_shape[1], heatmaps_shape[2] * heatmaps_shape[3]), dim=2).view(heatmaps_shape)
        # pred_heatmaps_max_values = torch.max(pred_heatmaps.view(heatmaps_shape[0], heatmaps_shape[1], heatmaps_shape[2] * heatmaps_shape[3]), dim=2)[0]
        # pred_heatmaps_max_values = pred_heatmaps_max_values[landmarks_mask.bool()].unsqueeze(0)
        landmarks_2d = self.softArgmax(pred_heatmaps)
        landmarks_2d = landmarks_2d[landmarks_mask.bool()].unsqueeze(0)
        landmarks_3d = landmarks_3d[landmarks_mask.bool()].unsqueeze(0)
        landmarks_2d_gt = landmarks_2d_gt[landmarks_mask.bool()].unsqueeze(0)
        # uncertainty = uncertainty[landmarks_mask.bool()].unsqueeze(0)
        combinations = torch.combinations(torch.arange(landmarks_2d.shape[1]), r=2)
        combinations_valid_list = []
        for combination in combinations:
            epsilon = 2
            landmark_2d_diff = torch.norm(landmarks_2d[:, combination[0], :] - landmarks_2d[:, combination[1], :])
            if landmark_2d_diff > epsilon:
                combinations_valid_list.append(combination)
        if not combinations_valid_list:
            print("No enough inliers!")
            return []
        combinations = torch.stack(combinations_valid_list, dim=0)
        landmarks_2d_comb_2p = landmarks_2d[:, combinations, :]  # (B, N_comb, N_point, 2)
        landmarks_3d_comb_2p = landmarks_3d[:, combinations, :]  # (B, N_comb, N_point, 3)
        # uncertainty_comb_2p = uncertainty[:, combinations]  # (B, N_comb, N_point)
        # uncertainty_comb = uncertainty_comb_2p[:, :, 0] * uncertainty_comb_2p[:, :, 1]
        landmarks_2d_1 = landmarks_2d_comb_2p[:, :, 0, :]
        landmarks_2d_2 = landmarks_2d_comb_2p[:, :, 1, :]
        landmarks_3d_1 = landmarks_3d_comb_2p[:, :, 0, :]
        landmarks_3d_2 = landmarks_3d_comb_2p[:, :, 1, :]
        # pred_heatmaps_max_values_comb_2p = pred_heatmaps_max_values[:, combinations]
        # az_weight = pred_heatmaps_max_values_comb_2p[:, :, 0] * pred_heatmaps_max_values_comb_2p[:, :, 1]
        # az_weight = (az_weight - torch.min(az_weight, dim=-1)[0]) / (torch.max(az_weight, dim=-1)[0] - torch.min(az_weight, dim=-1)[0])
        # az_weight = F.softmax(az_weight, dim=-1)
        # print(max_indices)
        # landmarks_3d_1 = torch.tensor([[[-14.36853027,  52.28883362,  60.90942383]]]).to(self.device)
        # landmarks_3d_2 = torch.tensor([[[-14.57270813,  52.83177185,  40.47167969]]]).to(self.device)
        # landmarks_2d_1 = torch.tensor([[[116.89237882,  80.9138476 ]]]).to(self.device)
        # landmarks_2d_2 = torch.tensor([[[116.73847762,  96.72422596]]]).to(self.device)
        transformation_gt_np = transformation_gt[0].detach().cpu().numpy()
        r = Rotation.from_matrix(transformation_gt_np[:3, :3])
        angle_gt = r.as_euler('xyz')
        if self.training:
            ax = angle_gt[0]
            ay = angle_gt[1]
        else:
            # ax = math.pi / 2 + np.deg2rad(np.random.normal(loc=0, scale=noise_gamma, size=1))
            # ay = 0 + np.deg2rad(np.random.normal(loc=0, scale=noise_beta, size=1))
            ax = math.pi / 2
            ay = 0
            # ax = angle_gt[0] + noise_gamma
            # ay = angle_gt[1] + noise_beta
        # az = angle_gt[2]
        Rx = torch.tensor([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]]).to(self.device, dtype=torch.float32)
        Ry = torch.tensor([[math.cos(ay), 0, math.sin(ay)], [0, 1, 0], [-math.sin(ay), 0, math.cos(ay)]]).to(self.device, dtype=torch.float32)
        Rx = torch.unsqueeze(Rx, dim=0)
        Ry = torch.unsqueeze(Ry, dim=0)
        # Rz = torch.tensor([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]])
        # print(torch.matmul(Rz, torch.matmul(Ry, Rx)))
        # point_3d_new = np.dot(Ry, np.dot(Rx, point_3d.T)).T
        point_3d_1 = torch.matmul(Ry, torch.matmul(Rx, landmarks_3d_1.permute(0, 2, 1))).permute(0, 2, 1)
        point_3d_2 = torch.matmul(Ry, torch.matmul(Rx, landmarks_3d_2.permute(0, 2, 1))).permute(0, 2, 1)
        aug_ones = torch.ones((landmarks_2d_1.shape[0], landmarks_2d_1.shape[1], 1)).to(self.device, dtype=torch.float32)
        point_2d_1 = torch.cat([landmarks_2d_1, aug_ones], dim=-1)
        point_2d_2 = torch.cat([landmarks_2d_2, aug_ones], dim=-1)
        # intrinsic = torch.tensor([[1.20e+03, 0.00e+00, 1.28e+02],
        #                             [0.00e+00, 1.20e+03, 1.28e+02],
        #                             [0.00e+00, 0.00e+00, 1.00e+00]]).to(self.device, dtype=torch.float32)
        intrinsic = torch.tensor([[804.18848168,   0.        , 128.        ],
                                    [  0.        , 804.18848168, 128.        ],
                                    [  0.        ,   0.        ,   1.        ]]).to(self.device, dtype=torch.float32)
        # intrinsic = torch.tensor([[909.09090909,   0.        , 128.        ],
        #                             [  0.        , 909.09090909, 128.        ],
        #                             [  0.        ,   0.        ,   1.        ]]).to(self.device, dtype=torch.float32)
        intrinsic = torch.unsqueeze(intrinsic, dim=0)
        bearing_1 = torch.matmul(torch.inverse(intrinsic), point_2d_1.permute(0, 2, 1)).permute(0, 2, 1)
        bearing_2 = torch.matmul(torch.inverse(intrinsic), point_2d_2.permute(0, 2, 1)).permute(0, 2, 1)
        bearing_1 = bearing_1 / torch.norm(bearing_1, dim=-1, keepdim=True)
        bearing_2 = bearing_2 / torch.norm(bearing_2, dim=-1, keepdim=True)
        axis = torch.tensor([0., 0., 1.]).to(self.device, dtype=torch.float32)
        pose_pred = self.compute_2p_sweeney(point_3d_1, point_3d_2, bearing_1, bearing_2, axis, landmarks_3d, landmarks_2d)
        if not pose_pred:
            print("Not satisfying Delta-Condition!")
            return []
        Rz_pred = pose_pred[0]
        t_pred = pose_pred[1]
        az_pred = pose_pred[2]

        # Uncertainty weight
        # az_weight = torch.softmax(-uncertainty_comb, dim=-1)

        # Reprojection weight
        landmarks_3d_Rxy = torch.matmul(Ry, torch.matmul(Rx, landmarks_3d.permute(0, 2, 1))).permute(0, 2, 1)
        landmarks_3d_Rxyz = torch.matmul(Rz_pred.reshape((Rz_pred.shape[0] * Rz_pred.shape[1], Rz_pred.shape[2], Rz_pred.shape[3])).unsqueeze(1), \
                                landmarks_3d_Rxy.reshape((landmarks_3d_Rxy.shape[0] * landmarks_3d_Rxy.shape[1], landmarks_3d_Rxy.shape[2])).unsqueeze(-1)).squeeze(-1)  # matmul —— A shape: (B * N_comb, 1, 3, 3), B shape: (B * N_point, 3, 1), out: (B * N_comb, B * N_point, 3, 1)
        landmarks_3d_Rt = landmarks_3d_Rxyz + t_pred.reshape(t_pred.shape[0] * t_pred.shape[1], t_pred.shape[2]).repeat(landmarks_3d_Rxyz.shape[1], 1, 1).permute(1, 0, 2)  # 3d points in camera's coordinate
        landmarks_2d_reproject = 1 / landmarks_3d_Rt[:, :, 2].unsqueeze(-1) * torch.matmul(intrinsic, landmarks_3d_Rt.permute(0, 2, 1)).permute(0, 2, 1)  # repreject 3d points to 2d using camera's intrinsic
        if self.training:
            landmarks_2d_diff = landmarks_2d_reproject[:, :, :2] - landmarks_2d_gt.reshape(landmarks_2d_gt.shape[0] * landmarks_2d_gt.shape[1], landmarks_2d_gt.shape[2]).repeat(landmarks_2d_reproject.shape[0], 1, 1)
        else:
            landmarks_2d_diff = landmarks_2d_reproject[:, :, :2] - landmarks_2d.reshape(landmarks_2d.shape[0] * landmarks_2d.shape[1], landmarks_2d.shape[2]).repeat(landmarks_2d_reproject.shape[0], 1, 1)
        landmarks_2d_diff_norm = torch.norm(landmarks_2d_diff, dim=-1)

        inlier_num_max = -1
        Rz_ret = None
        t_ret = None
        az_ret = None

        for i in range(self.times):
            random_index = random.randint(0, combinations.shape[0] - 1)
            # Inlier mask for weight
            # inlier_num = torch.sum(landmarks_2d_diff_norm[random_index] < 7.3)
            # inlier_num = torch.sum(landmarks_2d_diff_norm[random_index] < 4)
            inlier_num = torch.sum(landmarks_2d_diff_norm[random_index] < self.threshold)
            if inlier_num > inlier_num_max:
                inlier_num_max = inlier_num
                Rz_ret = Rz_pred[0, random_index]
                t_ret = t_pred[0, random_index].unsqueeze(0)
                az_ret = az_pred[0, random_index]

        return Rz_ret, t_ret, az_ret


class sweeney_2p_layer(nn.Module):
    def __init__(self, device, threshold=4):
        super(sweeney_2p_layer, self).__init__()
        self.softArgmax = SpatialSoftArgmax2d(temperature=10, normalized_coordinates=False)
        # self.linear = nn.Linear(256, 256)
        # self.softmax_w = nn.Parameter(torch.tensor(10.))
        self.softmax_w = nn.Parameter(torch.tensor(1.))
        self.softmax_b = nn.Parameter(torch.tensor(0.))
        self.register_parameter('softmax_w', self.softmax_w)
        self.register_parameter('softmax_b', self.softmax_b)
        self.threshold = threshold
        self.device = device

    def AddPoseSolution(self, axis, model_point_1, model_point_2, image_ray_1, image_ray_2, ray_length_1, ray_length_2):
        point_in_image_space_1 = ray_length_1.unsqueeze(-1) * image_ray_1
        point_in_image_space_2 = ray_length_2.unsqueeze(-1) * image_ray_2
        image_points_diff = point_in_image_space_1 - point_in_image_space_2
        model_points_diff = model_point_1 - model_point_2
        norm_model_points_diff = model_points_diff / torch.norm(model_points_diff, dim=-1, keepdim=True)
        axis = axis.repeat([norm_model_points_diff.shape[0], norm_model_points_diff.shape[1], 1])
        basis_vector_2 = torch.cross(axis, norm_model_points_diff, dim=-1)
        norm_axis = axis / torch.norm(axis, dim=-1, keepdim=True)
        basis_vector_1 = torch.cross(basis_vector_2, norm_axis, dim=-1)
        res={}
        # if 0 > np.dot(basis_vector_1, basis_vector_1):
        #     return
        dp_1 = torch.sum(basis_vector_1 * image_points_diff, dim=-1)
        dp_2 = torch.sum(basis_vector_2 * image_points_diff, dim=-1)
        angle = torch.atan2(dp_2, dp_1)
        # if axis[0] == 1:
        #     R = torch.tensor([[1, 0, 0], [0, math.cos(angle), -math.sin(angle)], [0, math.sin(angle), math.cos(angle)]])
        # else:
        #     if axis[1] == 1:
        #         R = torch.tensor([[math.cos(angle), 0, math.sin(angle)], [0, 1, 0], [-math.sin(angle), 0, math.cos(angle)]])
        #     else:
        #         if axis[2] == 1:
        #             R = torch.tensor([[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
        #         else:
        #             print("axis error")
        #             return
        sin_tensor = torch.sin(angle)
        cos_tensor = torch.cos(angle)
        ones_tensor = torch.ones(angle.shape).to(self.device, dtype=torch.float32)
        zeros_tensor = torch.zeros(angle.shape).to(self.device, dtype=torch.float32)
        R = torch.stack([cos_tensor, -sin_tensor, zeros_tensor, sin_tensor, cos_tensor, zeros_tensor, zeros_tensor, zeros_tensor, ones_tensor], dim=-1)
        # R = torch.tensor([[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
        R = R.reshape((R.shape[0], R.shape[1], 3, 3))
        t = point_in_image_space_1 - torch.matmul(R.reshape((R.shape[0] * R.shape[1], R.shape[2], R.shape[3])), \
            model_point_1.reshape((model_point_1.shape[0] * model_point_1.shape[1], model_point_1.shape[2], 1))).reshape(model_point_1.shape)
        
        R_mean = torch.mean(R, dim=(0, 1))
        t_mean = torch.mean(t, dim=(0, 1))
        if torch.max(torch.isnan(R_mean)):
            raise NotImplementedError()
        return R, t, angle

    def TwoPointPoseCore(self, axis, model_point_1, model_point_2, image_ray_1, image_ray_2, landmarks_3d, landmarks_2d):
        ray_1_axis_dp = torch.matmul(image_ray_1, axis)
        ray_2_axis_dp = torch.matmul(image_ray_2, axis)
        model_diff_axis_dp = torch.matmul(model_point_1 - model_point_2, axis)
        m = model_diff_axis_dp / ray_1_axis_dp
        n = ray_2_axis_dp / ray_1_axis_dp
        # ray_dp = np.dot(image_ray_1, image_ray_2)
        ray_dp = torch.sum(image_ray_1 * image_ray_2, dim=-1)
        a = n * (n - 2.0 * ray_dp) + 1.0
        # if a < 0:
        #     raise NotImplementedError()
        b = 2.0 * m * (n - ray_dp)
        c = m * m - torch.norm(model_point_1 - model_point_2, dim=-1) ** 2
        if torch.max(b ** 2 - 4 * a * c < 0):
            return []
        myroot = -b + torch.sqrt(b ** 2 - 4 * a * c) / (2 * a)
        if torch.max(torch.isnan(myroot)):
            raise NotImplementedError()
        if torch.max(torch.isinf(myroot)):
            raise NotImplementedError()
        ray_distance = m + n * myroot
        res = self.AddPoseSolution(axis, model_point_1, model_point_2, image_ray_1, image_ray_2, ray_distance, myroot)
        return res

    def compute_2p_sweeney(self, point_3d_1, point_3d_2, bearing_1, bearing_2, axis, landmarks_3d, landmarks_2d):
        model_point_1 = point_3d_1
        model_point_2 = point_3d_2
        image_ray_1 = bearing_1
        image_ray_2 = bearing_2

        Epsilon = 1e-9
        # if abs(torch.norm(image_ray_1) ** 2 - 1) > Epsilon or abs(torch.norm(image_ray_2) ** 2 - 1) > Epsilon:
        #     raise NotImplementedError()
        #     return
        # if abs(torch.matmul(image_ray_1, axis)) < Epsilon:
        #     raise NotImplementedError()
        #     if abs(torch.matmul(image_ray_2, axis)) > Epsilon:
        #         res = TwoPointPoseCore(axis, model_point_2, model_point_1, image_ray_2, image_ray_1)
        #     else:
        #         return
        # else:
        #     res = TwoPointPoseCore(axis, model_point_1, model_point_2, image_ray_1, image_ray_2)
        res = self.TwoPointPoseCore(axis, model_point_1, model_point_2, image_ray_1, image_ray_2, landmarks_3d, landmarks_2d)
        return res

    def forward(self, pred_heatmaps, target_landmarks, transformation_gt, noise_gamma=0, noise_beta=0):
        heatmaps_shape = pred_heatmaps.shape
        landmarks_mask = target_landmarks[:, :, 0]
        landmarks_2d_gt = target_landmarks[:, :, 1 : 3]
        landmarks_3d = target_landmarks[:, :, 3:]
        # pred_heatmaps = self.softmax_w * pred_heatmaps + self.softmax_b
        print("softmax_w, softmax_b:", self.softmax_w, self.softmax_b)
        # pred_heatmaps = F.softmax(pred_heatmaps.view(heatmaps_shape[0], heatmaps_shape[1], heatmaps_shape[2] * heatmaps_shape[3]), dim=2).view(heatmaps_shape)
        # pred_heatmaps_max_values = torch.max(pred_heatmaps.view(heatmaps_shape[0], heatmaps_shape[1], heatmaps_shape[2] * heatmaps_shape[3]), dim=2)[0]
        # pred_heatmaps_max_values = pred_heatmaps_max_values[landmarks_mask.bool()].unsqueeze(0)
        landmarks_2d = self.softArgmax(pred_heatmaps)
        landmarks_2d = landmarks_2d[landmarks_mask.bool()].unsqueeze(0)
        landmarks_3d = landmarks_3d[landmarks_mask.bool()].unsqueeze(0)
        landmarks_2d_gt = landmarks_2d_gt[landmarks_mask.bool()].unsqueeze(0)
        # uncertainty = uncertainty[landmarks_mask.bool()].unsqueeze(0)
        combinations = torch.combinations(torch.arange(landmarks_2d.shape[1]), r=2)
        combinations_valid_list = []
        for combination in combinations:
            epsilon = 2
            landmark_2d_diff = torch.norm(landmarks_2d[:, combination[0], :] - landmarks_2d[:, combination[1], :])
            if landmark_2d_diff > epsilon:
                combinations_valid_list.append(combination)
        if not combinations_valid_list:
            print("No enough inliers!")
            return []
        combinations = torch.stack(combinations_valid_list, dim=0)
        landmarks_2d_comb_2p = landmarks_2d[:, combinations, :]  # (B, N_comb, N_point, 2)
        landmarks_3d_comb_2p = landmarks_3d[:, combinations, :]  # (B, N_comb, N_point, 3)
        # uncertainty_comb_2p = uncertainty[:, combinations]  # (B, N_comb, N_point)
        # uncertainty_comb = uncertainty_comb_2p[:, :, 0] * uncertainty_comb_2p[:, :, 1]
        landmarks_2d_1 = landmarks_2d_comb_2p[:, :, 0, :]
        landmarks_2d_2 = landmarks_2d_comb_2p[:, :, 1, :]
        landmarks_3d_1 = landmarks_3d_comb_2p[:, :, 0, :]
        landmarks_3d_2 = landmarks_3d_comb_2p[:, :, 1, :]
        # pred_heatmaps_max_values_comb_2p = pred_heatmaps_max_values[:, combinations]
        # az_weight = pred_heatmaps_max_values_comb_2p[:, :, 0] * pred_heatmaps_max_values_comb_2p[:, :, 1]
        # az_weight = (az_weight - torch.min(az_weight, dim=-1)[0]) / (torch.max(az_weight, dim=-1)[0] - torch.min(az_weight, dim=-1)[0])
        # az_weight = F.softmax(az_weight, dim=-1)
        # print(max_indices)
        # landmarks_3d_1 = torch.tensor([[[-14.36853027,  52.28883362,  60.90942383]]]).to(self.device)
        # landmarks_3d_2 = torch.tensor([[[-14.57270813,  52.83177185,  40.47167969]]]).to(self.device)
        # landmarks_2d_1 = torch.tensor([[[116.89237882,  80.9138476 ]]]).to(self.device)
        # landmarks_2d_2 = torch.tensor([[[116.73847762,  96.72422596]]]).to(self.device)
        transformation_gt_np = transformation_gt[0].detach().cpu().numpy()
        r = Rotation.from_matrix(transformation_gt_np[:3, :3])
        angle_gt = r.as_euler('xyz')
        if self.training:
            ax = angle_gt[0]
            ay = angle_gt[1]
        else:
            # ax = math.pi / 2 + np.deg2rad(np.random.normal(loc=0, scale=noise_gamma, size=1))
            # ay = 0 + np.deg2rad(np.random.normal(loc=0, scale=noise_beta, size=1))
            ax = math.pi / 2
            ay = 0
            # ax = angle_gt[0] + noise_gamma
            # ay = angle_gt[1] + noise_beta
            # ax = angle_gt[0]
            # ay = angle_gt[1]
        # az = angle_gt[2]
        Rx = torch.tensor([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]]).to(self.device, dtype=torch.float32)
        Ry = torch.tensor([[math.cos(ay), 0, math.sin(ay)], [0, 1, 0], [-math.sin(ay), 0, math.cos(ay)]]).to(self.device, dtype=torch.float32)
        Rx = torch.unsqueeze(Rx, dim=0)
        Ry = torch.unsqueeze(Ry, dim=0)
        # Rz = torch.tensor([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]])
        # print(torch.matmul(Rz, torch.matmul(Ry, Rx)))
        # point_3d_new = np.dot(Ry, np.dot(Rx, point_3d.T)).T
        point_3d_1 = torch.matmul(Ry, torch.matmul(Rx, landmarks_3d_1.permute(0, 2, 1))).permute(0, 2, 1)
        point_3d_2 = torch.matmul(Ry, torch.matmul(Rx, landmarks_3d_2.permute(0, 2, 1))).permute(0, 2, 1)
        aug_ones = torch.ones((landmarks_2d_1.shape[0], landmarks_2d_1.shape[1], 1)).to(self.device, dtype=torch.float32)
        point_2d_1 = torch.cat([landmarks_2d_1, aug_ones], dim=-1)
        point_2d_2 = torch.cat([landmarks_2d_2, aug_ones], dim=-1)
        # intrinsic = torch.tensor([[1.20e+03, 0.00e+00, 1.28e+02],
        #                             [0.00e+00, 1.20e+03, 1.28e+02],
        #                             [0.00e+00, 0.00e+00, 1.00e+00]]).to(self.device, dtype=torch.float32)
        intrinsic = torch.tensor([[804.18848168,   0.        , 128.        ],
                                    [  0.        , 804.18848168, 128.        ],
                                    [  0.        ,   0.        ,   1.        ]]).to(self.device, dtype=torch.float32)
        # intrinsic = torch.tensor([[909.09090909,   0.        , 128.        ],
        #                             [  0.        , 909.09090909, 128.        ],
        #                             [  0.        ,   0.        ,   1.        ]]).to(self.device, dtype=torch.float32)
        intrinsic = torch.unsqueeze(intrinsic, dim=0)
        bearing_1 = torch.matmul(torch.inverse(intrinsic), point_2d_1.permute(0, 2, 1)).permute(0, 2, 1)
        bearing_2 = torch.matmul(torch.inverse(intrinsic), point_2d_2.permute(0, 2, 1)).permute(0, 2, 1)
        bearing_1 = bearing_1 / torch.norm(bearing_1, dim=-1, keepdim=True)
        bearing_2 = bearing_2 / torch.norm(bearing_2, dim=-1, keepdim=True)
        axis = torch.tensor([0., 0., 1.]).to(self.device, dtype=torch.float32)
        pose_pred = self.compute_2p_sweeney(point_3d_1, point_3d_2, bearing_1, bearing_2, axis, landmarks_3d, landmarks_2d)
        if not pose_pred:
            print("Not satisfying Delta-Condition!")
            return []
        Rz_pred = pose_pred[0]
        t_pred = pose_pred[1]
        az_pred = pose_pred[2]

        # Uncertainty weight
        # az_weight = torch.softmax(-uncertainty_comb, dim=-1)

        # Reprojection weight
        landmarks_3d_Rxy = torch.matmul(Ry, torch.matmul(Rx, landmarks_3d.permute(0, 2, 1))).permute(0, 2, 1)
        landmarks_3d_Rxyz = torch.matmul(Rz_pred.reshape((Rz_pred.shape[0] * Rz_pred.shape[1], Rz_pred.shape[2], Rz_pred.shape[3])).unsqueeze(1), \
                                landmarks_3d_Rxy.reshape((landmarks_3d_Rxy.shape[0] * landmarks_3d_Rxy.shape[1], landmarks_3d_Rxy.shape[2])).unsqueeze(-1)).squeeze(-1)  # matmul —— A shape: (B * N_comb, 1, 3, 3), B shape: (B * N_point, 3, 1), out: (B * N_comb, B * N_point, 3, 1)
        landmarks_3d_Rt = landmarks_3d_Rxyz + t_pred.reshape(t_pred.shape[0] * t_pred.shape[1], t_pred.shape[2]).repeat(landmarks_3d_Rxyz.shape[1], 1, 1).permute(1, 0, 2)  # 3d points in camera's coordinate
        landmarks_2d_reproject = 1 / landmarks_3d_Rt[:, :, 2].unsqueeze(-1) * torch.matmul(intrinsic, landmarks_3d_Rt.permute(0, 2, 1)).permute(0, 2, 1)  # repreject 3d points to 2d using camera's intrinsic
        if self.training:
            landmarks_2d_diff = landmarks_2d_reproject[:, :, :2] - landmarks_2d_gt.reshape(landmarks_2d_gt.shape[0] * landmarks_2d_gt.shape[1], landmarks_2d_gt.shape[2]).repeat(landmarks_2d_reproject.shape[0], 1, 1)
        else:
            landmarks_2d_diff = landmarks_2d_reproject[:, :, :2] - landmarks_2d.reshape(landmarks_2d.shape[0] * landmarks_2d.shape[1], landmarks_2d.shape[2]).repeat(landmarks_2d_reproject.shape[0], 1, 1)
        distance_2d_mean = torch.mean(torch.norm(landmarks_2d_diff, dim=-1), dim=-1)

        # Inlier mask for weight
        landmarks_2d_diff_norm = torch.norm(landmarks_2d_diff, dim=-1)
        # landmarks_2d_diff_norm = 7.3 - landmarks_2d_diff_norm  # for refinement
        # landmarks_2d_diff_norm = 4 - landmarks_2d_diff_norm  # best for simulation
        # landmarks_2d_diff_norm = 7.3 - landmarks_2d_diff_norm  # best for clean clinical
        # landmarks_2d_diff_norm = 2 - landmarks_2d_diff_norm  # hyper eval
        landmarks_2d_diff_norm = self.threshold - landmarks_2d_diff_norm  # hyper eval
        inlier_mask = landmarks_2d_diff_norm > 0
        landmarks_2d_diff_norm = landmarks_2d_diff_norm * inlier_mask
        distance_2d_sum = torch.sum(landmarks_2d_diff_norm, dim=-1)
        # distance_2d_sum = distance_2d_sum / torch.norm(distance_2d_sum)
        if self.training:
            temperature = 10
        else:
            temperature = 0.1
        distance_2d_sum = distance_2d_sum / temperature
        az_weight = F.softmax(distance_2d_sum, dim=0)
        az_weight = az_weight.reshape(az_pred.shape)

        # temperature = 1
        # distance_2d_with_temp = -distance_2d_mean / temperature
        # az_weight = F.softmax(distance_2d_with_temp, dim=0)
        # az_weight = az_weight.reshape(az_pred.shape)

        az_pred_mean = torch.sum(az_weight * az_pred, dim=-1)
        # print("az_weight:", az_weight)
        t_pred_mean = torch.sum(az_weight.repeat(3, 1, 1).permute(1, 2, 0) * t_pred, dim=1)

        # # Optimization
        # def fun_OptimizePose(pose, points_2d, points_3d, intrinsic, Rx, Ry):
        #     ax = pose[0]
        #     ay = pose[1]
        #     az = pose[2]
        #     tx = pose[3]
        #     ty = pose[4]
        #     tz = pose[5] * 1000
        #     t = np.array([tx, ty, tz])

        #     rz = Rotation.from_euler('xyz', [ax, ay, az])
        #     # Rz = rz.as_matrix()
        #     # R = np.dot(Rz, np.dot(Ry, Rx))
        #     R = rz.as_matrix()
        #     # Rx = np.array([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]])
        #     # Ry = np.array([[math.cos(ay), 0, math.sin(ay)], [0, 1, 0], [-math.sin(ay), 0, math.cos(ay)]])
        #     # Rz = np.array([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]])
        #     # R = np.dot(Rz, np.dot(Ry, Rx))
        #     points_3d_Rt = np.dot(R, points_3d.T).T + t
        #     points_2d_pred = 1 / np.expand_dims(points_3d_Rt[:, 2], axis=-1) * np.dot(intrinsic, points_3d_Rt.T).T
        #     points_2d_error = np.mean(np.linalg.norm(points_2d_pred[:, :2] - points_2d, axis=-1))
        #     # cost_value = points_2d_error + 10 * (np.abs(ax - np.pi / 2) + np.abs(ay))
        #     cost_value = points_2d_error
        #     # print(pose)
        #     # print(cost_value)
        #     return cost_value
        
        # inliner_treshold = 3
        # landmarks_2d_diff_norm = torch.norm(landmarks_2d_diff, dim=-1)
        # landmarks_2d_diff_norm_np = landmarks_2d_diff_norm.data.cpu().numpy()
        # landmarks_2d_diff_count = np.count_nonzero(landmarks_2d_diff_norm_np < inliner_treshold, axis=-1)
        # best_comp_index = np.argmax(landmarks_2d_diff_count)
        # # best_comp_index = torch.argmin(distance_2d_mean)

        # match_score = landmarks_2d_diff_norm[best_comp_index]
        # inliners_mask = match_score < inliner_treshold
        # landmarks_2d_inliner_np = landmarks_2d[0][inliners_mask].data.cpu().numpy()
        # landmarks_3d_inliner_np = landmarks_3d[0][inliners_mask].data.cpu().numpy()
        # intrinsic_np = intrinsic[0].data.cpu().numpy()
        # Rx_np = Rx[0].data.cpu().numpy()
        # Ry_np = Ry[0].data.cpu().numpy()
        # # az_pred_np = az_pred[0][best_comp_index].data.cpu().numpy()
        # # t_pred_np = t_pred[0][best_comp_index].data.cpu().numpy()
        # az_pred_np = az_pred_mean[0].data.cpu().numpy()
        # t_pred_np = t_pred_mean[0].data.cpu().numpy()

        # pose0 = np.array([np.pi / 2, 0., az_pred_np, t_pred_np[0], t_pred_np[1], t_pred_np[2] / 1000])
        # res_1 = least_squares(fun_OptimizePose, pose0, args=(landmarks_2d_inliner_np, landmarks_3d_inliner_np, intrinsic_np, Rx_np, Ry_np))
        # az_pred_mean = torch.tensor(res_1.x[2]).to(self.device, dtype=torch.float32)
        # t_pred_mean = torch.tensor(res_1.x[3:] * np.array([1, 1, 1000])).to(self.device, dtype=torch.float32).unsqueeze(0)

        return Rz_pred, t_pred_mean, az_pred_mean


            