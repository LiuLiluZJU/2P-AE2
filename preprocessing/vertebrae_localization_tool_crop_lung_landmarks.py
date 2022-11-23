import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy.ndimage as ndimage
from SiddonGpuPy import pySiddonGpu
import cv2
from scipy.spatial.transform import Rotation
import transforms3d
import random
import json
import os
import h5py


def get_rotation_mat_single_axis( axis, angle ):

    """It computes the 3X3 rotation matrix relative to a single rotation of angle(rad) 
    about the axis(string 'x', 'y', 'z') for a righr handed CS"""

    if axis == 'x' : return np.array(([1,0,0],[0, np.cos(angle), -np.sin(angle)],[0, np.sin(angle), np.cos(angle)]))

    if axis == 'y' : return np.array(([np.cos(angle),0,np.sin(angle)],[0, 1, 0],[-np.sin(angle), 0, np.cos(angle)]))

    if axis == 'z' : return np.array(([np.cos(angle),-np.sin(angle),0],[np.sin(angle), np.cos(angle), 0],[0, 0, 1]))


def get_rigid_motion_mat_from_euler( alpha, axis_1, beta, axis_2, gamma, axis_3, t_x, t_y, t_z ):
    
    """It computes the 4X4 rigid motion matrix given a sequence of 3 Euler angles about the 3 axes 1,2,3 
    and the translation vector t_x, t_y, t_z"""

    rot1 = get_rotation_mat_single_axis( axis_1, alpha )
    rot2 = get_rotation_mat_single_axis( axis_2, beta )
    rot3 = get_rotation_mat_single_axis( axis_3, gamma )

    rot_mat = np.dot(rot1, np.dot(rot2,rot3))

    t = np.array(([t_x], [t_y], [t_z]))

    output = np.concatenate((rot_mat, t), axis = 1)

    return np.concatenate((output, np.array([[0.,0.,0.,1.]])), axis = 0)


def dcm2quat(R):
    epsilon = 1e-5
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    assert trace > -1

    if np.fabs(trace + 1) < epsilon:
        if np.argmax([R[0, 0], R[1, 1], R[2, 2]]) == 0:
            t = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
            q0 = (R[2, 1] - R[1, 2]) / t
            q1 = t / 4
            q2 = (R[0, 2] + R[2, 0]) / t
            q3 = (R[0, 1] + R[1, 0]) / t
        elif np.argmax([R[0, 0], R[1, 1], R[2, 2]]) == 1:
            t = np.sqrt(1 - R[0, 0] + R[1, 1] - R[2, 2])
            q0 = (R[0, 2] - R[2, 0]) / t
            q1 = (R[0, 1] + R[1, 0]) / t
            q2 = t / 4
            q3 = (R[2, 1] + R[1, 2]) / t
        else:
            t = np.sqrt(1 - R[0, 0] - R[1, 1] + R[2, 2])
            q0 = (R[1, 0] - R[0, 1]) / t
            q1 = (R[0, 2] + R[2, 0]) / t
            q2 = (R[1, 2] - R[2, 1]) / t
            q3 = t / 4
    else:
        q0 = np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2
        q1 = (R[2, 1] - R[1, 2]) / (4 * q0)
        q2 = (R[0, 2] - R[2, 0]) / (4 * q0)
        q3 = (R[1, 0] - R[0, 1]) / (4 * q0)
    
    return np.array([q1, q2, q3, q0])


def get_2d_annotation(fiducial_file, transformation, distance, size_3d, spacing_3d, size_2d, spacing_2d):

    """Vertebrae localization"""

    with open(fiducial_file, 'r') as f:
        fiducial_points_dict = json.load(f)

    point_2d_list = []
    label_list = []
    for fiducial_point_dict in fiducial_points_dict:
        fiducial_point = np.array([fiducial_point_dict['Z'], fiducial_point_dict['Y'], size_3d[2] * spacing_3d[2] - fiducial_point_dict['X']])
        label = fiducial_point_dict['label']
        label_list.append(label)
        X = fiducial_point - size_3d / 2 * spacing_3d
        d = distance
        c = d / 2
        K = np.array([[d, 0, 0],
                        [0, d, 0],
                        [0, 0, 1]])
        h = np.array([[0, 0, c]]).T

        Tr_view_inv = np.linalg.inv(transformation)
        R_view = Tr_view_inv[0: 3, 0: 3]
        t_view = Tr_view_inv[:3, 3].T
        x_dot = np.dot(K, np.dot(np.hstack([R_view, np.array([t_view]).T + h]), np.append(X, 1).T))
        x_dot = (x_dot / x_dot[-1])[:2]
        point_dot = x_dot[:2]
        point_2d = point_dot / spacing_2d + size_2d / 2
        point_2d_list.append(point_2d)
        print("point_2d_gt:", point_2d)
    point_2d_list = np.asarray(point_2d_list)
    return point_2d_list, label_list


def get_2d_3d_annotation(fiducial_file, transformation, distance, size_3d, spacing_3d, size_2d, spacing_2d):

    """Vertebrae localization"""

    with open(fiducial_file, 'r') as f:
        fiducial_points_dict = json.load(f)

    Tr_view_inv = np.linalg.inv(transformation)
    center = size_3d / 2 * spacing_3d
    source = np.zeros(3, dtype=np.float32)
    source[0] = center[0]
    source[1] = center[1]
    source[2] = center[2] - distance / 2.
    R_cw = Tr_view_inv[:3, :3]
    t_cw = Tr_view_inv[:3, 3] + center - source
    point_2d_list = []
    point_3d_list = []
    label_list = []
    for fiducial_point_dict in fiducial_points_dict:
        fiducial_point = np.array([fiducial_point_dict['Z'], fiducial_point_dict['Y'], size_3d[2] * spacing_3d[2] - fiducial_point_dict['X']])
        label = fiducial_point_dict['label']
        label_list.append(label)
        X_w = fiducial_point - center
        X_c = np.dot(R_cw, X_w) + t_cw
        K = np.array([[distance / spacing_2d[0], 0, size_2d[0] / 2],
                        [0, distance / spacing_2d[1], size_2d[1] / 2],
                        [0, 0, 1]])
        point_2d = 1 / X_c[2] * np.dot(K, X_c)[:2]
        point_2d_list.append(point_2d)
        point_3d_list.append(X_w)
        print("point_2d_new:", point_2d)
    point_2d_list = np.asarray(point_2d_list)
    point_3d_list = np.asarray(point_3d_list)
    T_cw = np.identity(4)
    T_cw[:3, :3] = R_cw
    T_cw[:3, 3] = t_cw
    return point_3d_list, point_2d_list, label_list, T_cw, center, source


def get_2d_3d_annotation_lung(fiducials_3d, transformation, distance, size_3d, spacing_3d, size_2d, spacing_2d):

    """Vertebrae localization"""

    Tr_view_inv = np.linalg.inv(transformation)
    center = size_3d / 2 * spacing_3d
    source = np.zeros(3, dtype=np.float32)
    source[0] = center[0]
    source[1] = center[1]
    source[2] = center[2] - distance / 2.
    R_cw = Tr_view_inv[:3, :3]
    t_cw = Tr_view_inv[:3, 3] + center - source
    point_2d_list = []
    point_3d_list = []
    label_list = []
    label = 0
    for fiducial_3d in fiducials_3d:
        label_list.append(label)
        label += 1
        X_w = fiducial_3d - center
        X_c = np.dot(R_cw, X_w) + t_cw
        K = np.array([[distance / spacing_2d[0], 0, size_2d[0] / 2],
                        [0, distance / spacing_2d[1], size_2d[1] / 2],
                        [0, 0, 1]])
        point_2d = 1 / X_c[2] * np.dot(K, X_c)[:2]
        point_2d_list.append(point_2d)
        point_3d_list.append(X_w)
        print("point_2d_new:", point_2d)
    point_2d_list = np.asarray(point_2d_list)
    point_3d_list = np.asarray(point_3d_list)
    T_cw = np.identity(4)
    T_cw[:3, :3] = R_cw
    T_cw[:3, 3] = t_cw
    return point_3d_list, point_2d_list, label_list, T_cw, center, source


def AddPoseSolution(axis, model_point_1, model_point_2, image_ray_1, image_ray_2, ray_length_1, ray_length_2):
    point_in_image_space_1 = ray_length_1 * image_ray_1
    point_in_image_space_2 = ray_length_2 * image_ray_2
    image_points_diff = point_in_image_space_1 - point_in_image_space_2
    model_points_diff = model_point_1 - model_point_2
    norm_model_points_diff = model_points_diff / np.linalg.norm(model_points_diff)
    basis_vector_2 = np.cross(axis, norm_model_points_diff)
    norm_axis = axis / np.linalg.norm(axis)
    basis_vector_1 = np.cross(basis_vector_2, norm_axis)
    res={}
    if 0 > np.dot(basis_vector_1, basis_vector_1):
        return
    dp_1 = np.dot(basis_vector_1,image_points_diff)
    dp_2 = np.dot(basis_vector_2,image_points_diff)
    angle = np.arctan2(dp_2, dp_1)
    if axis[0] == 1:
        R = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
    else:
        if axis[1] == 1:
            R = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
        else:
            if axis[2] == 1:
                R = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
            else:
                print("axis error")
                return
    t = point_in_image_space_1 - np.dot(R, model_point_1)
    T = np.identity(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def TwoPointPoseCore(axis, model_point_1, model_point_2, image_ray_1, image_ray_2):
    ray_1_axis_dp = np.dot(image_ray_1, axis)
    ray_2_axis_dp = np.dot(image_ray_2, axis)
    model_diff_axis_dp = np.dot(model_point_1 - model_point_2, axis)
    m = model_diff_axis_dp / ray_1_axis_dp
    n = ray_2_axis_dp / ray_1_axis_dp
    ray_dp = np.dot(image_ray_1, image_ray_2)
    a = n * (n - 2.0 * ray_dp) + 1.0
    b = 2.0 * m * (n - ray_dp)
    c = m * m - np.linalg.norm(model_point_1 - model_point_2) ** 2
    res=[]
    if b ** 2 - 4 * a * c < 0:
        return
    # myroots = roots([a,b,c])
    myroots = [(-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a), (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)]
    num_solutions = 0
    for i in range(len(myroots)):
        myroot = myroots[i]
        if myroot > 0:
            ray_distance = m + n * myroot
            if ray_distance > 0:
                this_res = AddPoseSolution(axis, model_point_1, model_point_2, image_ray_1, image_ray_2, ray_distance, myroot)
                res.append(this_res)
                num_solutions += 1
    return res


def compute_2p_sweeney(point_3d_1, point_3d_2, bearing_1, bearing_2, axis):
    model_point_1 = point_3d_1
    model_point_2 = point_3d_2
    image_ray_1 = bearing_1
    image_ray_2 = bearing_2

    Epsilon = 1e-9
    if abs(np.linalg.norm(image_ray_1) ** 2 - 1) > Epsilon or abs(np.linalg.norm(image_ray_2) ** 2 - 1) > Epsilon:
        return
    if abs(np.dot(image_ray_1, axis)) < Epsilon:
        if abs(np.dot(image_ray_2, axis)) > Epsilon:
            res = TwoPointPoseCore(axis, model_point_2, model_point_1, image_ray_2, image_ray_1)
        else:
            return
    else:
        res = TwoPointPoseCore(axis, model_point_1, model_point_2, image_ray_1, image_ray_2)
    return res


def solve2P(point_3d, point_2d, intricsic, dist, Tr):
    r = Rotation.from_matrix(Tr[:3, :3])
    angle = r.as_euler('xyz')
    ax = angle[0]
    ay = angle[1]
    az = angle[2]
    Rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    Ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    Rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    print(np.dot(Rz, np.dot(Ry, Rx)))
    point_3d_new = np.dot(Ry, np.dot(Rx, point_3d.T)).T
    point_3d_1 = point_3d_new[0]
    point_3d_2 = point_3d_new[1]
    point_2d_1 = np.append(point_2d[0], 1)
    point_2d_2 = np.append(point_2d[1], 1)
    bearing_1 = np.dot(np.linalg.inv(intricsic), point_2d_1)
    bearing_2 = np.dot(np.linalg.inv(intricsic), point_2d_2)
    bearing_1 = bearing_1 / np.linalg.norm(bearing_1)
    bearing_2 = bearing_2 / np.linalg.norm(bearing_2)
    axis = np.array([0, 0, 1])
    result = compute_2p_sweeney(point_3d_1, point_3d_2, bearing_1, bearing_2, axis)[0]
    # print(euler)
    print(result)
    r_res = Rotation.from_matrix(result[:3, :3])
    angle_res = r_res.as_euler('xyz')
    print(angle_res)
    print(Tr)

    return result

max_error = 0
def solvePnP(fiducial_file, transformation, distance, size_3d, spacing_3d, size_2d, spacing_2d):

    with open(fiducial_file, 'r') as f:
        fiducial_points_dict = json.load(f)

    point_2d_list = []
    point_3d_list = []
    label_list = []
    for fiducial_point_dict in fiducial_points_dict:
        fiducial_point = np.array([fiducial_point_dict['Z'], fiducial_point_dict['Y'], size_3d[2] * spacing_3d[2] - fiducial_point_dict['X']])
        label = fiducial_point_dict['label']
        label_list.append(label)
        Tr_view_inv = np.linalg.inv(transformation)
        X_transformed = np.dot(Tr_view_inv, np.append(fiducial_point - size_3d / 2 * spacing_3d, 1))[:3] + size_3d / 2 * spacing_3d
        center = size_3d / 2 * spacing_3d
        source = np.zeros(3, dtype=np.float32)
        source[0] = center[0]
        source[1] = center[1]
        source[2] = center[2] - distance / 2.
        X_source = X_transformed - source
        A = fiducial_point - size_3d / 2 * spacing_3d
        B = size_3d / 2 * spacing_3d - source
        # print(np.dot(Tr_view_inv, (A + np.dot(np.linalg.inv(Tr_view_inv), B))))
        # print(np.dot(Tr_view_inv, A) + B)
        # print(np.dot(Tr_view_inv, np.append(fiducial_point - size_3d / 2 * spacing_3d, 1))[:3] + size_3d / 2 * spacing_3d - source)
        # print(np.dot(Tr_view_inv, np.append(fiducial_point - size_3d / 2 * spacing_3d + np.dot(np.linalg.inv(Tr_view_inv), np.append(size_3d / 2 * spacing_3d - source, 1))[:3], 1)))
        # X_to_save = A + np.dot(Tr_view_inv[:3, :3].T, B)
        X_to_save = fiducial_point - size_3d / 2 * spacing_3d
        print(center - source)
        
        K = np.array([[distance / spacing_2d[0], 0, size_2d[0] / 2],
                        [0, distance / spacing_2d[1], size_2d[1] / 2],
                        [0, 0, 1]])
        # point_2d = 1 / X_source[2] * np.dot(K, X_source.T)[:2]
        print(np.dot(Tr_view_inv, np.append(X_to_save, 1)))
        point_2d = 1 / X_source[2] * np.dot(K, X_source.T)[:2]
        point_2d_list.append(point_2d)
        point_3d_list.append(X_to_save)
        print("point_2d_new:", point_2d)
    point_2d_list = np.asarray(point_2d_list)
    point_3d_list = np.asarray(point_3d_list)
    # point_3d_list[0:4, 0] = np.random.randn() * 10
    
    dist = np.zeros((5, 1))
    found, r, t, _ = cv2.solvePnPRansac(point_3d_list, point_2d_list, K, dist) #计算雷达相机外参,r-旋转向量，t-平移向量
    # found, r, t = cv2.solvePnP(point_3d_list[9:13, :], point_2d_list[9:13, :], K, dist) #计算雷达相机外参,r-旋转向量，t-平移向量
    solve2P(point_3d_list[4:6, :], point_2d_list[4:6, :], K, dist, Tr_view_inv)
    R = cv2.Rodrigues(r)[0] #旋转向量转旋转矩阵
    camera_position = -np.dot(R.T, t).T #相机位置
    print(np.dot(transformation, np.append(source - center, 1).T)[:3] + center)
    print(R, t)
    print("camera_position:", camera_position)
    d2, _ = cv2.projectPoints(np.array([point_3d_list[0]]), r, t, K, dist)#重投影验证
    print(d2, point_2d_list[0])
    # translation_error = np.linalg.norm(camera_position - (np.dot(transformation, np.append(source - center, 1).T)[:3] + center))
    translation_error = np.linalg.norm(Tr_view_inv[:3, 3] - t.T)
    rotation_error = np.linalg.norm(Tr_view_inv[:3, :3] - R)
    print("errors:", translation_error, rotation_error)
    global max_error
    if rotation_error > max_error:
        max_error = rotation_error
    print("max rot error:", max_error)

    # Write files
    # f = open("/home/leko/SCN-pytorch/medicalLabels.txt", 'w')
    # f.write("images/1.png ")
    # q = dcm2quat(transformation[:3, :3])
    # t = transformation[:3, 3]
    # for item in q.tolist():
    #     f.write("%s " % item)
    # for item in t.tolist():
    #     f.write("%s " % item)
    # f.write("\n")
    # # f.writelines(t.tolist()) + str(q.tolist()) + str([21, 4, 5]))
    # for k in range(point_2d_list.shape[0]):
    #     point_2d = point_2d_list[k]
    #     point_3d = point_3d_list[k]
    #     if point_2d[0] > size_2d[0] or point_2d[1] > size_2d[1] or \
    #         point_2d[0] < 0 or point_2d[1] < 0:
    #         continue
    #     for item in point_2d.tolist():
    #         f.write("%s " % item)
    #     for item in point_3d.tolist():
    #         f.write("%s " % item)
    # f.close()

    return point_2d_list, label_list



# root_path = "/media/leko/Elements SE/VerSe2019_dataset/training_data"

# for i in range(250):
#     ct_name = i
#     ct_file_path = os.path.join(root_path, "verse%(number)03d.nii.gz"%{'number': ct_name})
#     fiducial_file_path = os.path.join(root_path, "verse%(number)03d_ctd.json"%{'number': ct_name})
#     if not os.path.exists(ct_file_path):
#         continue

# real_xray_flag = True
real_xray_flag = False
# clean_flag = True
clean_flag = False
augment_flag = True
# augment_flag = False
# root_path = "/home/leko/MedicalDataAugmentationTool-master/data/CT-hospital"
# root_path = "/home/leko/MedicalDataAugmentationTool-master/data/LIDC_data"
root_path = "/media/leko/Elements SE/LIDC-IDRI-CT"
accept_list = ['LIDC-IDRI-0228.nii.gz.png', 'LIDC-IDRI-0215.nii.gz.png', 'LIDC-IDRI-0668.nii.gz.png', 'LIDC-IDRI-0725.nii.gz.png', 'LIDC-IDRI-0771.nii.gz.png', 'LIDC-IDRI-0770.nii.gz.png', 'LIDC-IDRI-0869.nii.gz.png', 'LIDC-IDRI-0460.nii.gz.png', 'LIDC-IDRI-0754.nii.gz.png', 'LIDC-IDRI-0308.nii.gz.png', 'LIDC-IDRI-0835.nii.gz.png', 'LIDC-IDRI-0461.nii.gz.png', 'LIDC-IDRI-0004.nii.gz.png', 'LIDC-IDRI-0677.nii.gz.png', 'LIDC-IDRI-0118.nii.gz.png', 'LIDC-IDRI-0130.nii.gz.png', 'LIDC-IDRI-0122.nii.gz.png', 'LIDC-IDRI-0603.nii.gz.png', 'LIDC-IDRI-0655.nii.gz.png', 'LIDC-IDRI-0961.nii.gz.png', 'LIDC-IDRI-0792.nii.gz.png', 'LIDC-IDRI-0286.nii.gz.png', 'LIDC-IDRI-0875.nii.gz.png', 'LIDC-IDRI-0772.nii.gz.png', 'LIDC-IDRI-0244.nii.gz.png', 'LIDC-IDRI-0636.nii.gz.png', 'LIDC-IDRI-0803.nii.gz.png', 'LIDC-IDRI-0827.nii.gz.png', 'LIDC-IDRI-0941.nii.gz.png', 'LIDC-IDRI-0462.nii.gz.png', 'LIDC-IDRI-0522.nii.gz.png', 'LIDC-IDRI-0732.nii.gz.png', 'LIDC-IDRI-0325.nii.gz.png', 'LIDC-IDRI-0688.nii.gz.png', 'LIDC-IDRI-0451.nii.gz.png', 'LIDC-IDRI-0649.nii.gz.png', 'LIDC-IDRI-0547.nii.gz.png', 'LIDC-IDRI-0570.nii.gz.png', 'LIDC-IDRI-0564.nii.gz.png', 'LIDC-IDRI-0186.nii.gz.png', 'LIDC-IDRI-0392.nii.gz.png', 'LIDC-IDRI-0235.nii.gz.png', 'LIDC-IDRI-0100.nii.gz.png', 'LIDC-IDRI-0512.nii.gz.png', 'LIDC-IDRI-0824.nii.gz.png', 'LIDC-IDRI-0370.nii.gz.png', 'LIDC-IDRI-0644.nii.gz.png', 'LIDC-IDRI-0674.nii.gz.png', 'LIDC-IDRI-0947.nii.gz.png', 'LIDC-IDRI-0112.nii.gz.png', 'LIDC-IDRI-0887.nii.gz.png', 'LIDC-IDRI-0573.nii.gz.png', 'LIDC-IDRI-0620.nii.gz.png', 'LIDC-IDRI-0784.nii.gz.png', 'LIDC-IDRI-0607.nii.gz.png', 'LIDC-IDRI-0595.nii.gz.png', 'LIDC-IDRI-0287.nii.gz.png', 'LIDC-IDRI-0861.nii.gz.png', 'LIDC-IDRI-0499.nii.gz.png', 'LIDC-IDRI-0213.nii.gz.png', 'LIDC-IDRI-0327.nii.gz.png', 'LIDC-IDRI-0847.nii.gz.png', 'LIDC-IDRI-0046.nii.gz.png', 'LIDC-IDRI-0809.nii.gz.png', 'LIDC-IDRI-0889.nii.gz.png', 'LIDC-IDRI-0964.nii.gz.png', 'LIDC-IDRI-0464.nii.gz.png', 'LIDC-IDRI-0908.nii.gz.png', 'LIDC-IDRI-0852.nii.gz.png', 'LIDC-IDRI-0152.nii.gz.png', 'LIDC-IDRI-0505.nii.gz.png', 'LIDC-IDRI-0686.nii.gz.png', 'LIDC-IDRI-0113.nii.gz.png', 'LIDC-IDRI-0713.nii.gz.png', 'LIDC-IDRI-0111.nii.gz.png', 'LIDC-IDRI-0795.nii.gz.png', 'LIDC-IDRI-0473.nii.gz.png', 'LIDC-IDRI-0456.nii.gz.png', 'LIDC-IDRI-0358.nii.gz.png', 'LIDC-IDRI-1010.nii.gz.png', 'LIDC-IDRI-0063.nii.gz.png', 'LIDC-IDRI-0383.nii.gz.png', 'LIDC-IDRI-0055.nii.gz.png', 'LIDC-IDRI-0247.nii.gz.png', 'LIDC-IDRI-0756.nii.gz.png', 'LIDC-IDRI-0026.nii.gz.png', 'LIDC-IDRI-1001.nii.gz.png', 'LIDC-IDRI-0679.nii.gz.png', 'LIDC-IDRI-0973.nii.gz.png', 'LIDC-IDRI-0315.nii.gz.png', 'LIDC-IDRI-0997.nii.gz.png', 'LIDC-IDRI-0384.nii.gz.png', 'LIDC-IDRI-0860.nii.gz.png', 'LIDC-IDRI-0594.nii.gz.png', 'LIDC-IDRI-0769.nii.gz.png', 'LIDC-IDRI-0657.nii.gz.png', 'LIDC-IDRI-0068.nii.gz.png', 'LIDC-IDRI-0364.nii.gz.png', 'LIDC-IDRI-0094.nii.gz.png', 'LIDC-IDRI-0903.nii.gz.png', 'LIDC-IDRI-0762.nii.gz.png', 'LIDC-IDRI-0755.nii.gz.png', 'LIDC-IDRI-0609.nii.gz.png', 'LIDC-IDRI-0781.nii.gz.png', 'LIDC-IDRI-0210.nii.gz.png', 'LIDC-IDRI-0386.nii.gz.png', 'LIDC-IDRI-0399.nii.gz.png', 'LIDC-IDRI-0036.nii.gz.png', 'LIDC-IDRI-0540.nii.gz.png', 'LIDC-IDRI-0375.nii.gz.png', 'LIDC-IDRI-0434.nii.gz.png', 'LIDC-IDRI-0436.nii.gz.png', 'LIDC-IDRI-0430.nii.gz.png', 'LIDC-IDRI-0391.nii.gz.png', 'LIDC-IDRI-0410.nii.gz.png', 'LIDC-IDRI-0067.nii.gz.png', 'LIDC-IDRI-0164.nii.gz.png', 'LIDC-IDRI-0628.nii.gz.png', 'LIDC-IDRI-0108.nii.gz.png', 'LIDC-IDRI-0233.nii.gz.png', 'LIDC-IDRI-0896.nii.gz.png', 'LIDC-IDRI-0302.nii.gz.png', 'LIDC-IDRI-0555.nii.gz.png', 'LIDC-IDRI-0693.nii.gz.png', 'LIDC-IDRI-0192.nii.gz.png', 'LIDC-IDRI-0251.nii.gz.png', 'LIDC-IDRI-0196.nii.gz.png', 'LIDC-IDRI-0821.nii.gz.png', 'LIDC-IDRI-0613.nii.gz.png', 'LIDC-IDRI-0099.nii.gz.png', 'LIDC-IDRI-0019.nii.gz.png', 'LIDC-IDRI-0316.nii.gz.png', 'LIDC-IDRI-0894.nii.gz.png', 'LIDC-IDRI-0935.nii.gz.png', 'LIDC-IDRI-0363.nii.gz.png', 'LIDC-IDRI-0501.nii.gz.png', 'LIDC-IDRI-0484.nii.gz.png', 'LIDC-IDRI-0710.nii.gz.png', 'LIDC-IDRI-0873.nii.gz.png', 'LIDC-IDRI-0357.nii.gz.png', 'LIDC-IDRI-0429.nii.gz.png', 'LIDC-IDRI-0243.nii.gz.png', 'LIDC-IDRI-0425.nii.gz.png', 'LIDC-IDRI-0728.nii.gz.png', 'LIDC-IDRI-0970.nii.gz.png', 'LIDC-IDRI-0625.nii.gz.png', 'LIDC-IDRI-0178.nii.gz.png', 'LIDC-IDRI-0535.nii.gz.png', 'LIDC-IDRI-0508.nii.gz.png', 'LIDC-IDRI-0582.nii.gz.png', 'LIDC-IDRI-0053.nii.gz.png', 'LIDC-IDRI-0284.nii.gz.png', 'LIDC-IDRI-0909.nii.gz.png', 'LIDC-IDRI-0822.nii.gz.png', 'LIDC-IDRI-0841.nii.gz.png', 'LIDC-IDRI-0936.nii.gz.png', 'LIDC-IDRI-0497.nii.gz.png', 'LIDC-IDRI-0219.nii.gz.png', 'LIDC-IDRI-0565.nii.gz.png', 'LIDC-IDRI-0948.nii.gz.png', 'LIDC-IDRI-0734.nii.gz.png', 'LIDC-IDRI-0470.nii.gz.png', 'LIDC-IDRI-0255.nii.gz.png', 'LIDC-IDRI-0736.nii.gz.png', 'LIDC-IDRI-0454.nii.gz.png', 'LIDC-IDRI-0236.nii.gz.png', 'LIDC-IDRI-0851.nii.gz.png', 'LIDC-IDRI-0802.nii.gz.png', 'LIDC-IDRI-0617.nii.gz.png', 'LIDC-IDRI-0593.nii.gz.png', 'LIDC-IDRI-0925.nii.gz.png', 'LIDC-IDRI-0663.nii.gz.png', 'LIDC-IDRI-0977.nii.gz.png', 'LIDC-IDRI-0622.nii.gz.png', 'LIDC-IDRI-0799.nii.gz.png', 'LIDC-IDRI-0918.nii.gz.png', 'LIDC-IDRI-0533.nii.gz.png', 'LIDC-IDRI-0507.nii.gz.png', 'LIDC-IDRI-0996.nii.gz.png', 'LIDC-IDRI-0714.nii.gz.png', 'LIDC-IDRI-0495.nii.gz.png', 'LIDC-IDRI-0066.nii.gz.png', 'LIDC-IDRI-0338.nii.gz.png', 'LIDC-IDRI-0011.nii.gz.png', 'LIDC-IDRI-0952.nii.gz.png', 'LIDC-IDRI-0270.nii.gz.png', 'LIDC-IDRI-0864.nii.gz.png', 'LIDC-IDRI-0696.nii.gz.png', 'LIDC-IDRI-0344.nii.gz.png', 'LIDC-IDRI-0331.nii.gz.png', 'LIDC-IDRI-0722.nii.gz.png', 'LIDC-IDRI-0483.nii.gz.png', 'LIDC-IDRI-0209.nii.gz.png', 'LIDC-IDRI-0568.nii.gz.png', 'LIDC-IDRI-0129.nii.gz.png', 'LIDC-IDRI-0839.nii.gz.png', 'LIDC-IDRI-0837.nii.gz.png', 'LIDC-IDRI-0265.nii.gz.png', 'LIDC-IDRI-0924.nii.gz.png', 'LIDC-IDRI-0191.nii.gz.png', 'LIDC-IDRI-0017.nii.gz.png', 'LIDC-IDRI-0401.nii.gz.png', 'LIDC-IDRI-0954.nii.gz.png', 'LIDC-IDRI-0933.nii.gz.png', 'LIDC-IDRI-0022.nii.gz.png', 'LIDC-IDRI-0960.nii.gz.png', 'LIDC-IDRI-0346.nii.gz.png', 'LIDC-IDRI-0658.nii.gz.png', 'LIDC-IDRI-0955.nii.gz.png', 'LIDC-IDRI-0980.nii.gz.png', 'LIDC-IDRI-0712.nii.gz.png', 'LIDC-IDRI-0095.nii.gz.png', 'LIDC-IDRI-0939.nii.gz.png', 'LIDC-IDRI-0672.nii.gz.png', 'LIDC-IDRI-0652.nii.gz.png', 'LIDC-IDRI-0040.nii.gz.png', 'LIDC-IDRI-0858.nii.gz.png', 'LIDC-IDRI-0984.nii.gz.png', 'LIDC-IDRI-0849.nii.gz.png', 'LIDC-IDRI-0804.nii.gz.png', 'LIDC-IDRI-0746.nii.gz.png', 'LIDC-IDRI-0915.nii.gz.png', 'LIDC-IDRI-0976.nii.gz.png', 'LIDC-IDRI-0382.nii.gz.png', 'LIDC-IDRI-1008.nii.gz.png', 'LIDC-IDRI-0892.nii.gz.png', 'LIDC-IDRI-0981.nii.gz.png', 'LIDC-IDRI-0983.nii.gz.png', 'LIDC-IDRI-0538.nii.gz.png', 'LIDC-IDRI-0676.nii.gz.png', 'LIDC-IDRI-0880.nii.gz.png', 'LIDC-IDRI-0494.nii.gz.png', 'LIDC-IDRI-0431.nii.gz.png', 'LIDC-IDRI-0204.nii.gz.png', 'LIDC-IDRI-0664.nii.gz.png', 'LIDC-IDRI-0424.nii.gz.png', 'LIDC-IDRI-0024.nii.gz.png', 'LIDC-IDRI-0557.nii.gz.png', 'LIDC-IDRI-0923.nii.gz.png', 'LIDC-IDRI-0110.nii.gz.png', 'LIDC-IDRI-0426.nii.gz.png', 'LIDC-IDRI-1012.nii.gz.png', 'LIDC-IDRI-0597.nii.gz.png', 'LIDC-IDRI-0878.nii.gz.png', 'LIDC-IDRI-0480.nii.gz.png', 'LIDC-IDRI-0433.nii.gz.png', 'LIDC-IDRI-0806.nii.gz.png', 'LIDC-IDRI-0398.nii.gz.png', 'LIDC-IDRI-0385.nii.gz.png', 'LIDC-IDRI-0633.nii.gz.png', 'LIDC-IDRI-0142.nii.gz.png', 'LIDC-IDRI-0134.nii.gz.png', 'LIDC-IDRI-1003.nii.gz.png', 'LIDC-IDRI-0306.nii.gz.png', 'LIDC-IDRI-0602.nii.gz.png', 'LIDC-IDRI-0443.nii.gz.png', 'LIDC-IDRI-0708.nii.gz.png', 'LIDC-IDRI-0368.nii.gz.png', 'LIDC-IDRI-0940.nii.gz.png', 'LIDC-IDRI-0141.nii.gz.png', 'LIDC-IDRI-0056.nii.gz.png', 'LIDC-IDRI-0455.nii.gz.png', 'LIDC-IDRI-0445.nii.gz.png', 'LIDC-IDRI-0830.nii.gz.png', 'LIDC-IDRI-0988.nii.gz.png', 'LIDC-IDRI-0496.nii.gz.png', 'LIDC-IDRI-0413.nii.gz.png', 'LIDC-IDRI-0684.nii.gz.png', 'LIDC-IDRI-0810.nii.gz.png', 'LIDC-IDRI-0267.nii.gz.png', 'LIDC-IDRI-0971.nii.gz.png', 'LIDC-IDRI-0416.nii.gz.png', 'LIDC-IDRI-0678.nii.gz.png', 'LIDC-IDRI-0202.nii.gz.png', 'LIDC-IDRI-0288.nii.gz.png', 'LIDC-IDRI-0380.nii.gz.png', 'LIDC-IDRI-0482.nii.gz.png', 'LIDC-IDRI-0154.nii.gz.png', 'LIDC-IDRI-0349.nii.gz.png', 'LIDC-IDRI-0605.nii.gz.png', 'LIDC-IDRI-0414.nii.gz.png', 'LIDC-IDRI-0589.nii.gz.png', 'LIDC-IDRI-0982.nii.gz.png', 'LIDC-IDRI-0175.nii.gz.png', 'LIDC-IDRI-0558.nii.gz.png', 'LIDC-IDRI-0028.nii.gz.png', 'LIDC-IDRI-0485.nii.gz.png', 'LIDC-IDRI-0828.nii.gz.png', 'LIDC-IDRI-0610.nii.gz.png', 'LIDC-IDRI-0726.nii.gz.png', 'LIDC-IDRI-0591.nii.gz.png', 'LIDC-IDRI-0638.nii.gz.png', 'LIDC-IDRI-0637.nii.gz.png', 'LIDC-IDRI-0442.nii.gz.png', 'LIDC-IDRI-0575.nii.gz.png', 'LIDC-IDRI-0900.nii.gz.png', 'LIDC-IDRI-0517.nii.gz.png', 'LIDC-IDRI-0546.nii.gz.png', 'LIDC-IDRI-0768.nii.gz.png', 'LIDC-IDRI-0962.nii.gz.png', 'LIDC-IDRI-0774.nii.gz.png', 'LIDC-IDRI-0348.nii.gz.png', 'LIDC-IDRI-0775.nii.gz.png', 'LIDC-IDRI-0966.nii.gz.png', 'LIDC-IDRI-0407.nii.gz.png', 'LIDC-IDRI-1004.nii.gz.png', 'LIDC-IDRI-0826.nii.gz.png', 'LIDC-IDRI-0729.nii.gz.png', 'LIDC-IDRI-0062.nii.gz.png', 'LIDC-IDRI-0834.nii.gz.png', 'LIDC-IDRI-0542.nii.gz.png', 'LIDC-IDRI-0524.nii.gz.png', 'LIDC-IDRI-0931.nii.gz.png', 'LIDC-IDRI-0631.nii.gz.png', 'LIDC-IDRI-0250.nii.gz.png', 'LIDC-IDRI-0226.nii.gz.png', 'LIDC-IDRI-0608.nii.gz.png', 'LIDC-IDRI-0788.nii.gz.png', 'LIDC-IDRI-0117.nii.gz.png', 'LIDC-IDRI-0895.nii.gz.png', 'LIDC-IDRI-0879.nii.gz.png', 'LIDC-IDRI-0275.nii.gz.png', 'LIDC-IDRI-0127.nii.gz.png', 'LIDC-IDRI-0548.nii.gz.png', 'LIDC-IDRI-0634.nii.gz.png', 'LIDC-IDRI-0563.nii.gz.png', 'LIDC-IDRI-0437.nii.gz.png', 'LIDC-IDRI-0516.nii.gz.png', 'LIDC-IDRI-0711.nii.gz.png', 'LIDC-IDRI-0283.nii.gz.png', 'LIDC-IDRI-0271.nii.gz.png', 'LIDC-IDRI-0452.nii.gz.png', 'LIDC-IDRI-0707.nii.gz.png', 'LIDC-IDRI-0862.nii.gz.png', 'LIDC-IDRI-0640.nii.gz.png', 'LIDC-IDRI-0642.nii.gz.png', 'LIDC-IDRI-0193.nii.gz.png', 'LIDC-IDRI-0526.nii.gz.png', 'LIDC-IDRI-0614.nii.gz.png', 'LIDC-IDRI-0241.nii.gz.png', 'LIDC-IDRI-0188.nii.gz.png', 'LIDC-IDRI-0630.nii.gz.png', 'LIDC-IDRI-0543.nii.gz.png', 'LIDC-IDRI-0551.nii.gz.png', 'LIDC-IDRI-0833.nii.gz.png', 'LIDC-IDRI-0648.nii.gz.png', 'LIDC-IDRI-0586.nii.gz.png', 'LIDC-IDRI-0510.nii.gz.png', 'LIDC-IDRI-0300.nii.gz.png', 'LIDC-IDRI-0187.nii.gz.png', 'LIDC-IDRI-0616.nii.gz.png', 'LIDC-IDRI-0985.nii.gz.png', 'LIDC-IDRI-0377.nii.gz.png', 'LIDC-IDRI-0378.nii.gz.png', 'LIDC-IDRI-0259.nii.gz.png', 'LIDC-IDRI-0047.nii.gz.png', 'LIDC-IDRI-0820.nii.gz.png', 'LIDC-IDRI-0906.nii.gz.png', 'LIDC-IDRI-0335.nii.gz.png', 'LIDC-IDRI-0343.nii.gz.png', 'LIDC-IDRI-0397.nii.gz.png', 'LIDC-IDRI-0913.nii.gz.png', 'LIDC-IDRI-0280.nii.gz.png', 'LIDC-IDRI-0293.nii.gz.png', 'LIDC-IDRI-0926.nii.gz.png', 'LIDC-IDRI-0656.nii.gz.png', 'LIDC-IDRI-0929.nii.gz.png', 'LIDC-IDRI-0856.nii.gz.png', 'LIDC-IDRI-0963.nii.gz.png', 'LIDC-IDRI-0744.nii.gz.png', 'LIDC-IDRI-0662.nii.gz.png', 'LIDC-IDRI-0871.nii.gz.png', 'LIDC-IDRI-0580.nii.gz.png', 'LIDC-IDRI-0808.nii.gz.png', 'LIDC-IDRI-0372.nii.gz.png', 'LIDC-IDRI-0071.nii.gz.png', 'LIDC-IDRI-0190.nii.gz.png', 'LIDC-IDRI-0332.nii.gz.png', 'LIDC-IDRI-0023.nii.gz.png', 'LIDC-IDRI-0987.nii.gz.png', 'LIDC-IDRI-0748.nii.gz.png', 'LIDC-IDRI-0133.nii.gz.png', 'LIDC-IDRI-0222.nii.gz.png', 'LIDC-IDRI-0786.nii.gz.png', 'LIDC-IDRI-0942.nii.gz.png', 'LIDC-IDRI-0560.nii.gz.png', 'LIDC-IDRI-0371.nii.gz.png', 'LIDC-IDRI-0212.nii.gz.png', 'LIDC-IDRI-0881.nii.gz.png', 'LIDC-IDRI-0419.nii.gz.png', 'LIDC-IDRI-0393.nii.gz.png', 'LIDC-IDRI-0006.nii.gz.png', 'LIDC-IDRI-0388.nii.gz.png', 'LIDC-IDRI-0968.nii.gz.png', 'LIDC-IDRI-0521.nii.gz.png', 'LIDC-IDRI-0876.nii.gz.png', 'LIDC-IDRI-0797.nii.gz.png', 'LIDC-IDRI-0730.nii.gz.png', 'LIDC-IDRI-0449.nii.gz.png', 'LIDC-IDRI-0119.nii.gz.png', 'LIDC-IDRI-0276.nii.gz.png', 'LIDC-IDRI-0532.nii.gz.png', 'LIDC-IDRI-0465.nii.gz.png', 'LIDC-IDRI-0278.nii.gz.png', 'LIDC-IDRI-0740.nii.gz.png', 'LIDC-IDRI-0224.nii.gz.png', 'LIDC-IDRI-0661.nii.gz.png', 'LIDC-IDRI-0411.nii.gz.png', 'LIDC-IDRI-0897.nii.gz.png', 'LIDC-IDRI-0916.nii.gz.png', 'LIDC-IDRI-0717.nii.gz.png', 'LIDC-IDRI-0853.nii.gz.png', 'LIDC-IDRI-0928.nii.gz.png', 'LIDC-IDRI-0468.nii.gz.png', 'LIDC-IDRI-0767.nii.gz.png', 'LIDC-IDRI-0831.nii.gz.png', 'LIDC-IDRI-0362.nii.gz.png', 'LIDC-IDRI-0534.nii.gz.png', 'LIDC-IDRI-0556.nii.gz.png', 'LIDC-IDRI-0471.nii.gz.png', 'LIDC-IDRI-0432.nii.gz.png', 'LIDC-IDRI-0477.nii.gz.png', 'LIDC-IDRI-0503.nii.gz.png', 'LIDC-IDRI-0779.nii.gz.png', 'LIDC-IDRI-0498.nii.gz.png', 'LIDC-IDRI-0967.nii.gz.png', 'LIDC-IDRI-0838.nii.gz.png', 'LIDC-IDRI-0373.nii.gz.png', 'LIDC-IDRI-0624.nii.gz.png', 'LIDC-IDRI-0739.nii.gz.png', 'LIDC-IDRI-0520.nii.gz.png', 'LIDC-IDRI-0298.nii.gz.png', 'LIDC-IDRI-0635.nii.gz.png', 'LIDC-IDRI-0221.nii.gz.png', 'LIDC-IDRI-0150.nii.gz.png', 'LIDC-IDRI-0057.nii.gz.png', 'LIDC-IDRI-0237.nii.gz.png', 'LIDC-IDRI-0519.nii.gz.png', 'LIDC-IDRI-0408.nii.gz.png', 'LIDC-IDRI-0675.nii.gz.png', 'LIDC-IDRI-0765.nii.gz.png', 'LIDC-IDRI-0336.nii.gz.png', 'LIDC-IDRI-0307.nii.gz.png', 'LIDC-IDRI-0752.nii.gz.png', 'LIDC-IDRI-0704.nii.gz.png', 'LIDC-IDRI-0930.nii.gz.png', 'LIDC-IDRI-0157.nii.gz.png', 'LIDC-IDRI-0123.nii.gz.png', 'LIDC-IDRI-0665.nii.gz.png', 'LIDC-IDRI-0731.nii.gz.png', 'LIDC-IDRI-0395.nii.gz.png', 'LIDC-IDRI-0381.nii.gz.png', 'LIDC-IDRI-0020.nii.gz.png', 'LIDC-IDRI-0330.nii.gz.png', 'LIDC-IDRI-0511.nii.gz.png', 'LIDC-IDRI-0444.nii.gz.png', 'LIDC-IDRI-0500.nii.gz.png', 'LIDC-IDRI-0813.nii.gz.png', 'LIDC-IDRI-0891.nii.gz.png', 'LIDC-IDRI-0581.nii.gz.png', 'LIDC-IDRI-0596.nii.gz.png', 'LIDC-IDRI-0143.nii.gz.png', 'LIDC-IDRI-0479.nii.gz.png', 'LIDC-IDRI-0737.nii.gz.png', 'LIDC-IDRI-0701.nii.gz.png', 'LIDC-IDRI-0409.nii.gz.png', 'LIDC-IDRI-0529.nii.gz.png', 'LIDC-IDRI-0750.nii.gz.png', 'LIDC-IDRI-0789.nii.gz.png', 'LIDC-IDRI-0703.nii.gz.png', 'LIDC-IDRI-0994.nii.gz.png', 'LIDC-IDRI-0953.nii.gz.png', 'LIDC-IDRI-0623.nii.gz.png', 'LIDC-IDRI-0427.nii.gz.png', 'LIDC-IDRI-0716.nii.gz.png', 'LIDC-IDRI-0702.nii.gz.png', 'LIDC-IDRI-0304.nii.gz.png', 'LIDC-IDRI-0537.nii.gz.png', 'LIDC-IDRI-0945.nii.gz.png', 'LIDC-IDRI-0034.nii.gz.png', 'LIDC-IDRI-0995.nii.gz.png', 'LIDC-IDRI-0885.nii.gz.png', 'LIDC-IDRI-0934.nii.gz.png', 'LIDC-IDRI-0314.nii.gz.png', 'LIDC-IDRI-0753.nii.gz.png', 'LIDC-IDRI-0297.nii.gz.png', 'LIDC-IDRI-0699.nii.gz.png', 'LIDC-IDRI-0115.nii.gz.png', 'LIDC-IDRI-0463.nii.gz.png', 'LIDC-IDRI-0387.nii.gz.png', 'LIDC-IDRI-0805.nii.gz.png', 'LIDC-IDRI-0733.nii.gz.png', 'LIDC-IDRI-0239.nii.gz.png', 'LIDC-IDRI-0815.nii.gz.png', 'LIDC-IDRI-0606.nii.gz.png', 'LIDC-IDRI-0347.nii.gz.png', 'LIDC-IDRI-0645.nii.gz.png', 'LIDC-IDRI-0569.nii.gz.png', 'LIDC-IDRI-0491.nii.gz.png', 'LIDC-IDRI-0882.nii.gz.png', 'LIDC-IDRI-0747.nii.gz.png', 'LIDC-IDRI-0208.nii.gz.png', 'LIDC-IDRI-0167.nii.gz.png', 'LIDC-IDRI-0884.nii.gz.png', 'LIDC-IDRI-0269.nii.gz.png', 'LIDC-IDRI-0072.nii.gz.png', 'LIDC-IDRI-0641.nii.gz.png', 'LIDC-IDRI-0867.nii.gz.png', 'LIDC-IDRI-0029.nii.gz.png', 'LIDC-IDRI-0329.nii.gz.png', 'LIDC-IDRI-0128.nii.gz.png', 'LIDC-IDRI-0474.nii.gz.png', 'LIDC-IDRI-0048.nii.gz.png', 'LIDC-IDRI-0305.nii.gz.png', 'LIDC-IDRI-0902.nii.gz.png', 'LIDC-IDRI-0874.nii.gz.png', 'LIDC-IDRI-0310.nii.gz.png', 'LIDC-IDRI-0576.nii.gz.png', 'LIDC-IDRI-0604.nii.gz.png', 'LIDC-IDRI-0248.nii.gz.png', 'LIDC-IDRI-0181.nii.gz.png', 'LIDC-IDRI-0322.nii.gz.png', 'LIDC-IDRI-0438.nii.gz.png', 'LIDC-IDRI-0292.nii.gz.png', 'LIDC-IDRI-0670.nii.gz.png', 'LIDC-IDRI-0844.nii.gz.png', 'LIDC-IDRI-0911.nii.gz.png', 'LIDC-IDRI-0097.nii.gz.png', 'LIDC-IDRI-0912.nii.gz.png', 'LIDC-IDRI-0914.nii.gz.png', 'LIDC-IDRI-0156.nii.gz.png', 'LIDC-IDRI-0778.nii.gz.png', 'LIDC-IDRI-0950.nii.gz.png', 'LIDC-IDRI-0727.nii.gz.png', 'LIDC-IDRI-0946.nii.gz.png', 'LIDC-IDRI-0309.nii.gz.png', 'LIDC-IDRI-0692.nii.gz.png', 'LIDC-IDRI-0319.nii.gz.png', 'LIDC-IDRI-0513.nii.gz.png', 'LIDC-IDRI-0683.nii.gz.png', 'LIDC-IDRI-0525.nii.gz.png', 'LIDC-IDRI-0058.nii.gz.png', 'LIDC-IDRI-0690.nii.gz.png', 'LIDC-IDRI-0673.nii.gz.png', 'LIDC-IDRI-0105.nii.gz.png', 'LIDC-IDRI-0320.nii.gz.png', 'LIDC-IDRI-0328.nii.gz.png', 'LIDC-IDRI-0333.nii.gz.png', 'LIDC-IDRI-0459.nii.gz.png', 'LIDC-IDRI-0031.nii.gz.png', 'LIDC-IDRI-0400.nii.gz.png', 'LIDC-IDRI-0506.nii.gz.png', 'LIDC-IDRI-0374.nii.gz.png', 'LIDC-IDRI-0951.nii.gz.png', 'LIDC-IDRI-0766.nii.gz.png', 'LIDC-IDRI-0845.nii.gz.png', 'LIDC-IDRI-0162.nii.gz.png', 'LIDC-IDRI-0008.nii.gz.png', 'LIDC-IDRI-0163.nii.gz.png', 'LIDC-IDRI-0700.nii.gz.png', 'LIDC-IDRI-0440.nii.gz.png', 'LIDC-IDRI-0467.nii.gz.png', 'LIDC-IDRI-0138.nii.gz.png', 'LIDC-IDRI-0447.nii.gz.png', 'LIDC-IDRI-0859.nii.gz.png', 'LIDC-IDRI-0185.nii.gz.png', 'LIDC-IDRI-0705.nii.gz.png', 'LIDC-IDRI-0965.nii.gz.png', 'LIDC-IDRI-0531.nii.gz.png', 'LIDC-IDRI-0488.nii.gz.png', 'LIDC-IDRI-0584.nii.gz.png', 'LIDC-IDRI-1006.nii.gz.png', 'LIDC-IDRI-0014.nii.gz.png', 'LIDC-IDRI-0854.nii.gz.png', 'LIDC-IDRI-0863.nii.gz.png', 'LIDC-IDRI-0843.nii.gz.png', 'LIDC-IDRI-0229.nii.gz.png', 'LIDC-IDRI-0559.nii.gz.png', 'LIDC-IDRI-0137.nii.gz.png', 'LIDC-IDRI-0969.nii.gz.png', 'LIDC-IDRI-0257.nii.gz.png', 'LIDC-IDRI-0776.nii.gz.png', 'LIDC-IDRI-0149.nii.gz.png', 'LIDC-IDRI-0667.nii.gz.png', 'LIDC-IDRI-0659.nii.gz.png', 'LIDC-IDRI-0877.nii.gz.png', 'LIDC-IDRI-0487.nii.gz.png', 'LIDC-IDRI-0352.nii.gz.png', 'LIDC-IDRI-0691.nii.gz.png', 'LIDC-IDRI-0883.nii.gz.png', 'LIDC-IDRI-0295.nii.gz.png', 'LIDC-IDRI-0009.nii.gz.png', 'LIDC-IDRI-0611.nii.gz.png', 'LIDC-IDRI-0545.nii.gz.png', 'LIDC-IDRI-0083.nii.gz.png', 'LIDC-IDRI-0627.nii.gz.png', 'LIDC-IDRI-0562.nii.gz.png', 'LIDC-IDRI-0646.nii.gz.png', 'LIDC-IDRI-0253.nii.gz.png', 'LIDC-IDRI-0356.nii.gz.png', 'LIDC-IDRI-0840.nii.gz.png', 'LIDC-IDRI-0342.nii.gz.png', 'LIDC-IDRI-0599.nii.gz.png', 'LIDC-IDRI-0169.nii.gz.png', 'LIDC-IDRI-0738.nii.gz.png', 'LIDC-IDRI-0660.nii.gz.png', 'LIDC-IDRI-0136.nii.gz.png', 'LIDC-IDRI-0719.nii.gz.png', 'LIDC-IDRI-0015.nii.gz.png', 'LIDC-IDRI-0417.nii.gz.png', 'LIDC-IDRI-0695.nii.gz.png', 'LIDC-IDRI-0735.nii.gz.png', 'LIDC-IDRI-0759.nii.gz.png', 'LIDC-IDRI-0721.nii.gz.png', 'LIDC-IDRI-0743.nii.gz.png', 'LIDC-IDRI-0457.nii.gz.png', 'LIDC-IDRI-0571.nii.gz.png', 'LIDC-IDRI-0998.nii.gz.png', 'LIDC-IDRI-0561.nii.gz.png', 'LIDC-IDRI-0991.nii.gz.png', 'LIDC-IDRI-0077.nii.gz.png', 'LIDC-IDRI-0092.nii.gz.png', 'LIDC-IDRI-0303.nii.gz.png', 'LIDC-IDRI-0530.nii.gz.png', 'LIDC-IDRI-0836.nii.gz.png', 'LIDC-IDRI-0103.nii.gz.png', 'LIDC-IDRI-0080.nii.gz.png', 'LIDC-IDRI-0745.nii.gz.png', 'LIDC-IDRI-0723.nii.gz.png', 'LIDC-IDRI-0260.nii.gz.png', 'LIDC-IDRI-0793.nii.gz.png', 'LIDC-IDRI-0579.nii.gz.png', 'LIDC-IDRI-0870.nii.gz.png', 'LIDC-IDRI-0553.nii.gz.png', 'LIDC-IDRI-0317.nii.gz.png', 'LIDC-IDRI-0907.nii.gz.png', 'LIDC-IDRI-0974.nii.gz.png', 'LIDC-IDRI-0340.nii.gz.png', 'LIDC-IDRI-0855.nii.gz.png', 'LIDC-IDRI-0741.nii.gz.png', 'LIDC-IDRI-0800.nii.gz.png', 'LIDC-IDRI-0817.nii.gz.png', 'LIDC-IDRI-0757.nii.gz.png', 'LIDC-IDRI-0198.nii.gz.png', 'LIDC-IDRI-0448.nii.gz.png', 'LIDC-IDRI-0261.nii.gz.png', 'LIDC-IDRI-0091.nii.gz.png', 'LIDC-IDRI-0629.nii.gz.png', 'LIDC-IDRI-0846.nii.gz.png', 'LIDC-IDRI-0045.nii.gz.png', 'LIDC-IDRI-0025.nii.gz.png', 'LIDC-IDRI-0787.nii.gz.png', 'LIDC-IDRI-0027.nii.gz.png', 'LIDC-IDRI-0720.nii.gz.png', 'LIDC-IDRI-0158.nii.gz.png', 'LIDC-IDRI-0650.nii.gz.png', 'LIDC-IDRI-0206.nii.gz.png', 'LIDC-IDRI-0312.nii.gz.png', 'LIDC-IDRI-0783.nii.gz.png', 'LIDC-IDRI-0225.nii.gz.png', 'LIDC-IDRI-0420.nii.gz.png', 'LIDC-IDRI-0486.nii.gz.png', 'LIDC-IDRI-0161.nii.gz.png', 'LIDC-IDRI-0842.nii.gz.png', 'LIDC-IDRI-0899.nii.gz.png', 'LIDC-IDRI-0816.nii.gz.png', 'LIDC-IDRI-0121.nii.gz.png', 'LIDC-IDRI-0975.nii.gz.png', 'LIDC-IDRI-0296.nii.gz.png', 'LIDC-IDRI-0687.nii.gz.png', 'LIDC-IDRI-0076.nii.gz.png', 'LIDC-IDRI-0632.nii.gz.png', 'LIDC-IDRI-0301.nii.gz.png']
ct_names = os.listdir(root_path)
random.seed(0)
np.random.seed(0)
for ct_name in ct_names:

    if ct_name + ".png" not in accept_list:
            continue

    # ct_name = 'CBCT2_REC_CT1.nii.gz'
    if os.path.isdir(os.path.join(root_path, ct_name)):
        continue
    ct_file_path = os.path.join(root_path, ct_name)
    ct_mask_file_path = os.path.join(root_path.rstrip('LIDC-IDRI-CT'), "LIDC-IDRI-lungmask", ct_name.split('.')[0] + "-mask.mha")
    fiducial_file_path = os.path.join(root_path, "results", ct_name.split('.')[0] + "_ctd.json")
    if not (os.path.exists(ct_file_path) and os.path.exists(ct_mask_file_path) and os.path.exists(fiducial_file_path)):
        continue
    # ct_name = 'LIDC-IDRI-0090.nii.gz'

    projector_info = {'threadsPerBlock_x': 16,
                        'threadsPerBlock_y': 16,
                        'threadsPerBlock_z': 1,
                        'DRRsize_x': 256,
                        'DRRsize_y': 256,
                        'focal_lenght': 1200,
                        'DRR_ppx': 0,  # Physical length(mm)
                        'DRR_ppy': 0,  # Physical length(mm)
                        'DRRspacing_x': 1.4921875,
                        'DRRspacing_y': 1.4921875
                        }

    input_ct_image = sitk.ReadImage(ct_file_path)
    input_ct_mask_image = sitk.ReadImage(ct_mask_file_path)
    input_ct_array = sitk.GetArrayFromImage(input_ct_image)
    input_ct_mask_array = sitk.GetArrayFromImage(input_ct_mask_image)
    # input_ct_array[input_ct_array < 100] = 0
    print(np.min(input_ct_array), np.max(input_ct_array))
    bone_ct_array = input_ct_array.copy()
    # bone_ct_array = bone_ct_array + 1024
    bone_ct_array[bone_ct_array < 100] = 0
    # bone_ct_array = bone_ct_array + 1024
    input_ct_array = input_ct_array + 1024
    input_ct_array[input_ct_array < 0] = 0
    input_ct_array = input_ct_array + 2 * bone_ct_array
    # input_ct_array = bone_ct_array
    # input_ct_array[input_ct_array > 2500] = 2500
    input_ct_array_ap = np.squeeze(np.mean(input_ct_array, axis=1))

    # new_input_ct_array = new_input_ct_array - 1024
    # new_bone_ct_array = new_input_ct_array.copy()
    # new_bone_ct_array[new_bone_ct_array < 100] = 0
    # new_input_ct_array = new_input_ct_array - 0.8 * new_bone_ct_array  # -0.5 energy experiment
    # # new_input_ct_array = new_input_ct_array
    # # new_input_ct_array = new_input_ct_array + 0.5 * new_bone_ct_array  # +0.5 energy experiment
    # # new_input_ct_array = new_input_ct_array + new_bone_ct_array
    # new_input_ct_array = new_input_ct_array + 1024

    # plt.imshow(input_ct_array_ap, cmap='gray')
    # plt.show()

    # 读取CT图像信息
    movSpacing = np.asarray(input_ct_image.GetSpacing())
    movSize = np.asarray(input_ct_image.GetSize())
    movOrigin = np.asarray(input_ct_image.GetOrigin())
    movCenter = np.asarray(movOrigin) + np.multiply(movSpacing, np.divide(movSize, 2.)) - np.divide(movSpacing, 2.)
    movPhySize = np.multiply(movSpacing, movSize)

    # 计算边界平面
    X0 = movCenter[0] - movSpacing[0] * movSize[0] * 0.5
    Y0 = movCenter[1] - movSpacing[1] * movSize[1] * 0.5
    Z0 = movCenter[2] - movSpacing[2] * movSize[2] * 0.5

    # 设置GPU参数
    NumThreadsPerBlock = np.array([projector_info['threadsPerBlock_x'], projector_info['threadsPerBlock_y'],
                            projector_info['threadsPerBlock_z']]).astype(np.int32)
    DRRsize_forGpu = np.array([projector_info['DRRsize_x'], projector_info['DRRsize_y'], 1]).astype(np.int32)
    MovSize_forGpu = np.array([movSize[0], movSize[1], movSize[2]]).astype(np.int32)
    MovSpacing_forGpu = np.array([movSpacing[0], movSpacing[1], movSpacing[2]]).astype(np.float32)
    MovCenter_forGpu = movCenter.astype(np.float32)
    movImgArray_1d = np.ravel(input_ct_array.copy(), order='C').astype(np.float32)

    # 定义光源位置
    source = np.zeros(3, dtype=np.float32)
    source[0] = movCenter[0]
    source[1] = movCenter[1]
    source[2] = movCenter[2] - projector_info['focal_lenght'] / 2.

    # 定义DRR参数
    DRRsize = [0] * 3
    DRRsize[0] = projector_info['DRRsize_x']
    DRRsize[1] = projector_info['DRRsize_y']
    DRRsize[2] = 1

    DRRspacing = [0] * 3
    DRRspacing[0] = projector_info['DRRspacing_x']
    DRRspacing[1] = projector_info['DRRspacing_y']
    DRRspacing[2] = 1

    # DRRspacing = [0] * 3
    # DRRspacing[0] = ds
    # DRRspacing[1] = ds
    # DRRspacing[2] = 1

    DRRorigin = [0] * 3
    DRRorigin[0] = movCenter[0] - projector_info['DRR_ppx'] - DRRspacing[0] * (DRRsize[0] - 1.) / 2.
    DRRorigin[1] = movCenter[1] - projector_info['DRR_ppy'] - DRRspacing[1] * (DRRsize[1] - 1.) / 2.
    DRRorigin[2] = movCenter[2] + projector_info['focal_lenght'] / 2.

    DRR = sitk.Image([DRRsize[0], DRRsize[1], 1], sitk.sitkFloat64)
    movDirection = input_ct_image.GetDirection()
    DRR.SetOrigin(DRRorigin)
    DRR.SetSpacing(DRRspacing)
    # DRR.SetDirection(movDirection)
    PhysicalPointImagefilter = sitk.PhysicalPointImageSource()
    PhysicalPointImagefilter.SetReferenceImage(DRR)
    sourceDRR = PhysicalPointImagefilter.Execute()
    sourceDRR_array_to_reshape = sitk.GetArrayFromImage(sourceDRR)
    sourceDRR_array_1d = np.ravel(sourceDRR_array_to_reshape, order='C').astype(np.float32)

    # 定义虚假的固定X光
    fixedImgArray_1d = np.zeros(DRRsize[0] * DRRsize[1], dtype=np.float32)

    # 初始化投影器
    projector = pySiddonGpu(NumThreadsPerBlock,
                            movImgArray_1d,
                            MovSize_forGpu,
                            MovSpacing_forGpu,
                            X0.astype(np.float32), Y0.astype(np.float32), Z0.astype(np.float32),
                            DRRsize_forGpu,
                            fixedImgArray_1d,
                            sourceDRR_array_1d,
                            source.astype(np.float32),
                            MovCenter_forGpu)

    # 设置位姿扰动值
    if augment_flag:
        d_alpha = random.uniform(-10, 10)
        d_beta = random.uniform(-10, 10)
        d_gamma = random.uniform(-10, 10)
        d_x = random.uniform(-20, 20)
        d_y = random.uniform(-20, 20)
        d_z = random.uniform(-20, 20)
    else:
        d_alpha = 0
        d_beta = 0
        d_gamma = 0
        d_x = 0
        d_y = 0
        d_z = 0

    # 设置变换参数
    Tr_ap_init = get_rigid_motion_mat_from_euler(np.deg2rad(-90), 'x', np.deg2rad(0), 'y', np.deg2rad(0), 'z', 0, -50, 0)
    Tr_lat_init = get_rigid_motion_mat_from_euler(np.deg2rad(0), 'x', np.deg2rad(90), 'y', np.deg2rad(-90), 'z', -50, 0, 0)
    Tr_delta = get_rigid_motion_mat_from_euler(np.deg2rad(d_alpha), 'x', np.deg2rad(d_beta), 'y', np.deg2rad(d_gamma), 'z', d_x, d_y, d_z)
    Tr_ap = np.dot(Tr_delta, Tr_ap_init)
    Tr_lat = np.dot(Tr_delta, Tr_lat_init)

    # 产生DRR
    invT_ap_1d = np.ravel(Tr_ap, order='C').astype(np.float32)
    drr1_to_reshape = projector.generateDRR(invT_ap_1d)
    drr1 = np.reshape(drr1_to_reshape, (DRRsize[1], DRRsize[0]), order='C')
    projector.computeMetric()
    invT_lat_1d = np.ravel(Tr_lat, order='C').astype(np.float32)
    drr2_to_reshape = projector.generateDRR(invT_lat_1d)
    drr2 = np.reshape(drr2_to_reshape, (DRRsize[1], DRRsize[0]), order='C')
    projector.computeMetric()

    # DRR像素规范化
    drr1 = (drr1 - np.min(drr1)) / (np.max(drr1) - np.min(drr1)) * 255
    drr2 = (drr2 - np.min(drr2)) / (np.max(drr2) - np.min(drr2)) * 255

    # 肺部标记点生成
    lung_fiducials_3d_list = []
    lung_fiducial_3d_indices = np.array(np.where(input_ct_mask_array > 0)).T
    point_count = 0
    while(point_count < 20):
        index = np.random.randint(0, lung_fiducial_3d_indices.shape[0])
        lung_fiducial_3d_index = lung_fiducial_3d_indices[index][::-1]
        lung_fiducial_3d = lung_fiducial_3d_index * movSpacing
        lung_fiducials_3d_list.append(lung_fiducial_3d)
        point_count += 1
    lung_fiducials_3d = np.asarray(lung_fiducials_3d_list)

    # 2D脊柱定位
    fiducials_3d_ap, fiducials_2d_ap, labels_2d_ap, transformation_ap, center_ap, source_ap = get_2d_3d_annotation(fiducial_file_path, Tr_ap, projector_info['focal_lenght'], movSize, movSpacing, np.asarray(DRRsize)[:2], np.asarray(DRRspacing)[:2])
    fiducials_3d_ap_lung, fiducials_2d_ap_lung, labels_2d_ap_lung, transformation_ap_lung, center_ap_lung, source_ap_lung = get_2d_3d_annotation_lung(lung_fiducials_3d_list, Tr_ap, projector_info['focal_lenght'], movSize, movSpacing, np.asarray(DRRsize)[:2], np.asarray(DRRspacing)[:2])
    # solvePnP(fiducial_file_path, Tr_ap, projector_info['focal_lenght'], movSize, movSpacing, np.asarray(DRRsize)[:2], np.asarray(DRRspacing)[:2])
    fiducials_3d_lat, fiducials_2d_lat, labels_2d_lat, transformation_lat, center_lat, source_lat = get_2d_3d_annotation(fiducial_file_path, Tr_lat, projector_info['focal_lenght'], movSize, movSpacing, np.asarray(DRRsize)[:2], np.asarray(DRRspacing)[:2])
    fiducials_3d = fiducials_3d_ap
    fiducials_2d = fiducials_2d_ap
    labels_2d = labels_2d_ap
    transformation = transformation_ap
    center = center_ap
    source = source_ap
    # fiducials_3d = fiducials_3d_lat
    # fiducials_2d = fiducials_2d_lat
    # labels_2d = labels_2d_lat
    # transformation = transformation_lat
    # center = center_lat
    # source = source_lat
    landmarks = np.zeros((25, 6))
    for k in range(fiducials_2d.shape[0]):
        landmark_pos = fiducials_2d[k]
        landmark_pos_3d = fiducials_3d[k]
        if landmark_pos[0] > DRRsize[0] or landmark_pos[1] > DRRsize[1] or \
            landmark_pos[0] < 0 or landmark_pos[1] < 0 :
            continue
        landmark_label = int(labels_2d[k]) - 1
        landmarks[landmark_label, 0] = 1
        landmarks[landmark_label, 1 : 3] = landmark_pos
        landmarks[landmark_label, 3 : 6] = landmark_pos_3d
    print(landmarks)

    landmarks_lung = np.zeros((20, 6))
    for k in range(fiducials_2d_ap_lung.shape[0]):
        landmark_pos = fiducials_2d_ap_lung[k]
        landmark_pos_3d = fiducials_3d_ap_lung[k]
        if landmark_pos[0] > DRRsize[0] or landmark_pos[1] > DRRsize[1] or \
            landmark_pos[0] < 0 or landmark_pos[1] < 0 :
            continue
        landmark_label = int(labels_2d_ap_lung[k]) - 1
        landmarks_lung[landmark_label, 0] = 1
        landmarks_lung[landmark_label, 1 : 3] = landmark_pos
        landmarks_lung[landmark_label, 3 : 6] = landmark_pos_3d
    print(landmarks_lung)

    # Test
    # movImgArray_1d = np.ravel(bone_ct_array.copy(), order='C').astype(np.float32)
    # projector_bone = pySiddonGpu(NumThreadsPerBlock,
    #                                 movImgArray_1d,
    #                                 MovSize_forGpu,
    #                                 MovSpacing_forGpu,
    #                                 X0.astype(np.float32), Y0.astype(np.float32), Z0.astype(np.float32),
    #                                 DRRsize_forGpu,
    #                                 fixedImgArray_1d,
    #                                 sourceDRR_array_1d,
    #                                 source.astype(np.float32),
    #                                 MovCenter_forGpu)
    # Tr_ap_init = get_rigid_motion_mat_from_euler(np.deg2rad(-90), 'x', np.deg2rad(0), 'y', np.deg2rad(0), 'z', 0, -600, 0)
    # Tr_lat_init = get_rigid_motion_mat_from_euler(np.deg2rad(0), 'x', np.deg2rad(90), 'y', np.deg2rad(-90), 'z', -600, 0, 0)
    # Tr_ap = Tr_ap_init
    # Tr_lat = Tr_lat_init
    # invT_ap_1d = np.ravel(Tr_ap, order='C').astype(np.float32)
    # drr1_to_reshape = projector_bone.generateDRR(invT_ap_1d)
    # drr1 = np.reshape(drr1_to_reshape, (DRRsize[1], DRRsize[0]), order='C')
    # projector_bone.computeMetric()
    # invT_lat_1d = np.ravel(Tr_lat, order='C').astype(np.float32)
    # drr2_to_reshape = projector_bone.generateDRR(invT_lat_1d)
    # drr2 = np.reshape(drr2_to_reshape, (DRRsize[1], DRRsize[0]), order='C')
    # projector_bone.computeMetric()
    # drr1_bone = (drr1 - np.min(drr1)) / (np.max(drr1) - np.min(drr1))
    # drr2_bone = (drr2 - np.min(drr2)) / (np.max(drr2) - np.min(drr2))
    # drr1 = drr1_background + drr1_bone * 0.5
    # drr2 = drr2_background + drr2_bone * 0.5
    # drr1 = (drr1 - np.min(drr1)) / (np.max(drr1) - np.min(drr1)) * 255
    # drr2 = (drr2 - np.min(drr2)) / (np.max(drr2) - np.min(drr2)) * 255

    # 显示DRR
    colors = matplotlib.cm.get_cmap('Paired')
    label_min = 0
    label_max = 25
    # plt.subplot(121)
    plt.imshow(drr1, cmap='gray')
    plt.axis('off')
    # for fiducial_2d in fiducials_2d_ap:
    #     plt.scatter(fiducial_2d[0], fiducial_2d[1], marker='x')
    # for k in range(len(labels_2d_ap)):
    #     if(fiducials_2d_ap[k, 0] > DRRsize[1] or fiducials_2d_ap[k, 0] < 0 or fiducials_2d_ap[k, 1] > DRRsize[0] or fiducials_2d_ap[k, 1] < 0):
    #         continue
    #     plt.scatter(fiducials_2d_ap[k, 0], fiducials_2d_ap[k, 1], marker='x', c=colors(((labels_2d_ap[k] - label_min) % 12) / 12))
    #     plt.scatter(fiducials_2d_ap[k, 0], fiducials_2d_ap[k, 1], marker='o', c='' , edgecolors=colors(((labels_2d_ap[k] - label_min) % 12) / 12), s=60, linewidths=2)
    #     # plt.text(fiducials_2d_ap[k, 0], fiducials_2d_ap[k, 1], str(labels_2d_ap[k]), c='b')
    for k in range(len(labels_2d_ap_lung)):
        if(fiducials_2d_ap_lung[k, 0] > DRRsize[1] or fiducials_2d_ap_lung[k, 0] < 0 or fiducials_2d_ap_lung[k, 1] > DRRsize[0] or fiducials_2d_ap_lung[k, 1] < 0):
            continue
        # plt.scatter(fiducials_2d_ap_lung[k, 0], fiducials_2d_ap_lung[k, 1], marker='x', c=colors(((labels_2d_ap_lung[k] - label_min) % 12) / 12))
        # plt.scatter(fiducials_2d_ap_lung[k, 0], fiducials_2d_ap_lung[k, 1], marker='o', c='' , edgecolors=colors(((labels_2d_ap_lung[k] - label_min) % 12) / 12), s=60, linewidths=2)
        plt.scatter(fiducials_2d_ap_lung[k, 0], fiducials_2d_ap_lung[k, 1], s=120, marker='+', c=(0, 1, 0), linewidths=2)
        # plt.text(fiducials_2d_ap[k, 0], fiducials_2d_ap[k, 1], str(labels_2d_ap[k]), c='b')
    for k in range(len(labels_2d_ap)):
        if(fiducials_2d_ap[k, 0] > DRRsize[1] or fiducials_2d_ap[k, 0] < 0 or fiducials_2d_ap[k, 1] > DRRsize[0] or fiducials_2d_ap[k, 1] < 0):
            continue
        plt.scatter(fiducials_2d_ap[k, 0], fiducials_2d_ap[k, 1], marker='x', c=colors(((labels_2d_ap[k] - label_min) % 12) / 12))
        plt.scatter(fiducials_2d_ap[k, 0], fiducials_2d_ap[k, 1], marker='o', c='' , edgecolors=colors(((labels_2d_ap[k] - label_min) % 12) / 12), s=60, linewidths=2)
        # plt.scatter(fiducials_2d_ap[k, 0], fiducials_2d_ap[k, 1], s=120, marker='+', c=(0, 1, 0), linewidths=2)
        # plt.text(fiducials_2d_ap[k, 0], fiducials_2d_ap[k, 1], str(labels_2d_ap[k]), c='b')
    # plt.subplot(122)
    # plt.imshow(drr2, cmap='gray')
    # # for fiducial_2d in fiducials_2d_lat:
    # #     plt.scatter(fiducial_2d[0], fiducial_2d[1], marker='x')
    # for k in range(len(labels_2d_lat)):
    #     plt.scatter(fiducials_2d_lat[k, 0], fiducials_2d_lat[k, 1], marker='x', c=colors((labels_2d_lat[k] - label_min) / label_max))
    #     # plt.text(fiducials_2d_lat[k, 0], fiducials_2d_lat[k, 1], str(labels_2d_lat[k]), c='b')
    # plt.subplot(122)
    # plt.imshow(drr1, cmap='gray')
    plt.axis('off')
    # plt.savefig("/home/leko/Nutstore Files/我的坚果云/jbhi/current/drr_sim_lung/{}_lung.png".format(ct_name), dpi=300)
    # plt.clf()
    plt.show()

    # # 保存数据
    # # h5_file = h5py.File("/home/leko/SCN-pytorch/data_hospital/lat/" + ct_name + ".h5", 'w')
    # h5_file = h5py.File("/home/leko/SCN-pytorch-jbhi/data/ap_crop_lung_hyper/data/" + ct_name + ".h5", 'w')
    # # h5_file = h5py.File("/home/leko/SCN-pytorch/data/lat/" + ct_name + ".h5", 'w')
    # # h5_file = h5py.File("/home/leko/SCN-pytorch/data/ap/" + ct_name + ".h5", 'w')
    # # h5_file = h5py.File("/home/leko/SCN-pytorch/data/ap_aug/" + ct_name + ".h5", 'w')
    # # h5_file = h5py.File("/home/leko/SCN-pytorch/data/ap_aug_clean/" + ct_name + ".h5", 'w')
    # h5_file.create_dataset('image', data=drr1.astype(np.float32))
    # # h5_file.create_dataset('image', data=drr2.astype(np.float32))
    # h5_file.create_dataset('landmarks', data=landmarks.astype(np.float32))
    # h5_file.create_dataset('landmarks_lung', data=landmarks_lung.astype(np.float32))
    # h5_file.create_dataset('transformation', data=transformation.astype(np.float32))
    # h5_file.create_dataset('center', data=center.astype(np.float32))
    # h5_file.create_dataset('source', data=source.astype(np.float32))
    # h5_file.close()

    # 保存图片
    # cv2.imwrite("/home/leko/SCN-pytorch/1.png", drr1)

    # 释放显存
    projector.delete()