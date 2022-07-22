import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2
import glob
import random


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

def quat2mat(quat):
    x, y, z, w = quat[0], quat[1], quat[2], quat[3]

    B = 1
    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    n = w2 + x2 + y2 + z2
    x = x / n
    y = y / n
    z = z / n
    w = w / n
    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([1 - 2*y2 - 2*z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, 1 - 2*x2 - 2*z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, 1 - 2*x2 - 2*y2]).reshape(3, 3)
    return rotMat

def read_json_file(path):
    '''
    Read json file
    '''
    json_data = []

    with open(path) as fp:
        for json_object in fp:
            json_data.append(json.loads(json_object))

    return json_data

def pose_dict_to_numpy(pose):
    '''
    Convert pose dictionary to numpy array
    '''

    pose = np.array([pose['position']['x'],
                     pose['position']['y'],
                     pose['position']['z'],
                     pose['rotation']['x'],
                     pose['rotation']['y'],
                     pose['rotation']['z'],
                     pose['rotation']['w']
                    ])

    return pose

def pose_2_matrix(pose):
    '''
    Function to convert pose to transformation matrix
    '''
    flip_x = torch.eye(4)
    flip_x[2, 2] *= -1
    flip_x[1, 1] *= -1

    rot_mat = quat2mat(pose[3:]) # num_views 3 3
    translation_mat = pose[:3] # num_views 3 1
    translation_mat = translation_mat.reshape((3, 1))
            
    transformation_mat = torch.hstack([rot_mat, translation_mat])
    transformation_mat = np.vstack((transformation_mat, np.array([0.0, 0.0, 0.0, 1.0])))
    transformation_mat = torch.from_numpy(transformation_mat)
    flip_x = flip_x.inverse().type_as(transformation_mat)

    # 180 degree rotation around x axis due to blender's coordinate system
    return (transformation_mat @ flip_x).squeeze()

def construct_camera_matrix(focal_x, focal_y, c_x, c_y, res):
    '''
    Obtain camera intrinsic matrix
    '''

    K = np.array([[focal_x * res, 0, c_x * res],
                  [0, -focal_y * res, c_y * res],
                  [0, 0, -1]])

    return K

def load_image_names(path):
    '''
    Load names of image, mask, and camera pose file names
    '''

    camera_pose = []
    mask = []
    view = []
    depth = []

    for paths, dirs, files in os.walk(path):
        for file in files:
            if "view_" not in file:
                continue

            view_number = file.split('_')[1].split('.')[0]
            view_name = os.path.join(path, 'view_' + view_number + '.jpg')
            mask_name = os.path.join(path, 'mask_' + view_number + '.jpg')
            pose_name = os.path.join(path, ('CameraPose_' + view_number + '.json'))
            depth_name = os.path.join(path, 'depth_' + view_number + '.tiff')

            view.append(view_name)
            mask.append(mask_name)
            camera_pose.append(pose_name)
            depth.append(depth_name)

    view = sorted(view)
    mask = sorted(mask)
    camera_pose = sorted(camera_pose)
    depth = sorted(depth)

    return view, mask, camera_pose, depth

def split_image(view_image, view_mask, pose_params, view_depth, num_views = 3):

    '''
    Split the images in respective views
    Arguments:
        view_image   :   H x W x 3 - image of concatenated views
        num_views    :      scalar - number of views to split image

    Returns:
        images_split :  dictionary - index and image
    '''

    view_list = []
    mask_list = []
    pose_list = []
    depth_list = []

    width = int(view_image.shape[1] / num_views)
    images_split = {}
    images_split['num_views'] = num_views

    for i in range(num_views):
        index = i - int(num_views / 2)

        # Extracting parameters
        pose_dict = pose_params[i]
        pose_np = pose_dict_to_numpy(pose_dict)
        pose = pose_2_matrix(torch.from_numpy(pose_np))
        pose = pose.cpu().detach().numpy()
        # pose = np.linalg.inv(pose)

        image = view_image[ :, i * width:(i + 1) * width]
        # .transpose((2, 0, 1))
        mask = view_mask[ :, i * width:(i + 1) * width]
        # .transpose((2, 0, 1))

        depth = view_depth[ :, i * width:(i + 1) * width]
        # .transpose((2, 0, 1))

        # Inserting in list to stack
        if index == 0:
            view_list.insert(index, image)
            mask_list.insert(index, mask)
            pose_list.insert(index, pose)
            depth_list.insert(index, depth)

        else:
            view_list.append(image)
            mask_list.append(mask)
            pose_list.append(pose)
            depth_list.append(depth)

    images_split["views"] = np.stack(view_list)
    images_split["masks"] = np.stack(mask_list)
    images_split["poses"] = np.stack(pose_list)
    images_split["depths"] = np.stack(depth_list)

    return view_list[0], mask_list[0], pose_list[0], depth_list[0]

def load_draco_data(basedir, res = 1, skip = 1):
    K = construct_camera_matrix(888.88, 1000, 320, 240, res)
    views, masks, poses, depths = load_image_names(basedir)

    all_imgs = []
    all_masks = []
    all_poses = []
    all_depths = []

    for index in range(len(views)):
        image_rgb_concat = imageio.imread(views[index]) / 255.0

        image_mask_concat = cv2.imread(masks[index], cv2.IMREAD_GRAYSCALE) / 255.0
        image_mask_concat = image_mask_concat.reshape(image_mask_concat.shape[0], image_mask_concat.shape[1], 1)
        image_mask_concat = np.rint(image_mask_concat)

        camera_pose = read_json_file(poses[index])

        image_depth_concat = imageio.imread(depths[index])
        image_depth_concat = image_depth_concat.reshape(image_depth_concat.shape[0], image_depth_concat.shape[1], 1)

        n_image, n_mask, n_pose, n_depth = split_image(image_rgb_concat, image_mask_concat, camera_pose, image_depth_concat)

        h, w = n_image.shape[:2]
        resized_h = round(h * res)
        resized_w = round(w * res)
        n_image = cv2.resize(n_image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
        n_mask = cv2.resize(n_mask, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
        n_depth= np.where(n_depth == np.inf, 0, n_depth)
        n_depth = cv2.resize(n_depth, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

        n_image = np.dstack((n_image, n_mask))
        all_imgs.append(n_image)
        all_masks.append(n_mask)
        all_poses.append(n_pose)
        all_depths.append(n_depth)

    all_imgs = np.array(all_imgs).astype(np.float32)
    all_masks = np.array(all_masks).astype(np.float32)
    all_poses = np.array(all_poses)
    all_depths = np.array(all_depths).astype(np.float32)

    indices = np.arange(len(all_imgs))
    i_train = np.random.choice(indices, round(0.8 * len(all_imgs)), replace=False)
    indices = np.array(list(set(indices).difference(set(i_train))))
    i_val = np.random.choice(indices, round(0.1 * len(all_imgs)), replace=False)
    i_test = np.array(list(set(indices).difference(set(i_val))))
    i_split = [i_train, i_val, i_test]

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0)

    meta = {
            "intrinsic_mat": K,
            "height": resized_h,
            "width": resized_w,
            "fx": 888.88 * res
        }

    return all_imgs, all_poses, render_poses, meta, all_depths, all_masks, i_split


