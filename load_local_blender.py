import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2
import glob
import random


os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

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
    pose = pose[0]
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
    translation_mat = translation_mat.reshape((3,1))
            
    transformation_mat = torch.hstack([rot_mat, translation_mat])
    transformation_mat = np.vstack((transformation_mat, np.array([0.0,0.0,0.0,1.0])))
    transformation_mat = torch.from_numpy(transformation_mat)
    flip_x = flip_x.inverse().type_as(transformation_mat)

    # 180 degree rotation around x axis due to blender's coordinate system
    return (transformation_mat @ flip_x).squeeze()

def load_dataset(directory):
    # print(directory)
    # directory = "/home2/ragaram/scene_00/"

    pose_dir = directory + "pose/"
    pose_files = glob.glob(pose_dir + "*.json")
    pose_files.sort()

    imgs = {}
    image_dir = directory + "rgb/"
    images = glob.glob(image_dir + "*.png")
    images.sort()

    segmetation_meta_dir = directory + "seg_metadata/"
    mask_dir = directory + "mask/"
    depth_dir = directory + "depth/"
    m = 0

    for i in range(len(images)):
        image_current = images[i]
        image_id = int(image_current.split("_")[-3])
        '''
        if m != image_id:
            print("Missing image for id ", image_id)
            m = image_id + 1
        else:
            m = m + 1'''

        camera_pose = read_json_file(pose_files[i])
        pose_np     = pose_dict_to_numpy(camera_pose)
        pose        = pose_2_matrix(torch.from_numpy(pose_np))
        pose = pose.cpu().detach().numpy()
        # print(pose)
        #Got the scale below from NOCS function above
        pose[:3, 3] = pose[:3, 3]
        # * 0.16072596331333025
        # pose = np.linalg.inv(pose)
        # print(pose)

        imgs[image_id] = {
            "camera_id": image_id,
            "r": pose[:3, :3],
            "t": pose[:3, 3].reshape(3, 1),
            "R": pose[:3, :3],
            "center": pose[:3, 3].reshape(3,1),
            "path": images[i],
            "pose": pose
        }

        imgs[image_id]["mask_path"] = mask_dir + "frame_" + image_current.split("_")[-3] + "_Mask_00.png"
        imgs[image_id]["segmentation_meta"] = mask_dir + "frame_" + image_current.split("_")[-3] + "_Mask_00.png"
        imgs[image_id]["depth_path"] = depth_dir + "frame_" + image_current.split("_")[-3] + "_Depth_00.exr"

    cams = {0: {'width': 640, 'height': 480, 'fx': 888.8889, 'fy': 1000.0000, 'px': 320.0000, 'py': 240.0000}}
    #exit()
    return imgs, cams

def main_loader(root_dir, scale):
    imgs, cams = load_dataset(root_dir)
    #print(imgs)
    #print(cams)

    cams[0]["fx"] = fx = cams[0]["fx"] * scale
    cams[0]["fy"] = fy = cams[0]["fy"] * scale
    cams[0]["px"] = px = cams[0]["px"] * scale
    cams[0]["py"] = py = cams[0]["py"] * scale

    rand_key = random.choice(list(imgs))
    test_img = cv2.imread(imgs[rand_key]["path"])
    h, w = test_img.shape[:2]
    cams[0]["height"] = round(h * scale)
    cams[0]["width"] = round(w * scale)

    cams[0]["intrinsic_mat"] = np.array([[fx, 0, px],
                              [0, fy, py],
                              [0, 0, 1]])

    for it in range(len(imgs)):
        #print(imgs[it]["path"])
        pose_rotation = imgs[it]['r']
        pose_translation = imgs[it]['t']
        pose = np.hstack((pose_rotation, pose_translation))
        pose = np.reshape(pose, (1, 3, 4))
        pose = torch.from_numpy(pose)
        #print(imgs[it]["rays"].shape)

    return imgs, cams 

def load_local_blender_data(basedir, res=1, skip=1):
    imgs, cams = main_loader(basedir, res)
    all_imgs = []
    all_poses = []
    all_depths = []

    for index in range(len(imgs)):
        # n_image = cv2.imread(imgs[index]["path"])
        # n_image = cv2.cvtColor(n_image, cv2.COLOR_BGR2RGB) / 255.0
        n_image = imageio.imread(imgs[index]["path"]) / 255.0
        h, w = n_image.shape[:2]
        resized_h = round(h * res)
        resized_w = round(w * res)
        n_image = cv2.resize(n_image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
        # n_image = torch.from_numpy(n_image)
        all_imgs.append(n_image)

        n_pose = imgs[index]["pose"]
        # n_pose = torch.from_numpy(n_pose)
        all_poses.append(n_pose)
        
        n_depth = np.array(cv2.imread(imgs[index]["depth_path"], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH))[:, :, 0]
        n_depth = np.where(n_depth == np.inf, 0, n_depth)
        # n_depth = imageio.imread(imgs[index]["depth_path"])
        # n_depth = n_depth.reshape(n_depth.shape[0], n_depth.shape[1], 1)
        n_depth = cv2.resize(n_depth, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
        all_depths.append(n_depth)
    
    all_imgs = np.array(all_imgs).astype(np.float32)
    all_poses = np.array(all_poses)
    all_depths = np.array(all_depths).astype(np.float32)

    indices = np.arange(len(all_imgs))
    i_train = np.random.choice(indices, round(0.8 * len(all_imgs)), replace=False)
    indices = np.array(list(set(indices).difference(set(i_train))))
    i_val = np.random.choice(indices, round(0.1 * len(all_imgs)), replace=False)
    i_test = np.array(list(set(indices).difference(set(i_val))))
    i_split = [i_train, i_val, i_test]

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40+1)[:-1]], 0)

    return all_imgs, all_poses, render_poses, cams[0], i_split


