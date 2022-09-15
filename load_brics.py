import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2
import glob
import random
import matplotlib.pyplot as plt
import pickle


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

def read_pickle_file(path):
    objects = []
    with open(path, "rb") as fp:
        while True:
            try:
                obj = pickle.load(fp)
                objects.append(obj)

            except EOFError:
                break

    return objects 

def load_dataset(directory):
    # print(directory)

    cam_data_path = os.path.join(directory, "cam_data.pkl")
    cam_data = read_pickle_file(cam_data_path)[0]
    cams = {"width": 1280, "height": 720}

    imgs = {}
    image_dir = os.path.join(directory, "render/")
    images = glob.glob(image_dir + "**/*.png", recursive = True)
    images.sort()

    mask_dir = os.path.join(directory, "mask/")
    depth_dir = os.path.join(directory, "depth/")

    for i in range(len(images)):
        image_current = images[i]
        image_id = os.path.basename(image_current).split(".")[0]
        image_parent_dir = image_current.split("/")[-2]

        cam = cam_data[image_id]["K"]
        [cams["fx"], cams["fy"], cams["cx"], cams["cy"]] = cam
        pose = cam_data[image_id]["extrinsics_opencv"]
        pose = np.vstack([pose, np.array([0, 0, 0, 1])])
        pose = np.linalg.inv(pose)
        # print(pose)

        imgs[i] = {
            "camera_id": image_id,
            "t": pose[:3, 3].reshape(3, 1),
            "R": pose[:3, :3],
            "path": images[i],
            "pose": pose
        }

        imgs[i]["mask_path"] = os.path.join(mask_dir, "%s/%s_seg.png" % (image_parent_dir, image_id))
        imgs[i]["depth_path"] = os.path.join(depth_dir, "%s/%s_depth.npz" % (image_parent_dir, image_id))

    return imgs, cams

def main_loader(root_dir, scale):
    imgs, cams = load_dataset(root_dir)
    #print(imgs)
    #print(cams)

    cams["fx"] = fx = cams["fx"] * scale
    cams["fy"] = fy = cams["fy"] * scale
    cams["cx"] = cx = cams["cx"] * scale
    cams["cy"] = cy = cams["cy"] * scale

    rand_key = random.choice(list(imgs))
    test_img = cv2.imread(imgs[rand_key]["path"])
    h, w = test_img.shape[:2]
    cams["height"] = round(h * scale)
    cams["width"] = round(w * scale)

    cams["intrinsic_mat"] = np.array([
        [fx, 0, cx],
        [0, -fy, cy],
        [0, 0, -1]
        ])

    return imgs, cams 

def pallette_to_labels(mask):
    uniq_vals = np.unique(mask)

    for i in range(len(uniq_vals)):
        mask = np.where(mask == uniq_vals[i], i, mask)

    return mask

def load_brics_data(basedir, res=1, skip=1, max_ind=54):
    imgs, cams = main_loader(basedir, res)
    all_imgs = []
    all_poses = []
    all_seg_masks = []
    all_depths = []

    for index in range(0, max_ind, skip):
        n_image = imageio.imread(imgs[index]["path"]) / 255.0
        h, w = n_image.shape[:2]
        resized_h = round(h * res)
        resized_w = round(w * res)
        n_image = cv2.resize(n_image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
        all_imgs.append(n_image)

        n_pose = imgs[index]["pose"]
        all_poses.append(n_pose)
        
        n_seg_mask = cv2.imread(imgs[index]["mask_path"], cv2.IMREAD_GRAYSCALE)
        n_seg_mask = cv2.resize(n_seg_mask, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
        n_seg_mask = pallette_to_labels(n_seg_mask)
        all_seg_masks.append(n_seg_mask)

        n_depth = np.load(imgs[index]["depth_path"])['arr_0']
        n_depth = np.where(n_depth == np.inf, 0, n_depth)
        n_depth = cv2.resize(n_depth, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
        all_depths.append(n_depth)
    
    all_imgs = np.array(all_imgs).astype(np.float32)
    all_poses = np.array(all_poses)
    all_seg_masks = np.array(all_seg_masks).astype(np.float32)
    all_depths = np.array(all_depths).astype(np.float32)

    indices = np.arange(len(all_imgs))
    i_train = np.random.choice(indices, round(0.8 * len(all_imgs)), replace=False)
    indices = np.array(list(set(indices).difference(set(i_train))))
    i_val = np.random.choice(indices, round(0.1 * len(all_imgs)), replace=False)
    i_test = np.array(list(set(indices).difference(set(i_val))))
    i_split = [i_train, i_val, i_test]

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40+1)[:-1]], 0)

    return all_imgs, all_poses, render_poses, cams, all_seg_masks, all_depths, i_split
