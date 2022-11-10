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
import h5py
from scipy.spatial.transform import Rotation as R


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


def load_h5(path):
    fx_input = h5py.File(path, "r")
    x = fx_input["data"][:]
    fx_input.close()
    return x

def load_models(path):
    models = []
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            model = os.path.basename(line[:-1])
            model = model[:-15]
            models.append(model)

    return models

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

def read_txt_file(path):
    arr = []
    with open(path, "r") as f:
        lines = f.readlines()

        for line in lines:
            vec = np.array([float(i) for i in line.split(',')])
            arr.append(vec)

    arr = np.array(arr)
    return arr

def extract_pose(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)  # Also outputs Jacobian
    t = tvec
    # t = -R.T @ tvec

    # R = np.linalg.inv(R)
    R = R.T
    t = -t

    pose = np.identity(4)
    pose[:3, :3] = R
    pose[:3, 3] = t
    # pose = np.linalg.inv(pose)

    return pose

def load_dataset_old(directory, canonical_pose = None):
    # print(directory)

    cam_data_path = os.path.join(os.path.dirname(directory), "cameras")
    rots_path = os.path.join(cam_data_path, "rvecs.txt")
    ts_path = os.path.join(cam_data_path, "tvecs.txt")
    rvecs = read_txt_file(rots_path)
    tvecs = read_txt_file(ts_path)

    intrinsics_path = os.path.join(cam_data_path, "intrinsics.txt")
    K = read_txt_file(intrinsics_path)
    cams = {
            "width": 1280,
            "height": 720,
            "fx": K[0][0],
            "fy": K[1][1],
            "cx": K[0][2],
            "cy": K[1][2]
            }

    imgs = {}
    image_dir = directory
    images = glob.glob(image_dir + "/**/*0.jpg", recursive = True)
    images.sort()

    # mask_dir = os.path.join(directory, "mask/")
    # depth_dir = os.path.join(directory, "depth/")

    for i in range(len(images)):
        image_current = images[i]
        image_id = int(os.path.dirname(image_current).split("_")[-1])
        # image_parent_dir = image_current.split("/")[-2]

        pose = extract_pose(rvecs[i], tvecs[i])
        # print(pose)

        if canonical_pose is not None:
            canonical_pose_4 = np.identity(4)
            canonical_pose_4[:3, :3] = canonical_pose

            t = np.array([0.0, -0.5, 4.5]).T
            final_pose = np.identity(4)
            final_pose[:3, -1] = -t
            final_pose = canonical_pose_4 @ final_pose
            final_pose[:3, -1] += t
            final_pose = pose @ final_pose
            final_pose = np.linalg.inv(final_pose)
            pose = final_pose

            # canonical_pose_4 = np.identity(4)
            # canonical_pose_4[:3, :3] = canonical_pose

            # t = np.array([0.0, -0.5, 4.5]).T
            # final_pose = np.identity(4)
            # final_pose[:3, -1] = -t
            # final_pose = canonical_pose_4 @ final_pose
            # final_pose[:3, -1] += t
            # final_pose = np.linalg.inv(pose) @ final_pose
            # final_pose = np.linalg.inv(final_pose)
            # pose = final_pose

        imgs[i] = {
            "camera_id": image_id,
            "t": pose[:3, 3].reshape(3, 1),
            "R": pose[:3, :3],
            "path": images[i],
            "pose": pose
        }

        # imgs[i]["mask_path"] = os.path.join(mask_dir, "%s/%s_seg.png" % (image_parent_dir, image_id))
        # imgs[i]["depth_path"] = os.path.join(depth_dir, "%s/%s_depth.npz" % (image_parent_dir, image_id))

    return imgs, cams

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def normalize(x):
    return x / np.linalg.norm(x)

def poses_avg(poses):
    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    # c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    c2w = viewmatrix(vec2, up, center)

    return c2w

def recenter_poses(poses):
    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses

def convert_frame(transformation_mat):
    flip_x = np.identity(4)
    flip_x[2, 2] *= -1
    flip_x[1, 1] *= -1

    flip_x = np.linalg.inv(flip_x)

    # 180 degree rotation around x axis due to blender's coordinate system
    return (np.linalg.inv(flip_x) @ transformation_mat)

def load_dataset(directory, scale = 1.0, canonical_pose = None):
    params_path = os.path.join(os.path.dirname(directory), "09_11_params.txt")
    params = np.loadtxt(f"{params_path}",
            dtype=[
                ('cam_id', int), ('width', int), ('height', int),
                ('fx', float), ('fy', float),
                ('cx', float), ('cy', float),
                ('k1', float), ('k2', float),
                ('p1', float), ('p2', float),
                ('cam_name', '<U10'),
                ('qvecw', float), ('qvecx', float), ('qvecy', float), ('qvecz', float),
                ('tvecx', float), ('tvecy', float), ('tvecz', float)
                ])

    params = np.sort(params, order='cam_name')
    proj_mats = {}
    extrs = []
    extrs_map = {}

    imgs = {}
    image_dir = directory

    # mask_dir = os.path.join(directory, "mask/")
    # depth_dir = os.path.join(directory, "depth/")

    for i in range(params.shape[0]):
        qvec = [params[i]['qvecx'], params[i]['qvecy'], params[i]['qvecz'], params[i]['qvecw']]
        tvec = np.asarray([params[i]['tvecx'], params[i]['tvecy'], params[i]['tvecz']])
        r = R.from_quat(qvec).as_matrix()
        extr = np.vstack([np.hstack([r, tvec[:,None]]), np.zeros((1, 4))])
        extr[3, 3] = 1

        # extr = convert_frame(extr)
        extr = np.linalg.inv(extr)
        # extr = np.concatenate([extr[:, 1:2], -extr[:, 0:1], extr[:, 2:]], 1)
        # extrs.append(extr)
        # extrs_map[len(extrs) - 1] = i
        # extr = extr[:3]
    
        intr = np.eye(3)
        intr[0, 0] = params[i]['fx'] * scale
        intr[1, 1] = -params[i]['fy'] * scale
        intr[0, 2] = params[i]['cx'] * scale
        intr[1, 2] = params[i]['cy'] * scale
        intr[2, 2] = -1

        # proj_mats[params[i]['cam_name']] = intr @ extr
        w = params[i]['width'] * scale
        h = params[i]['height'] * scale
        
        image_id = params[i]['cam_name']
        image_dir_path = os.path.join(image_dir, image_id)
        image_current = os.path.join(image_dir_path, os.listdir(image_dir_path)[0])

        if canonical_pose is not None:
            canonical_pose_4 = np.identity(4)
            canonical_pose_4[:3, :3] = canonical_pose

            t = np.array([0.0, -0.5, 4.5]).T
            final_pose = np.identity(4)
            final_pose[:3, -1] = -t
            final_pose = canonical_pose_4 @ final_pose
            final_pose[:3, -1] += t
            final_pose = pose @ final_pose
            final_pose = np.linalg.inv(final_pose)
            pose = final_pose

            # canonical_pose_4 = np.identity(4)
            # canonical_pose_4[:3, :3] = canonical_pose

            # t = np.array([0.0, -0.5, 4.5]).T
            # final_pose = np.identity(4)
            # final_pose[:3, -1] = -t
            # final_pose = canonical_pose_4 @ final_pose
            # final_pose[:3, -1] += t
            # final_pose = np.linalg.inv(pose) @ final_pose
            # final_pose = np.linalg.inv(final_pose)
            # pose = final_pose

        distortion = np.array([params[i]["k1"], params[i]["k2"], params[i]["p1"], params[i]["p2"]])
        imgs[i] = {
            "camera_id": image_id,
            "t": extr[:3, 3].reshape(3, 1),
            "R": extr[:3, :3],
            "path": image_current,
            "pose": extr,
            "K": intr,
            "width": w,
            "height": h,
            "distortion": distortion
        }

        # imgs[i]["mask_path"] = os.path.join(mask_dir, "%s/%s_seg.png" % (image_parent_dir, image_id))
        # imgs[i]["depth_path"] = os.path.join(depth_dir, "%s/%s_depth.npz" % (image_parent_dir, image_id))

    # extrs = np.array(extrs)
    # extrs = recenter_poses(extrs)
    # for i in range(len(extrs)):
        # imgs[extrs_map[i]]["pose"] = extrs[i] 

    return imgs, None

def pallette_to_labels(mask):
    uniq_vals = np.unique(mask)

    for i in range(len(uniq_vals)):
        mask = np.where(mask == uniq_vals[i], i, mask)

    return mask

def load_brown_real_data(basedir, res=1, skip=1, max_ind=54, canonical_pose = None):
    imgs, cams = load_dataset(basedir, res, canonical_pose)
    all_ids = []
    all_imgs = []
    all_poses = []
    all_seg_masks = []
    all_depths = []

    if cams is None:
        cams = []

    for index in range(0, max_ind, skip):
        if index >= len(imgs):
            break

        all_ids.append(imgs[index]["camera_id"])

        newK = []
        n_image = imageio.imread(imgs[index]["path"]) / 255.0
        K = np.copy(imgs[index]["K"])
        K[1][1] *= -1
        K[2][2] *= -1
        newK, roi = cv2.getOptimalNewCameraMatrix(K, imgs[index]["distortion"], (n_image.shape[1], n_image.shape[0]), 1, (n_image.shape[1], n_image.shape[0]))
        n_image = cv2.undistort(n_image[:, :, :3], K, imgs[index]["distortion"], None, newK) 
        n_image = n_image[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
        newK[1][1] *= -1
        newK[2][2] *= -1

        h, w = n_image.shape[:2]
        resized_h = round(h * res)
        resized_w = round(w * res)
        n_image = cv2.resize(n_image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
        n_image = np.array(n_image).astype(np.float32)
        all_imgs.append(n_image)

        n_pose = imgs[index]["pose"]
        all_poses.append(n_pose)
        
        if type(cams) == list:
            if newK != []:
                cams.append(newK) 
            else:
                cams.append(imgs[index]["K"]) 

        # n_seg_mask = cv2.imread(imgs[index]["mask_path"], cv2.IMREAD_GRAYSCALE)
        # n_seg_mask = cv2.resize(n_seg_mask, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
        # n_seg_mask = pallette_to_labels(n_seg_mask)
        # all_seg_masks.append(n_seg_mask)

        # n_depth = np.load(imgs[index]["depth_path"])['arr_0']
        # n_depth = np.where(n_depth == np.inf, 0, n_depth)
        # n_depth = np.where(n_depth > 100, 0, n_depth)
        # n_depth = cv2.resize(n_depth, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
        # all_depths.append(n_depth)
    
    # all_imgs = np.array(all_imgs).astype(np.float32)
    all_poses = np.array(all_poses)
    # all_seg_masks = np.array(all_seg_masks).astype(np.float32)
    # all_depths = np.array(all_depths).astype(np.float32)

    i_val = []
    for side_idx in range(6):
        val_idx = np.random.randint(side_idx * 9, side_idx * 9 + 9) 
        i_val.append(val_idx)

    indices = np.arange(len(all_imgs))
    i_train = np.array(list(set(indices).difference(set(i_val))))
    i_test = i_val
    i_split = [i_train, i_val, i_test]

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40+1)[:-1]], 0)

    return all_imgs, all_poses, render_poses, cams, i_split, [imgs[0]["height"], imgs[0]["width"], imgs[0]["K"][0][0]]
