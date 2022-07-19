import numpy as np
import open3d as o3d
import os, glob
import argparse
import matplotlib.pyplot as plt

def get_files(input_directory):

    image_files = glob.glob(input_directory + "**.png")
    image_files.sort()
    # mask_files = glob.glob(input_directory + "**_mask.png")
    # mask_files.sort()
    depth_files = glob.glob(input_directory + "depth_**.npy")
    depth_files.sort()
    K_file = input_directory + "K.npy"
    poses = glob.glob(input_directory + "c2w_**.npy")
    poses.sort()

    return {
            "image_files": image_files, 
            # "mask_files": mask_files, 
            "depth_files": depth_files, 
            "K_file": K_file, 
            "pose_files": poses}


def inverse_project_points(K, image, depth, pose, near = 0.0, far = 4.0, segmentation = False):

    # import pdb
    # pdb.set_trace()
    x, y = np.indices((image.shape[0], image.shape[1]))
    _1 = np.ones(x.reshape(1, -1).shape)
    pts = np.vstack([y.reshape(1, -1), x.reshape(1, -1), _1])

    d = depth.reshape(1, -1)

    pts = np.linalg.inv(K) @ pts
    pts = pts / pts[2, :]
    pts = pts * d

    pts_mask = ((pts[2, :] > near) * 1 * (pts[2, :] < far) * 1 == 1)

    pts_color = image.reshape(-1, 3)
    # print(pose.shape)
    pts = (pose @ np.vstack([pts, np.ones((1, pts.shape[-1]))]))[:3, :]
    pts = pts.T


    # pts = pts[pts_mask, :]
    # pts_color = pts_color[pts_mask, :]

    if segmentation:
        pts_mask_seg = (pts_color.sum(-1) != 3.0)
        # print(pts_mask_seg.shape)
        pts = pts[pts_mask_seg, :]
        pts_color = pts_color[pts_mask_seg, :]
    return pts, pts_color


def points_to_o3d(pts, pts_colors):

    pcd_o3d = o3d.geometry.PointCloud()

    pcd_o3d.points = o3d.utility.Vector3dVector(pts)
    pcd_o3d.colors = o3d.utility.Vector3dVector(pts_colors)

    return pcd_o3d

def SFM(input_directory, max_files = None, near = 0.0, far = 4.0, skip = 1):
    """
    Reconstruct the scene from NeRF outputs
    """

    in_directory = os.path.join(input_directory,"")
    all_files = get_files(in_directory)

    num_files_to_visualize = len(all_files["image_files"])

    if max_files is not None:
        num_files_to_visualize = max_files


    K = np.load(all_files["K_file"])[:3, :3]
    pointcloud_list = []
    segmentation_list = []
    for i in range(0, num_files_to_visualize, skip):
        image = plt.imread(all_files["image_files"][i])[:, :, :3]
        # mask = plt.imread(all_files["mask_files"][i])[:, :, :3]
        depth = np.load(all_files["depth_files"][i])
        pose = np.load(all_files["pose_files"][i])
        pose = np.vstack([pose, np.array([0, 0, 0, 1])])
        # pose = np.linalg.inv(pose)

        pts, pts_colors = inverse_project_points(K, image, depth, pose, near = near, far = far)
        # pts_s, pts_colors_s = inverse_project_points(K, mask, depth, pose, near = near, far = far, segmentation = True)
        pcd_o3d = points_to_o3d(pts, pts_colors)
        # pcd_o3d_s = points_to_o3d(pts_s, pts_colors_s)
        pointcloud_list.append(pcd_o3d)
        # segmentation_list.append(pcd_o3d_s)
        # o3d.visualization.draw_geometries([pcd_o3d])

    o3d.visualization.draw_geometries(pointcloud_list)
    # o3d.visualization.draw_geometries(segmentation_list)

if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Reconstruction from NeRF outputs")
    parser.add_argument("--input", required=True, type = str)
    parser.add_argument("--max_files", default=None, type = int)
    parser.add_argument("--near", default=0.0, type = float)
    parser.add_argument("--far", default=4.0, type = float)
    parser.add_argument("--skip", default=1, type = int)


    args = parser.parse_args()
    SFM(args.input, max_files = args.max_files, near = args.near, far = args.far, skip = args.skip)
