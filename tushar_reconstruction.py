import numpy as np
import open3d as o3d
import os, glob
import argparse
import matplotlib.pyplot as plt
import imageio
import cv2
import json
import torch

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

def inverse_project_points(K, image, depth, pose):
    
    #pose = np.linalg.inv(pose)
    #pose = np.vstack((pose,np.array([0.0,0.0,0.0,1.0])))
    #pose = np.linalg.inv(pose)
    #import pdb
    #pdb.set_trace()
    x, y = np.indices((image.shape[0], image.shape[1]))
    _1 = np.ones(x.reshape(1, -1).shape)
    pts = np.vstack([y.reshape(1, -1), x.reshape(1, -1), _1])

    d = depth.reshape(1, -1)

    pts = np.linalg.inv(K) @ pts
    pts = pts / pts[2, :]
    pts = pts * d

    #pts_mask = ((pts[2, :] > near) * 1 * (pts[2, :] < far) * 1 == 1)

    pts_color = image.reshape(-1, 3)
    # print(pose.shape)
    
    pts = (pose @ np.vstack([pts, np.ones((1, pts.shape[-1]))]))[:3]
    pts = pts.T
    
    
    return pts, pts_color



def get_files(input_directory):

    image_files = glob.glob(input_directory + "image_*.png")
    image_files.sort()
    depth_files = glob.glob(input_directory + "depth_*.npy")
    depth_files.sort()
    K_file = input_directory + "intrinsics.npy"
    poses = glob.glob(input_directory + "pose_*.npy")
    poses.sort()
    return {"image_files": image_files, "depth_files": depth_files, "K_file": K_file, "pose_files": poses}


def points_to_o3d(pts, pts_colors = None):

    pcd_o3d = o3d.geometry.PointCloud()

    pcd_o3d.points = o3d.utility.Vector3dVector(pts)
    if pts_colors is not None:
        pcd_o3d.colors = o3d.utility.Vector3dVector(pts_colors)

    return pcd_o3d




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



def pose_2_matrix(pose):
            '''
            Function to convert pose to transformation matrix
            '''
            import pdb
            pdb.set_trace()
            flip_x = torch.eye(4)
            flip_x[2, 2] *= -1
            flip_x[1, 1] *= -1

            rot_mat = quat2mat(pose[3:]) # num_views 3 3
            translation_mat = pose[:3] # num_views 3 1
            translation_mat = translation_mat.reshape((3,1))
            
            

            transformation_mat = torch.hstack([rot_mat, translation_mat])
            
            
            transformation_mat = np.vstack(( transformation_mat,np.array([0.0,0.0,0.0,1.0])))
            transformation_mat = torch.from_numpy(transformation_mat)
            flip_x = flip_x.inverse().type_as(transformation_mat)

            # 180 degree rotation around x axis due to blender's coordinate system
            return (transformation_mat @ flip_x).squeeze()

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


def points_to_o3d(pts, pts_colors = None):

    pcd_o3d = o3d.geometry.PointCloud()

    pcd_o3d.points = o3d.utility.Vector3dVector(pts)
    if pts_colors is not None:
        pcd_o3d.colors = o3d.utility.Vector3dVector(pts_colors)

    return pcd_o3d

def SFM():
    """
    Reconstruct the scene from NeRF outputs
    """

    # in_directory = os.path.join(input_directory,"")
    # all_files = get_files(in_directory)

    depth_path = './logs/gt_registration/renderonly_test_000000/depth/'
    rgb_path = './logs/gt_registration/renderonly_test_000000/rgb/'
    poses_path = './logs/gt_registration/renderonly_test_000000/pose/'
    # mask_path = 'logs/gt_registration/renderonly_test_000000/mask/'

    depths = os.listdir(depth_path)
    depths.sort()

    poses = os.listdir(poses_path)
    poses.sort()

    rgb = os.listdir(rgb_path)
    rgb.sort()

    # masks = os.listdir(mask_path)
    # masks.sort()

    K = [[888.8889, 0.0, 320.0], [0.0, 1000.0, 240.0], [0.0, 0.0, 1.0]]
    K = np.array(K)
    pointcloud_list = []
    main_pose = None
    for _ in range(10):
        print(depths[_])
        print(rgb[_])
        print(poses[_])
        # if _ < 10:
            # continue
        #dpth = imageio.imread(depth_path+depths[_],pilmode="RGB")[:,:,0]
        # dpth = np.array(cv2.imread(depth_path+depths[_], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH))
        
        # dpth = np.where(dpth == np.inf, 0, dpth)
        dpth = np.load(depth_path+depths[_])
        image_rgb_concat = cv2.imread(rgb_path+rgb[_])
        image_rgb_concat = cv2.cvtColor(image_rgb_concat, cv2.COLOR_BGR2RGB)
        image_rgb_concat = image_rgb_concat / 255.0
        # camera_pose = read_json_file(poses_path+poses[_])
        pose = np.load(poses_path+poses[_])
        pose = np.vstack([pose, np.array([0, 0, 0, 1])])
        pose = np.linalg.inv(pose)
        # import pdb
        # pdb.set_trace()

        # msk = cv2.imread(mask_path+masks[_], 0)/255.0
        #print(msk.shape)
        #exit()
        #import pdb
        #pdb.set_trace()
        dpth = dpth


        #import pdb
        #pdb.set_trace()
        # pose_np     = pose_dict_to_numpy(camera_pose)
        # pose        = pose_2_matrix(torch.from_numpy(pose_np))
        
        # pose = pose.cpu().detach().numpy()
        
        
        pts, pts_colors = inverse_project_points(K, image_rgb_concat, dpth, pose)
        pcd_o3d = points_to_o3d(pts, pts_colors)
        file_name = "pcds/point_cloud_"+str(_)+".pcd"
        o3d.io.write_point_cloud(file_name, pcd_o3d)
        pointcloud_list.append(pcd_o3d)
    o3d.visualization.draw_geometries(pointcloud_list)
    vis = o3d.visualization.Visualizer()












   


    # # K = np.load(all_files["K_file"])[:3, :3]
    # K = np.array([[888.8889, 0.0, 320.0], [0,0, 1000.0, 240.0], [0.0, 0.0, 1.0]])

    # pointcloud_list = []
    # segmentation_list = []
    # for i in range(start, num_files_to_visualize, skip):

    #     image = plt.imread(all_files["image_files"][i])[:, :, :3]
    #     depth = np.load(all_files["depth_files"][i])
    #     pose = np.load(all_files["pose_files"][i])
    #     pts, pts_colors = inverse_project_points(K, image, depth, pose, near = near, far = far)
    #     pcd_o3d = points_to_o3d(pts, pts_colors)
    #     pointcloud_list.append(pcd_o3d)

    # o3d.visualization.draw_geometries(pointcloud_list)
    #     #pointcloud_list = []

if __name__=="__main__":

    # parser = argparse.ArgumentParser(description="Reconstruction from NeRF outputs")
    # parser.add_argument("--input", required=True, type = str)
    # parser.add_argument("--max_files", default=None, type = int)
    # parser.add_argument("--near", default=0.0, type = float)
    # parser.add_argument("--far", default=3.0, type = float)
    # parser.add_argument("--skip", default=1, type = int)
    # parser.add_argument("--start", default=0, type = int)


    # args = parser.parse_args()
    SFM()

