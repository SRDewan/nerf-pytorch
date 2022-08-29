import numpy as np
import open3d as o3d
import argparse
import os

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="NeRF sigma mesh visualization")
    parser.add_argument("--input", required=True, type = str)
    parser.add_argument("--num_samples", default=1000, type = int)
    parser.add_argument("--ground_truth", required=True, type = str)
    parser.add_argument("--max_files", default=10, type = int)
    parser.add_argument("--sigma_thresh", default=10.0, type = float)
    args = parser.parse_args()

    sigmas_path = os.path.join(args.input, "original_alphas_32.npy")
    samples_path = os.path.join(args.input, "samples_32.npy")
    sigmas = np.load(sigmas_path).reshape((-1, 1))
    samples = np.load(samples_path).reshape((-1, 3))
    occ = np.where(sigmas > args.sigma_thresh)[0]
    print("Total = %d, Occupied = %d, Occupancy = %f" % (len(sigmas), len(occ), len(occ) / len(sigmas)))

    thresh_pcd = o3d.geometry.PointCloud()
    thresh_points = samples[np.where(sigmas > args.sigma_thresh)[0]]
    thresh_pcd.points = o3d.utility.Vector3dVector(thresh_points)
    o3d.visualization.draw_geometries([thresh_pcd])

    mesh_path = os.path.join(args.input, "mesh_%.1f.ply" % (args.sigma_thresh))
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    o3d.visualization.draw_geometries([mesh])

    pcd = mesh.sample_points_uniformly(number_of_points=args.num_samples)
    o3d.visualization.draw_geometries([pcd])
    pts = np.array(pcd.points)
    minCorner = np.array([np.min(pts[:, 0]), np.min(pts[:, 1]), np.min(pts[:, 2])])
    maxCorner = np.array([np.max(pts[:, 0]), np.max(pts[:, 1]), np.max(pts[:, 2])])
    print(minCorner, maxCorner, (minCorner + maxCorner) / 2)

    o3d.visualization.draw_geometries([thresh_pcd, mesh])
    o3d.visualization.draw_geometries([pcd, mesh])

    trans_thresh_points = np.array(thresh_pcd.points) + np.array([0, 0, 1])
    thresh_pcd.points = o3d.utility.Vector3dVector(trans_thresh_points)
    trans_points = np.array(pcd.points) + np.array([0, 0, 2])
    pcd.points = o3d.utility.Vector3dVector(trans_points)
    o3d.visualization.draw_geometries([thresh_pcd, pcd, mesh])

    # Groundtruth load and visualization
    dir_path = args.ground_truth
    max_files = args.max_files
    
    count = 0
    pcds = []
    final_pcd = o3d.geometry.PointCloud()
    final_pcd_pts = []
    final_pcd_cols = []

    for path, dirs, files in os.walk(dir_path):
        for file in sorted(files):
            if "ply" in file:
                pcd_path = os.path.join(path, file)
                pcd = o3d.io.read_point_cloud(pcd_path)
                pcds.append(pcd)
    
                pts = np.array(pcd.points)
                final_pcd_pts.append(pts)
                # colors = np.array(pcd.colors)
                # final_pcd_cols.append(colors)
    
                # print(file)
                # o3d.visualization.draw_geometries([pcd])
    
                count += 1
                if count >= max_files:
                    break
    
    final_pcd_pts = np.concatenate(final_pcd_pts)
    minCorner = np.array([np.min(pts[:, 0]), np.min(pts[:, 1]), np.min(pts[:, 2])])
    maxCorner = np.array([np.max(pts[:, 0]), np.max(pts[:, 1]), np.max(pts[:, 2])])
    print(minCorner, maxCorner, (minCorner + maxCorner) / 2)
    # final_pcd_cols = np.concatenate(final_pcd_cols)
    final_pcd.points = o3d.utility.Vector3dVector(final_pcd_pts)
    # final_pcd.colors = o3d.utility.Vector3dVector(final_pcd_cols)
    
    o3d.visualization.draw_geometries(pcds)
    o3d.visualization.draw_geometries([final_pcd])
    