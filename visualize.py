import open3d as o3d
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Reconstruction from NeRF outputs")
parser.add_argument("--input_dir", required=True, type = str)
parser.add_argument("--max_files", default=10, type = int)

args = parser.parse_args()
dir_path = args.input_dir
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
            colors = np.array(pcd.colors)
            final_pcd_cols.append(colors)

            # print(file)
            # o3d.visualization.draw_geometries([pcd])

            count += 1
            if count >= max_files:
                break

final_pcd_pts = np.concatenate(final_pcd_pts)
final_pcd_cols = np.concatenate(final_pcd_cols)
final_pcd.points = o3d.utility.Vector3dVector(final_pcd_pts)
final_pcd.colors = o3d.utility.Vector3dVector(final_pcd_cols)

o3d.visualization.draw_geometries(pcds)
# o3d.visualization.draw_geometries([final_pcd])

# o3d.io.write_point_cloud("final_pcd.ply", final_pcd)
