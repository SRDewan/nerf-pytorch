import numpy as np
import open3d as o3d

mesh_path = "logs/draco_32_samples_lr_0001_precrop_random_10_1000_epochs_res_64/renderonly_path_039999/mesh.ply"
mesh = o3d.io.read_triangle_mesh(mesh_path)
o3d.visualization.draw_geometries([mesh])
