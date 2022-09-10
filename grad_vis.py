import sys
sys.path.append('../')
import open3d as o3d
import seaborn as sns
import numpy as np
from scipy.spatial.transform import Rotation as Rotation
import torch
from scipy.spatial import distance
import torch.nn
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import gaussian_filter
import argparse
import os
import copy


def get_gradient_density(x):
    
    # x - B, H, W, D
    
    B, H, W, D = x.shape
    x_grad = torch.zeros(B,H,W,D)
    y_grad = torch.zeros(B,H,W,D)
    z_grad = torch.zeros(B,H,W,D)
    
    data = x
    # .detach().numpy()
    
    # x_grad = torch.from_numpy(np.gradient(data,axis=3,edge_order=2))
    # y_grad = torch.from_numpy(np.gradient(data,axis=2,edge_order=2))
    # z_grad = torch.from_numpy(np.gradient(data,axis=1,edge_order=2))
    x_grad = torch.gradient(data, dim=3, edge_order=2)[0]
    y_grad = torch.gradient(data, dim=2, edge_order=2)[0]
    z_grad = torch.gradient(data, dim=1, edge_order=2)[0]
    
   
    gradient = torch.stack([x_grad, y_grad, z_grad], 1)

    return gradient # B, 3, H, W, D
    


def rotate_density(rotation, density_field, affine = True):
    """
    rotation - B, 3, 3
    density_field - B, H, W, D or B, C, H, W, D
    """
    if len(density_field.shape) == 4:
        out = density_field.unsqueeze(1)
    else:
        out = density_field

    rotation = rotation.type_as(density_field)
    t = torch.tensor([0, 0, 0]).unsqueeze(0).unsqueeze(2).repeat(rotation.shape[0], 1, 1).type_as(density_field)
    theta = torch.cat([rotation, t], dim = -1)
    if affine == True:
        rot_grid = torch.nn.functional.affine_grid(theta, out.shape, align_corners = True)
    else:
        x = torch.linspace(-1, 1, density_field.shape[2]).type_as(density_field)
        grid = torch.stack(torch.meshgrid(x, x, x), axis = -1).unsqueeze(0).repeat(out.shape[0], 1, 1, 1, 1) 
        # print(grid.shape, rotation.shape)

        rot_grid = torch.einsum("bij, bhwdj-> bhwdi", rotation, grid)
    #print(rot_grid)
    rotated_grid = torch.nn.functional.grid_sample(out, rot_grid, align_corners = True, mode="nearest")#, padding_mode = "border")

    if len(density_field.shape) == 4:
        rotated_grid = rotated_grid.squeeze(1)
        
    return rotated_grid


def draw_oriented_pointcloud(x, n, t=1.0):
    a = x
    b = x + t*n
    points = []
    lines = []
    for i in range(a.shape[0]):
        ai = [a[i, 0], a[i, 1], a[i, 2]]
        bi = [b[i, 0], b[i, 1], b[i, 2]]
        points.append(ai)
        points.append(bi)
        lines.append([2*i, 2*i+1])
    colors = [[1, 0, 0] for i in range(len(lines))]
    
    pcd = o3d.geometry.PointCloud()
    point_colors = np.ones(x.shape)
    pcd.points = o3d.utility.Vector3dVector(a)
    pcd.colors = o3d.utility.Vector3dVector(point_colors)

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.6, origin=[-1, -1, -1])
    o3d.visualization.draw_geometries([line_set, pcd,mesh_frame])
    return line_set, pcd


def create_color_samples(N):
    '''
    Creates N distinct colors
    N x 3 output
    '''

    palette = sns.color_palette(None, N)
    palette = np.array(palette)

    return palette
    

parser = argparse.ArgumentParser(description="NeRF sigma gradient visualization")
parser.add_argument("--input_dir", required=True, type = str)
parser.add_argument("--sdf_file_name", required=True, type = str)
parser.add_argument("--coords_file_name", required=True, type = str)
parser.add_argument("--scale", default=0.01, type = float)
args = parser.parse_args()

sdfPath = os.path.join(args.input_dir, args.sdf_file_name) 
coords_path = os.path.join(args.input_dir, args.coords_file_name) 



density = np.load(sdfPath,allow_pickle=False)
# density = density.transpose(2,1,0)
# density = gaussian_filter(density, sigma=1)
grads = copy.deepcopy(density)
density = torch.from_numpy(density)

coords = np.load(coords_path,allow_pickle=False)
coords = coords.transpose(2,1,0,3)
coords_ = coords.copy()
coords = coords.reshape(-1,3)
scale_ = args.scale 




# grad = get_gradient_density(density.unsqueeze(0)).squeeze(0)
grad = copy.deepcopy(grads)
grad = grad.reshape(-1,3)
# grad = grad.detach().numpy()

draw_oriented_pointcloud(coords,grad,scale_)



rot_mat_ = Rotation.from_euler('zyz', [90,0,0], degrees=True)
rot_mat = torch.from_numpy(rot_mat_.as_matrix()).unsqueeze(0)
rot_mat_np = rot_mat_.as_matrix()
print(rot_mat)

# gradients_ = get_gradient_density(density.unsqueeze(0))
gradients_ = density
rotated_gradients = gradients_.squeeze(0).permute(1,2,3,0).reshape(-1,3).detach().numpy()
rotated_gradients = np.matmul(rot_mat_np,rotated_gradients.T).T
l1, p1 = draw_oriented_pointcloud(coords,rotated_gradients,scale_)


rot_density = rotate_density(torch.inverse(rot_mat), density.unsqueeze(0))   # Rotate the signal
grad_on_rotated_density = get_gradient_density(rot_density)                  # Take the gradient on rotated signal
vis_grad_on_rotated_density = grad_on_rotated_density.squeeze(0).permute(1,2,3,0).reshape(-1,3).detach().numpy()  # For visualization purpose convert to numy array
draw_oriented_pointcloud(coords,vis_grad_on_rotated_density,scale_)   


grad_on_rotated_density = rotate_density(rot_mat, grad_on_rotated_density).squeeze(0)   # Re arrange the signal back to original postion
grad_on_rotated_density = grad_on_rotated_density.permute(1,2,3,0).reshape(-1,3).detach().numpy()
l2, p2 = draw_oriented_pointcloud(coords,grad_on_rotated_density,scale_)# Visualize the Gradient field
o3d.visualization.draw_geometries([l1, l2, p1, p2])
   





rotated_gradients = torch.from_numpy(rotated_gradients).to(torch.float32)
grad_on_rotated_density = torch.from_numpy(grad_on_rotated_density).to(torch.float32)





diff_ = (rotated_gradients - grad_on_rotated_density).detach().numpy()
draw_oriented_pointcloud(coords,diff_,scale_)

match = 0
un_match = 0
non_zero_features = 0
for i in range(rotated_gradients.shape[0]):
    if torch.count_nonzero(rotated_gradients[i]) > 0:
        non_zero_features = non_zero_features + 1
    else:
        continue
    if  torch.allclose(rotated_gradients[i],grad_on_rotated_density[i],atol=1e-6):
      match = match +1 
    else:
      import pdb
      pdb.set_trace()
      un_match = un_match + 1
    
print("matched features for the type  = ",match)
print("unmatched features for the type  = ",un_match)
print(match - un_match)


cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
sim = cos(rotated_gradients, grad_on_rotated_density)

match = 0
un_match = 0
non_zero_features = 0
for i in range(sim.shape[0]):
    if torch.count_nonzero(sim[i]) > 0:
        non_zero_features = non_zero_features + 1
    else:
        continue
    if sim[i] >= 0.8 :
      match = match +1 
    else:
      
      un_match = un_match + 1
    
print("matched features for the type  = ",match)
print("unmatched features for the type  = ",un_match)
print(match - un_match)


