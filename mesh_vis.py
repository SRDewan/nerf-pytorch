import numpy as np
import open3d as o3d
import argparse
import os
from sklearn.cluster import KMeans

def labels_to_pallette(mask, tensor = False):
    classes = {
            0: [255, 255, 255], # White
            1: [255, 0, 0],     # red
            2: [0, 255, 0],     # green
            3: [0, 0, 255],     # blue
            4: [255, 0, 255],   # pink
            5: [255, 255, 0],   # yellow
            6: [153, 51, 102]   # magenta
            }

    result = np.zeros((mask.shape[0], 3))

    if tensor:
        mask = mask.cpu().numpy()

    for key, value in classes.items():
        result[np.where(mask == key)[0]] = value

    if tensor:
        result = Image.fromarray(result.astype('uint8'), 'RGB')
        result = T.ToTensor()(result)

    return result

def cluster_sigmas(sigmas, n_clusters=2, power=1.0, exp=False, scale=1.0):
    print("Number of clusters = ", n_clusters)
    # dim, _, _ = sigmas.shape
    # sigmas = sigmas.reshape((-1, 1))
    # sigmas = sigmas + 1e2

    print("Sigmas range = ", np.min(sigmas), np.max(sigmas))
    relu_sigmas = np.where(sigmas > 0, sigmas, 0)
    powered_sigmas = relu_sigmas ** power
    print("Sigmas powered range = ", np.min(powered_sigmas), np.max(powered_sigmas))
    if exp:
        sigmas = 1. - np.exp(-scale * powered_sigmas)
    print("Sigmas final range = ", np.min(sigmas), np.max(sigmas))

    # model = GaussianMixture(n_components=2,init_params="k-means++",weights_init=[0.9,0.1])
    model = KMeans(init="k-means++", n_clusters=n_clusters)
    model.fit(sigmas)
    print("Cluster centers = ", model.cluster_centers_)

    labels = model.predict(sigmas)
    (clusters, counts) = np.unique(labels, return_counts=True)
    bg_label = clusters[np.where(counts == counts.max())[0]]
    clustered_sigmas = np.where(labels == bg_label, 0, 1)
    return clustered_sigmas
# .reshape((dim, dim, dim))

def cluster_points(points, ref_point):
    tol = 5
    prev_inertia = -1
    for n_clusters in range(1, 11):
        print("Number of clusters = ", n_clusters)

        model = KMeans(init="k-means++", n_clusters=n_clusters)
        model.fit(points)
        print("Cluster centers = ", model.cluster_centers_)
        inertia = model.inertia_

        labels = model.predict(sigmas)
        (clusters, counts) = np.unique(labels, return_counts=True)
        bg_label = clusters[np.where(counts == counts.max())[0]]
        clustered_sigmas = np.where(labels == bg_label, 0, 1)

        labels = model.predict(sigmas)
    return clustered_sigmas

def visualize(sigmas_path, samples_path, sigma_thresh, semantics_path=None):
    sigmas = np.load(sigmas_path).reshape((-1, 1))
    samples = np.load(samples_path).reshape((-1, 3))

    # occ = np.where(sigmas > sigma_thresh)[0]
    # print("Thresholding with %f: Total = %d, Occupied = %d, Occupancy = %f" % (sigma_thresh, len(sigmas), len(occ), len(occ) / len(sigmas)))

    # thresh_pcd = o3d.geometry.PointCloud()
    # thresh_points = samples[np.where(sigmas > sigma_thresh)[0]]
    # thresh_pcd.points = o3d.utility.Vector3dVector(thresh_points)
    # o3d.visualization.draw_geometries([thresh_pcd])

    clustered_sigmas = cluster_sigmas(sigmas, 2, 2.0, True, 0.3109375)
    occ = np.where(clustered_sigmas != 0)[0]
    print("Clustering: Total = %d, Occupied = %d, Occupancy = %f" % (len(sigmas), len(occ), len(occ) / len(sigmas)))

    thresh_pcd = o3d.geometry.PointCloud()
    thresh_points = samples[np.where(clustered_sigmas != 0)[0]]
    # clustered_points = cluster_points(thresh_points, samples[len(samples) // 2])
    thresh_pcd.points = o3d.utility.Vector3dVector(thresh_points)
    o3d.visualization.draw_geometries([thresh_pcd])

    if semantics_path:
        semantics = np.load(semantics_path).reshape((-1, 1))
        
        thresh_pcd = o3d.geometry.PointCloud()
        thresh_points = samples[np.where(np.logical_and(
            sigmas.reshape(-1, 1) > sigma_thresh,
            # semantics != 0,
            semantics != 0))[0]]
        thresh_pcd.points = o3d.utility.Vector3dVector(thresh_points)
        o3d.visualization.draw_geometries([thresh_pcd])

        colors = labels_to_pallette(semantics)[np.where(np.logical_and(
            sigmas.reshape(-1, 1) > sigma_thresh,
            # semantics != 0,
            semantics != 0))[0], :]
        print(colors.shape, thresh_points.shape)
        thresh_pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([thresh_pcd])

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="NeRF sigma mesh visualization")
    parser.add_argument("--multi_scene", action='store_true')
    parser.add_argument("--input", required=True, type = str)
    parser.add_argument("--res", default=32, type = int)
    parser.add_argument("--sigma_thresh", default=10.0, type = float)
    parser.add_argument("--semantic_en", action='store_true')
    parser.add_argument("--class_id", default=1, type = int)
    # parser.add_argument("--num_samples", default=1000, type = int)
    # parser.add_argument("--ground_truth", required=True, type = str)
    # parser.add_argument("--max_files", default=10, type = int)
    args = parser.parse_args()

    if args.multi_scene:
        for path, dirs, files in os.walk(args.input):
            for file in files:
                if "sigmas_%d.npy" % (args.res) != file:
                    continue

                print("Processing %s %s" % (path, file))
                sigmas_path = os.path.join(path, file)
                samples_path = os.path.join(path, file.replace("sigmas", "samples"))
                semantics_path = None
                if args.semantic_en:
                    semantics_path = os.path.join(path, file.replace("sigmas", "semantics"))
                visualize(sigmas_path, samples_path, args.sigma_thresh, semantics_path)


    sigmas_path = os.path.join(args.input, "sigmas_%d.npy" % (args.res))
    samples_path = os.path.join(args.input, "samples_%d.npy" % (args.res))
    semantics_path = None
    if args.semantic_en:
        semantics_path = os.path.join(args.input, "semantics_%d.npy" % (args.res))
    visualize(sigmas_path, samples_path, args.sigma_thresh, semantics_path)


    if args.semantic_en:
        sigmas_path = os.path.join(args.input, "class%d_sigmas_32.npy" % (args.class_id))
        samples_path = os.path.join(args.input, "class%d_samples_32.npy" % (args.class_id))
        visualize(sigmas_path, samples_path, args.sigma_thresh, semantics_path)

    # mesh_path = os.path.join(args.input, "mesh_%.1f.ply" % (args.sigma_thresh))
    # mesh = o3d.io.read_triangle_mesh(mesh_path)
    # o3d.visualization.draw_geometries([mesh])

    # pcd = mesh.sample_points_uniformly(number_of_points=args.num_samples)
    # o3d.visualization.draw_geometries([pcd])
    # pts = np.array(pcd.points)
    # minCorner = np.array([np.min(pts[:, 0]), np.min(pts[:, 1]), np.min(pts[:, 2])])
    # maxCorner = np.array([np.max(pts[:, 0]), np.max(pts[:, 1]), np.max(pts[:, 2])])
    # print(minCorner, maxCorner, (minCorner + maxCorner) / 2)

    # o3d.visualization.draw_geometries([thresh_pcd, mesh])
    # o3d.visualization.draw_geometries([pcd, mesh])

    # trans_thresh_points = np.array(thresh_pcd.points) + np.array([0, 0, 1])
    # thresh_pcd.points = o3d.utility.Vector3dVector(trans_thresh_points)
    # trans_points = np.array(pcd.points) + np.array([0, 0, 2])
    # pcd.points = o3d.utility.Vector3dVector(trans_points)
    # o3d.visualization.draw_geometries([thresh_pcd, pcd, mesh])

    # # Groundtruth load and visualization
    # dir_path = args.ground_truth
    # max_files = args.max_files
    
    # count = 0
    # pcds = []
    # final_pcd = o3d.geometry.PointCloud()
    # final_pcd_pts = []
    # final_pcd_cols = []

    # for path, dirs, files in os.walk(dir_path):
        # for file in sorted(files):
            # if "ply" in file:
                # pcd_path = os.path.join(path, file)
                # pcd = o3d.io.read_point_cloud(pcd_path)
                # pcds.append(pcd)
    
                # pts = np.array(pcd.points)
                # final_pcd_pts.append(pts)
                # # colors = np.array(pcd.colors)
                # # final_pcd_cols.append(colors)
    
                # # print(file)
                # # o3d.visualization.draw_geometries([pcd])
    
                # count += 1
                # if count >= max_files:
                    # break
    
    # final_pcd_pts = np.concatenate(final_pcd_pts)
    # minCorner = np.array([np.min(pts[:, 0]), np.min(pts[:, 1]), np.min(pts[:, 2])])
    # maxCorner = np.array([np.max(pts[:, 0]), np.max(pts[:, 1]), np.max(pts[:, 2])])
    # print(minCorner, maxCorner, (minCorner + maxCorner) / 2)
    # # final_pcd_cols = np.concatenate(final_pcd_cols)
    # final_pcd.points = o3d.utility.Vector3dVector(final_pcd_pts)
    # # final_pcd.colors = o3d.utility.Vector3dVector(final_pcd_cols)
    
    # o3d.visualization.draw_geometries(pcds)
    # o3d.visualization.draw_geometries([final_pcd])
    
