import open3d as o3d
import cv2
import json
import os
import numpy as np
import sys
from tqdm import tqdm
import random
from mesh_to_sdf import mesh_to_sdf, sample_sdf_near_surface
import trimesh
import math
import copy

def readData(dirPath, thresh):
    meshPath = os.path.join(dirPath, "mesh_%.1f.ply" % (thresh))
    mesh = o3d.io.read_triangle_mesh(meshPath)
    mesh.compute_vertex_normals()
    
    tmesh = trimesh.load(meshPath)
    return tmesh, mesh

def getSampleCoords(minCoord, maxCoord, sampleCtr=128):
    xs = np.linspace(minCoord[0], maxCoord[0], sampleCtr)
    ys = np.linspace(minCoord[1], maxCoord[1], sampleCtr)
    zs = np.linspace(minCoord[2], maxCoord[2], sampleCtr)

    coords = []
    for x in xs:
        for y in ys:
            for z in zs:
                coords.append(np.array([x, y, z]))

    coords = np.array(coords)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    return pcd, coords

def getCoords(minCoord, maxCoord, sampleCtr=128):
    xdists = np.linspace(minCoord[0], maxCoord[0], sampleCtr)
    ydists = np.linspace(minCoord[1], maxCoord[1], sampleCtr)
    zdists = np.linspace(minCoord[2], maxCoord[2], sampleCtr)

    # xs, ys, zs = np.meshgrid(xdists, ydists, zdists)
    # xs, ys, zs = xs.reshape((-1, 1)), ys.reshape((-1, 1)), zs.reshape((-1, 1))
    # coords = np.hstack([xs, ys, zs])
    coords = np.stack(np.meshgrid(xdists, ydists, zdists, indexing='ij'), axis=-1).astype(np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords.reshape((-1, 3)))
    return pcd, coords

def meshToPcd(mesh, numPts=10000):
    pcd = mesh.sample_points_uniformly(number_of_points=numPts)
    pts = np.array(pcd.points)
    colors = np.zeros(pts.shape)
    colors[:, :] = np.array([0, 0, 0])
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd, pts

def getSDF(mesh, coords, dims=(128, 128, 128)):
    sdf = []
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh)  # we do not need the geometry ID for mesh

    # for coord in tqdm(coords):
        # query_point = o3d.core.Tensor([coord], dtype=o3d.core.Dtype.Float32)

        # # Compute distance of the query point from the surface
        # unsigned_distance = scene.compute_distance(query_point)
        # signed_distance = scene.compute_signed_distance(query_point)
        # sdf.append(signed_distance.numpy())

    signed_distance = scene.compute_signed_distance(coords)
    sdf = signed_distance.numpy()
    # sdf = np.reshape(np.array(sdf), dims)
    return sdf

def sdfToPcd(sdf, coords):
    (xs, ys, zs) = np.where(sdf <= 0)
    inds = np.vstack([xs, ys, zs]).T

    pts = []
    for ind in inds:
        pts.append(coords[ind[0], ind[1], ind[2]])

    pts = np.array(pts)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd

def sdfToPcdIn(sdf, coords):
    sdf = sdf.reshape((-1, 1))
    coords = coords.reshape((-1, 3))
    inds = np.where(sdf < -0.01)[0]

    pts = []
    colors = []
    for ind in inds:
        pts.append(coords[ind, :])
        colors.append(np.array([1, 0, 0]))

    pts = np.array(pts)
    colors = np.array(colors)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def sdfToPcdBound(sdf, coords):
    sdf = sdf.reshape((-1, 1))
    coords = coords.reshape((-1, 3))
    inds = np.where(sdf == 0)[0]

    pts = []
    colors = []
    for ind in inds:
        pts.append(coords[ind, :])
        colors.append(np.array([1, 0, 0]))

    pts = np.array(pts)
    colors = np.array(colors)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def sdfToPcdOut(sdf, coords):
    sdf = sdf.reshape((-1, 1))
    coords = coords.reshape((-1, 3))
    inds = np.where(sdf > 0)[0]

    pts = []
    colors = []
    for ind in inds:
        pts.append(coords[ind, :])
        colors.append(np.array([1, 0, 0]))

    pts = np.array(pts)
    colors = np.array(colors)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def sdfToPcd2(sdf, coords):
    inInds = np.where(sdf < 0) 
    surfaceInds = np.where(sdf == 0)
    outInds = np.where(sdf > 0)

    colors = np.ones(coords.shape)
    colors[inInds[0], inInds[1], inInds[2], :] = np.array([1, 0, 0])
    colors[surfaceInds[0], surfaceInds[1], surfaceInds[2], :] = np.array([1, 1, 1])
    colors[outInds[0], outInds[1], outInds[2], :] = np.array([0, 0, 1])

    pts = np.array(coords.reshape((-1, 3)))
    colors = np.array(colors.reshape((-1, 3)))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def sdfToPcd3(sdf, coords):
    sdf = sdf.reshape((-1, 1))
    coords = coords.reshape((-1, 3))

    inInds = np.where(sdf < 0) 
    surfaceInds = np.where(sdf == 0)
    outInds = np.where(sdf > 0)

    colors = np.ones(coords.shape)
    colors[inInds[0], :] = np.array([1, 0, 0])
    colors[surfaceInds[0], :] = np.array([1, 1, 1])
    colors[outInds[0], :] = np.array([0, 0, 1])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def saveArr(arr, filePath):
    np.save(filePath, arr)

def getMaxCube(minCorner, maxCorner):
    # dims = maxPt - minPt
    # maxDim = np.max(dims)
    # maxInd = np.where(dims == maxDim)[0]
    minPt, maxPt = copy.deepcopy(minCorner), copy.deepcopy(maxCorner)
    diagLen = math.dist(minPt, maxPt)

    for i in range(len(minPt)):
        # if i == maxInd:
            # continue

        midPt = (minPt[i] + maxPt[i]) / 2
        minPt[i] = midPt - diagLen / 2
        maxPt[i] = midPt + diagLen / 2

    return minPt, maxPt

def sdfFilter(sdfPts, sdf, minCorner, maxCorner):
    inds = np.where(sdf < 0)[0]
    for ind in inds:
        pt = sdfPts[ind]
        outside = False

        for i in range(len(pt)):
            if pt[i] < minCorner[i] or pt[i] > maxCorner[i]:
                outside = True
                break

        if outside:
            sdf[ind] = 1

    sdf = np.where(sdf > 0, 1, sdf)
    return sdf

def translateObj(pts, minCorner, maxCorner):
    mids = (minCorner + maxCorner) / 2
    pts = pts - mids
    return pts

loadPath = ""
if len(sys.argv) > 1:
    loadPath = sys.argv[1]

thresh = 10 
if len(sys.argv) > 2:
    thresh = int(sys.argv[2])

numSamples = 128 
if len(sys.argv) > 3:
    numSamples = int(sys.argv[3])

saveDir = "sdfs"
if len(sys.argv) > 4:
    saveDir = sys.argv[4]

if not os.path.exists(saveDir):
    os.system("mkdir %s" % (saveDir))

selPaths = [loadPath]
for dirPath in tqdm(selPaths):
    parts = dirPath.split("/")
    print("Currently processing %s" % (parts[-2]))

    tmesh, mesh = readData(dirPath, thresh)
    # print(meta["min"], meta["max"])
    # boxPcd, coords = getCoords(meta["min"], meta["max"])
    # minPt, maxPt = [-1, -1, -1], [1, 1, 1]
    # boxPcd, coords = getCoords(minPt, maxPt, numSamples)
    # o3d.visualization.draw_geometries([mesh])
    # o3d.visualization.draw_geometries([mesh, boxPcd])

    # sdfPts = coords.reshape((-1, 3))
    # sdf = mesh_to_sdf(tmesh, sdfPts, surface_point_method='scan', sign_method='normal')
    # sdf = getSDF(mesh, coords, (numSamples, numSamples, numSamples))
    # print("SDF Min = %f, SDF Max = %f" % (np.min(sdf), np.max(sdf)))
    # sdfPcd = sdfToPcd3(sdf, sdfPts)
    # sdfPcdIn = sdfToPcdIn(sdf, sdfPts)
    # sdfPcdOut = sdfToPcdOut(sdf, sdfPts)

    # o3d.visualization.draw_geometries([sdfPcd])
    # o3d.visualization.draw_geometries([sdfPcdIn])
    # o3d.visualization.draw_geometries([sdfPcdOut])

    pcd, pts = meshToPcd(mesh)
    # o3d.visualization.draw_geometries([pcd])
    minCorner = np.array([np.min(pts[:, 0]), np.min(pts[:, 1]), np.min(pts[:, 2])])
    maxCorner = np.array([np.max(pts[:, 0]), np.max(pts[:, 1]), np.max(pts[:, 2])])
    minPt, maxPt = getMaxCube(minCorner, maxCorner)
    # print("PCD Min = ", minCorner, ", PCD Max = ", maxCorner)
    # print("PCD Mid = ", (minCorner + maxCorner) / 2)
    # print("Box Min = ", minPt, ", Box Max = ", maxPt)

    # boxPcd, coords = getCoords(minCorner, maxCorner, numSamples)
    # o3d.visualization.draw_geometries([mesh, boxPcd])
    
    # # sdfPts, sdf = sample_sdf_near_surface(tmesh, number_of_points = numSamples ** 3)
    # # sdf = getSDF(mesh, coords, (numSamples, numSamples, numSamples))
    # sdfPts = coords.reshape((-1, 3))
    # sdf = mesh_to_sdf(tmesh, sdfPts, surface_point_method='scan', sign_method='normal')
    # sdf = sdfFilter(sdfPts, sdf, minCorner, maxCorner)
    # # sdf = getSDF(mesh, coords, (numSamples, numSamples, numSamples))
    # # print("SDF Min = %f, SDF Max = %f" % (np.min(sdf), np.max(sdf)))
    # sdfPcd = sdfToPcd3(sdf, sdfPts)
    # sdfPcdIn = sdfToPcdIn(sdf, sdfPts)
    # sdfPcdOut = sdfToPcdOut(sdf, sdfPts)

    # o3d.visualization.draw_geometries([sdfPcd])
    # o3d.visualization.draw_geometries([sdfPcdIn])
    # o3d.visualization.draw_geometries([sdfPcdOut])
    # print("Number of inside points = %d, Number of outside points = %d" % (len(sdfPcdIn.points), len(sdfPcdOut.points)))

    boxPcd, coords = getCoords(minPt, maxPt, numSamples)
    # o3d.visualization.draw_geometries([mesh, boxPcd])
    
    # sdfPts, sdf = sample_sdf_near_surface(tmesh, number_of_points = numSamples ** 3)
    # sdf = getSDF(mesh, coords, (numSamples, numSamples, numSamples))
    sdfPts = coords.reshape((-1, 3))
    sdf = mesh_to_sdf(tmesh, sdfPts, surface_point_method='scan', sign_method='normal')
    sdf = sdfFilter(sdfPts, sdf, minCorner, maxCorner)
    # sdf = getSDF(mesh, coords, (numSamples, numSamples, numSamples))
    # print("SDF Min = %f, SDF Max = %f" % (np.min(sdf), np.max(sdf)))
    # sdfPcd = sdfToPcd3(sdf, sdfPts)
    # sdfPcdIn = sdfToPcdIn(sdf, sdfPts)
    # sdfPcdOut = sdfToPcdOut(sdf, sdfPts)

    # o3d.visualization.draw_geometries([sdfPcd])
    # o3d.visualization.draw_geometries([sdfPcdIn])
    # o3d.visualization.draw_geometries([sdfPcdOut])
    # print("Number of inside points = %d, Number of outside points = %d, Occupancy = %f" % (len(sdfPcdIn.points), len(sdfPcdOut.points), len(sdfPcdIn.points) / len(sdfPcdOut.points)))
    # o3d.visualization.draw_geometries([pcd, sdfPcd])
    # o3d.visualization.draw_geometries([pcd, sdfPcdIn])
    # o3d.visualization.draw_geometries([pcd, sdfPcdBound])
    # o3d.visualization.draw_geometries([pcd, sdfPcdOut])

    sdfPts = translateObj(sdfPts, minPt, maxPt)
    minCorner = np.array([np.min(sdfPts[:, 0]), np.min(sdfPts[:, 1]), np.min(sdfPts[:, 2])])
    maxCorner = np.array([np.max(sdfPts[:, 0]), np.max(sdfPts[:, 1]), np.max(sdfPts[:, 2])])
    # print("PCD Min = ", minCorner, ", PCD Max = ", maxCorner)
    # print("PCD Mid = ", (minCorner + maxCorner) / 2)

    # sdfPcd = sdfToPcd3(sdf, sdfPts)
    sdfPcdIn = sdfToPcdIn(sdf, sdfPts)
    # sdfPcdOut = sdfToPcdOut(sdf, sdfPts)
    
    # o3d.visualization.draw_geometries([sdfPcd])
    o3d.visualization.draw_geometries([sdfPcdIn])
    # o3d.visualization.draw_geometries([sdfPcdOut])

    sdf = sdf.reshape((numSamples, numSamples, numSamples))
    sdfPts = sdfPts.reshape((numSamples, numSamples, numSamples, 3))
    sdfPath = os.path.join(saveDir, "sigmas_%.1f_%d_sdf.npy" % (thresh, numSamples))
    ptsPath = os.path.join(saveDir, "sigmas_%.1f_%d_pts.npy" % (thresh, numSamples))
    # saveArr(sdf, sdfPath)
    # saveArr(sdfPts, ptsPath)
