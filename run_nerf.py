import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

import matplotlib.pyplot as plt

from run_nerf_helpers import *

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_local_blender import load_local_blender_data
from load_LINEMOD import load_LINEMOD_data
from load_draco import load_draco_data

import open3d as o3d
import wandb
import gc
import copy

import cv2
from PIL import Image
import mcubes
from plyfile import PlyData, PlyElement
import math
from sklearn.cluster import KMeans

device_idx = 0
gc.collect()
torch.cuda.empty_cache()
device = torch.device("cuda:%d" % (device_idx) if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


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
    
    result = np.zeros((mask.shape[0], mask.shape[1], 3))
    
    if tensor:
        mask = mask.cpu().numpy()
    
    for key, value in classes.items():
        result[np.where(mask == key)] = value
    
    if tensor:
        result = Image.fromarray(result.astype('uint8'), 'RGB')
        result = T.ToTensor()(result)
    
    return result


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None, gt_image=None, gt_depth=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        c2w = torch.tensor(c2w).to(device)
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    if gt_depth is not None:
        gt_depth = torch.tensor(gt_depth).to(device).reshape((-1, 1))
        points = rays_o + gt_depth * rays_d

        all_ret = {}
        all_ret['rgb_map'] = torch.tensor(gt_image).to(device)
        all_ret['disp_map'] = torch.tensor([]).to(device)
        all_ret['acc_map'] = torch.tensor([]).to(device)
        all_ret['weights'] = torch.tensor([]).to(device)
        all_ret['sigma_map'] = torch.tensor([]).to(device)
        all_ret['sample_points'] = torch.tensor([]).to(device)
        all_ret['depth_map'] = gt_depth
        all_ret['points'] = points
        all_ret['semantic_map'] = torch.tensor([]).to(device)

    else:
        all_ret = batchify_rays(rays, chunk, **kwargs)
        for k in all_ret:
            k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_sh)

    all_ret['K'] = K
    all_ret['c2w'] = c2w
    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0, gt_depths=None):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []
    depths = []
    pcds = []
    Ks = []
    c2ws = []
    weights = []
    sigmas = []
    sample_points = []
    semantics = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        if gt_depths is not None:
            rgb, disp, acc, extras = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], gt_image=gt_imgs[i], gt_depth=gt_depths[i], **render_kwargs)
        else:
            rgb, disp, acc, extras = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())

        if render_kwargs['retdepth']:
            weights.append(extras['weights'].cpu().numpy())
            sigmas.append(extras['sigma_map'].cpu().numpy())
            sample_points.append(extras['sample_points'].cpu().numpy())
            depths.append(extras['depth_map'].cpu().numpy())

            points = extras['points'].cpu().numpy().reshape(-1, 3)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(rgbs[-1].reshape(-1, 3))
            pcds.append(pcd)

            Ks.append(extras['K'])
            c2ws.append(extras['c2w'].cpu().numpy())
        if render_kwargs['semantic_en']:
            semantics.append(extras['semantic_map'].cpu().numpy())

        if i==0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

            if render_kwargs['retdepth']:
                # weight8 = weights[-1]
                # weights_filename = os.path.join(savedir, 'weights_{:03d}.npy'.format(i))
                # np.save(weights_filename, weight8)
                # sigma8 = sigmas[-1]
                # sigmas_filename = os.path.join(savedir, 'sigmas_{:03d}.npy'.format(i))
                # np.save(sigmas_filename, sigma8)
                # sample8 = sample_points[-1]
                # samples_filename = os.path.join(savedir, 'samples_{:03d}.npy'.format(i))
                # np.save(samples_filename, sample8)

                depth8 = depths[-1]
                depth_filename = os.path.join(savedir, 'depth_{:03d}.npy'.format(i))
                np.save(depth_filename, depth8)
                pcd_filename = os.path.join(savedir, '{:03d}.ply'.format(i))
                o3d.io.write_point_cloud(pcd_filename, pcds[-1])

                c2w8 = c2ws[-1]
                c2w_filename = os.path.join(savedir, 'c2w_{:03d}.npy'.format(i))
                np.save(c2w_filename, c2w8)
                K8 = Ks[-1]
                K_filename = os.path.join(savedir, 'K_{:03d}.npy'.format(i))
                np.save(K_filename, K8)

            if render_kwargs['semantic_en']:
                semantic8 = semantics[-1]
                semantic_filename = os.path.join(savedir, 'semantic_{:03d}.npy'.format(i))
                np.save(semantic_filename, semantic8)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)
    if render_kwargs['retdepth']:
        depths = np.stack(depths, 0)

    return rgbs, disps, depths


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs,
                 semantic_en=args.semantic_en, num_classes=args.num_classes).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs,
                          semantic_en=args.semantic_en, num_classes=args.num_classes).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
        'retdepth': False,
        'semantic_en': args.semantic_en,
        'num_classes': args.num_classes,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.
    render_kwargs_test['retdepth'] = True
    # render_kwargs_test['gt_register'] = args.gt_register

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)
    sigma_map = raw[..., 3]

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    if raw.shape[-1] > 4:
        semantic = raw[..., 4:]
        semantic_map = torch.sum(weights[...,None] * semantic, -2)  # [N_rays, 3]
        return rgb_map, disp_map, acc_map, weights, depth_map, sigma_map, semantic_map

    return rgb_map, disp_map, acc_map, weights, depth_map, sigma_map


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                retdepth=True,
                semantic_en=False,
                num_classes=2,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    near = near.to(device)
    far = far.to(device)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    rays_o = rays_o.to(device)
    rays_d = rays_d.to(device)
    viewdirs = viewdirs.to(device)
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]


#     raw = run_network(pts)
    raw = network_query_fn(pts, viewdirs, network_fn)
    if semantic_en:
        rgb_map, disp_map, acc_map, weights, depth_map, sigma_map, semantic_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)
    else:
        rgb_map, disp_map, acc_map, weights, depth_map, sigma_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)
    points = rays_o + depth_map.unsqueeze(1) * rays_d

    if N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0, depth_map_0, points_0 = rgb_map, disp_map, acc_map, depth_map, points

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
#         raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, viewdirs, run_fn)

        if semantic_en:
            semantic_map_0 = semantic_map
            rgb_map, disp_map, acc_map, weights, depth_map, sigma_map, semantic_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)
        else:
            rgb_map, disp_map, acc_map, weights, depth_map, sigma_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)
        points = rays_o + depth_map.unsqueeze(1) * rays_d

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
    if retraw:
        ret['raw'] = raw
    if retdepth:
        ret['weights'] = weights 
        ret['sigma_map'] = sigma_map
        ret['sample_points'] = pts
        ret['depth_map'] = depth_map
        ret['points'] = points
    if semantic_en:
        ret['semantic_map'] = semantic_map

    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

        if retdepth:
            ret['depth0'] = depth_map_0
            ret['points0'] = points_0
        if semantic_en:
            ret['semantic0'] = semantic_map_0

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--semantic_en", action='store_true', 
                        help='predict a semantic map in addition to regular NeRF outputs')
    parser.add_argument("--num_classes", type=int, default=2, 
                        help='number of semantic classes')

    # loss weights
    parser.add_argument("--rgb_wt", type=float, default=1,
                        help='rgb loss weight')
    parser.add_argument("--semantic_wt", type=float, default=0,
                        help='semantic loss weight')
    parser.add_argument("--rays_sparsity_wt", type=float, default=0,
                        help='rays sparsity loss weight')
    parser.add_argument("--rays_sparsity_scale", type=float, default=0,
                        help='rays sparsity loss hyperparameter')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--near", type=float, default=0.,
                        help='closest point to sample during ray rendering')
    parser.add_argument("--far", type=float, default=1.,
                        help='farthest point to sample during ray rendering')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--gt_register", action='store_true', 
                        help='groundtruth data registration')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 
    parser.add_argument("--iters", type=int, default=10000,
                        help='number of steps to train for')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='blender', 
                        help='options: llff / blender / local_blender / deepvoxels / draco')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')
    parser.add_argument("--max_ind", type=int, default=100,
                        help='max index used in loader')


    # sigma mesh flags
    parser.add_argument('--x_range', nargs="+", type=float, default=[-1.0, 1.0],
                        help='x range of the object')
    parser.add_argument('--y_range', nargs="+", type=float, default=[-1.0, 1.0],
                        help='x range of the object')
    parser.add_argument('--z_range', nargs="+", type=float, default=[-1.0, 1.0],
                        help='x range of the object')
    parser.add_argument('--sigma_threshold', type=float, default=20.0,
                        help='threshold to consider a location is occupied')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--res", type=float, default=1.0,
                        help='load blender synthetic data at given resolution instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--wand_en", action='store_true',  
                        help='wandb logging enabled')
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=100, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='frequency of render_poses video saving')

    return parser


def get_max_cube(minCorner, maxCorner):
    minPt, maxPt = copy.deepcopy(minCorner), copy.deepcopy(maxCorner)
    diagLen = math.dist(minPt, maxPt)

    for i in range(len(minPt)):
        midPt = (minPt[i] + maxPt[i]) / 2
        minPt[i] = midPt - diagLen / 2
        maxPt[i] = midPt + diagLen / 2

    return minPt, maxPt


def get_coords(minCoord, maxCoord, sampleCtr=128):
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


def cluster(sigmas, n_clusters=2):
    print(n_clusters)
    dim, _, _ = sigmas.shape
    sigmas = sigmas.reshape((-1, 1))
    sigmas = sigmas + 1e2
  
    # model = GaussianMixture(n_components=2,init_params="k-means++",weights_init=[0.9,0.1])
    model = KMeans(init="k-means++", n_clusters=n_clusters)
    
    model.fit(sigmas)
    labels = model.predict(sigmas)
    (clusters, counts) = np.unique(labels, return_counts=True)
    fg_label = clusters[np.where(counts == counts.min())[0]]
    clustered_sigmas = np.where(labels == fg_label, 1, 0)
    return clustered_sigmas.reshape((dim, dim, dim))


def plot_sigmas(sigmas, save_path, plot_file_name):
    print(plot_file_name)
    sigma_hist_vals = sigmas.astype(int).reshape(-1)
    plt.figure()
    plt.hist(sigma_hist_vals)
    plt.show()

    fig_file_path = os.path.join(save_path, plot_file_name)
    plt.savefig(fig_file_path)


def translate_obj(pts, minCorner, maxCorner):
    mids = (minCorner + maxCorner) / 2
    pts = pts - mids
    return pts


def probs_to_semantic_3d(probs, N):
    semantic_pred = torch.nn.Softmax(dim=2)(probs).max(dim=2).indices
    semantic_pred = semantic_pred.detach().cpu().numpy().reshape((N, N, N))
    return semantic_pred


def extract_mesh(N_samples, x_range, y_range, z_range, sigma_threshold, network_query_fn, network_fn, min_b, max_b, save_path, kwargs, use_vertex_normal = True, near_t = 1.0):

    # define the dense grid for query
    N = N_samples
    xmin, xmax = x_range
    ymin, ymax = y_range
    zmin, zmax = z_range
    # assert xmax-xmin == ymax-ymin == zmax-zmin, 'the ranges must have the same length!'
    x = np.linspace(xmin, xmax, N)
    y = np.linspace(ymin, ymax, N)
    z = np.linspace(zmin, zmax, N)

    xyz_ = torch.FloatTensor(np.stack(np.meshgrid(x, y, z), -1).reshape(N ** 2, N, 3)).cuda()
    dir_ = torch.zeros(N ** 2, 3).cuda()
           # sigma is independent of direction, so any value here will produce the same result

    # predict sigma (occupancy) for each grid location
    print('Predicting occupancy ...')
    raw = network_query_fn(xyz_, dir_, network_fn)
    sigma = raw[..., 3].cpu().numpy().reshape((N, N, N))
    plot_sigmas(sigma, save_path, 'original_sigmas.png')
    sigmas_filename = os.path.join(save_path, 'original_sigmas_%d.npy' % (N))
    np.save(sigmas_filename, sigma)

    if kwargs['semantic_en']:
        semantic_map = probs_to_semantic_3d(raw[..., 4:], N)
        plot_sigmas(semantic_map, save_path, 'original_semantics.png')
        semantics_filename = os.path.join(save_path, 'original_semantics_%d.npy' % (N))
        np.save(semantics_filename, semantic_map)

    raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)
    z_vals = torch.Tensor(z).to(device).unsqueeze(0).repeat(N * N, 1)
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]
    alpha = raw2alpha(raw[...,3], dists)  # [N_rays, N_samples]
    weights = (alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]).cpu().numpy()
    alpha = alpha.cpu().numpy()

    plot_sigmas(alpha.reshape((N, N, N)), save_path, 'original_alphas.png')
    plot_sigmas(weights.reshape((N, N, N)), save_path, 'original_weights.png')
    alphas_filename = os.path.join(save_path, 'original_alphas_%d.npy' % (N))
    np.save(alphas_filename, alpha)
    weights_filename = os.path.join(save_path, 'original_weights_%d.npy' % (N))
    np.save(weights_filename, weights)

    clustered_sigma = cluster(sigma, 2)
    plot_sigmas(clustered_sigma, save_path, 'clustered_sigmas.png')

    # occ_inds = np.where(clustered_sigma > 0.5)
    occ_inds = np.where(semantic_map != 0)
    samples = np.stack(np.meshgrid(x, y, z), -1)
    occ_samples = samples[occ_inds[0], occ_inds[1], occ_inds[2], :]
    min_corner = np.array([np.min(occ_samples[:, 0]), np.min(occ_samples[:, 1]), np.min(occ_samples[:, 2])])
    max_corner = np.array([np.max(occ_samples[:, 0]), np.max(occ_samples[:, 1]), np.max(occ_samples[:, 2])])
    min_pt, max_pt = get_max_cube(min_corner, max_corner)
    box_pcd, coords = get_coords(min_pt, max_pt, N)
    print(min_corner, max_corner, min_pt, max_pt)

    xyz_ = torch.FloatTensor(coords.reshape(N ** 2, N, 3)).cuda()
    dir_ = torch.zeros(N ** 2, 3).cuda()
           # sigma is independent of direction, so any value here will produce the same result

    # predict sigma (occupancy) for each grid location
    print('Predicting occupancy for resized cube...')
    raw = network_query_fn(xyz_, dir_, network_fn)
    sigma = raw[..., 3].cpu().numpy().reshape((N, N, N))
    plot_sigmas(sigma, save_path, 'resampled_sigmas.png')

    if kwargs['semantic_en']:
        semantic_map = probs_to_semantic_3d(raw[..., 4:], N)
        plot_sigmas(semantic_map, save_path, 'semantics.png')
        semantics_filename = os.path.join(save_path, 'semantics_%d.npy' % (N))
        np.save(semantics_filename, semantic_map)

    pos_sigma_inds = np.where(sigma > 0)
    pos_sigma = sigma[pos_sigma_inds[0], pos_sigma_inds[1], pos_sigma_inds[2]]
    plot_sigmas(pos_sigma, save_path, 'resampled_sigmas_positive.png')

    thresh_sigma_inds = np.where(sigma > sigma_threshold)
    thresh_sigma = sigma[thresh_sigma_inds[0], thresh_sigma_inds[1], thresh_sigma_inds[2]]
    plot_sigmas(thresh_sigma, save_path, 'resampled_sigmas_thresh.png')

    sigmas_filename = os.path.join(save_path, 'sigmas_%d.npy' % (N))
    np.save(sigmas_filename, sigma)

    raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)
    z_vals = torch.Tensor(coords[..., 2]).to(device).reshape((N * N, N))
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]
    alpha = raw2alpha(raw[...,3], dists)  # [N_rays, N_samples]
    weights = (alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]).cpu().numpy().reshape((N, N, N))
    alpha = alpha.cpu().numpy().reshape((N, N, N))
    plot_sigmas(alpha, save_path, 'resized_alphas.png')
    plot_sigmas(weights, save_path, 'resized_weights.png')
    alphas_filename = os.path.join(save_path, 'alphas_%d.npy' % (N))
    np.save(alphas_filename, alpha)
    weights_filename = os.path.join(save_path, 'weights_%d.npy' % (N))
    np.save(weights_filename, weights)

    samples = coords.reshape((-1, 3))
    samples = translate_obj(samples, min_pt, max_pt)
    min_corner = np.array([np.min(samples[:, 0]), np.min(samples[:, 1]), np.min(samples[:, 2])])
    max_corner = np.array([np.max(samples[:, 0]), np.max(samples[:, 1]), np.max(samples[:, 2])])
    samples = samples / max_corner
    samples = samples.reshape((N, N, N, 3))
    samples_filename = os.path.join(save_path, 'samples_%d.npy' % (N))
    np.save(samples_filename, samples)

    # perform marching cube algorithm to retrieve vertices and triangle mesh
    print('Extracting mesh ...')
    vertices, triangles = mcubes.marching_cubes(sigma, sigma_threshold)

    ##### Until mesh extraction here, it is the same as the original repo. ######

    vertices_ = (vertices/N).astype(np.float32)
    ## invert x and y coordinates (WHY? maybe because of the marching cubes algo)
    x_ = (max_pt[1]-min_pt[1]) * vertices_[:, 1] + min_pt[1]
    y_ = (max_pt[0]-min_pt[0]) * vertices_[:, 0] + min_pt[0]
    vertices_[:, 0] = x_
    vertices_[:, 1] = y_
    vertices_[:, 2] = (zmax-zmin) * vertices_[:, 2] + zmin
    vertices_.dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]

    face = np.empty(len(triangles), dtype=[('vertex_indices', 'i4', (3,))])
    face['vertex_indices'] = triangles

    PlyData([PlyElement.describe(vertices_[:, 0], 'vertex'),
             PlyElement.describe(face, 'face')]).write(f'{save_path}/mesh_{sigma_threshold}_{N}.ply')

    # remove noise in the mesh by keeping only the biggest cluster
    print('Removing noise ...')
    mesh = o3d.io.read_triangle_mesh(f"{save_path}/mesh_{sigma_threshold}_{N}.ply")
    idxs, count, _ = mesh.cluster_connected_triangles()
    max_cluster_idx = np.argmax(count)
    triangles_to_remove = [i for i in range(len(face)) if idxs[i] != max_cluster_idx]
    mesh.remove_triangles_by_index(triangles_to_remove)
    mesh.remove_unreferenced_vertices()
    print(f'Mesh has {len(mesh.vertices):.2f} vertices and {len(mesh.triangles):.2f} faces.')
    print(f'Mesh has {len(mesh.vertices)/1e6:.2f} M vertices and {len(mesh.triangles)/1e6:.2f} M faces.')

    vertices_ = np.asarray(mesh.vertices).astype(np.float32)
    triangles = np.asarray(mesh.triangles)

    # perform color prediction
    # Step 0. define constants (image width, height and intrinsics)
    # W, H = args.img_wh
    # K = np.array([[dataset.focal, 0, W/2],
                  # [0, dataset.focal, H/2],
                  # [0,             0,   1]]).astype(np.float32)

    # Step 1. transform vertices into world coordinate
    N_vertices = len(vertices_)
    vertices_homo = np.concatenate([vertices_, np.ones((N_vertices, 1))], 1) # (N, 4)

    if use_vertex_normal: ## use normal vector method as suggested by the author.
                               ## see https://github.com/bmild/nerf/issues/44
        mesh.compute_vertex_normals()
        rays_d = torch.FloatTensor(np.asarray(mesh.vertex_normals))
        near = min_b * torch.ones_like(rays_d[:, :1])
        far = max_b * torch.ones_like(rays_d[:, :1])
        rays_o = torch.FloatTensor(vertices_) - rays_d * near * near_t
        rays = [rays_o, rays_d]

        rgb, disp, acc, extras = render(0, 0, [[]], chunk=1024*32, rays=rays,
                                        **kwargs)

    else: ## use my color average method. see README_mesh.md
        ## buffers to store the final averaged color
        non_occluded_sum = np.zeros((N_vertices, 1))
        v_color_sum = np.zeros((N_vertices, 3))

        # Step 2. project the vertices onto each training image to infer the color
        print('Fusing colors ...')
        for idx in tqdm(range(len(dataset.image_paths))):
            ## read image of this pose
            image = Image.open(dataset.image_paths[idx]).convert('RGB')
            image = image.resize(tuple(args.img_wh), Image.LANCZOS)
            image = np.array(image)

            ## read the camera to world relative pose
            P_c2w = np.concatenate([dataset.poses[idx], np.array([0, 0, 0, 1]).reshape(1, 4)], 0)
            P_w2c = np.linalg.inv(P_c2w)[:3] # (3, 4)
            ## project vertices from world coordinate to camera coordinate
            vertices_cam = (P_w2c @ vertices_homo.T) # (3, N) in "right up back"
            vertices_cam[1:] *= -1 # (3, N) in "right down forward"
            ## project vertices from camera coordinate to pixel coordinate
            vertices_image = (K @ vertices_cam).T # (N, 3)
            depth = vertices_image[:, -1:]+1e-5 # the depth of the vertices, used as far plane
            vertices_image = vertices_image[:, :2]/depth
            vertices_image = vertices_image.astype(np.float32)
            vertices_image[:, 0] = np.clip(vertices_image[:, 0], 0, W-1)
            vertices_image[:, 1] = np.clip(vertices_image[:, 1], 0, H-1)

            ## compute the color on these projected pixel coordinates
            ## using bilinear interpolation.
            ## NOTE: opencv's implementation has a size limit of 32768 pixels per side,
            ## so we split the input into chunks.
            colors = []
            remap_chunk = int(3e4)
            for i in range(0, N_vertices, remap_chunk):
                colors += [cv2.remap(image,
                                    vertices_image[i:i+remap_chunk, 0],
                                    vertices_image[i:i+remap_chunk, 1],
                                    interpolation=cv2.INTER_LINEAR)[:, 0]]
            colors = np.vstack(colors) # (N_vertices, 3)

            ## predict occlusion of each vertex
            ## we leverage the concept of NeRF by constructing rays coming out from the camera
            ## and hitting each vertex; by computing the accumulated opacity along this path,
            ## we can know if the vertex is occluded or not.
            ## for vertices that appear to be occluded from every input view, we make the
            ## assumption that its color is the same as its neighbors that are facing our side.
            ## (think of a surface with one side facing us: we assume the other side has the same color)

            ## ray's origin is camera origin
            rays_o = torch.FloatTensor(dataset.poses[idx][:, -1]).expand(N_vertices, 3)
            ## ray's direction is the vector pointing from camera origin to the vertices
            rays_d = torch.FloatTensor(vertices_) - rays_o # (N_vertices, 3)
            rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
            near = dataset.bounds.min() * torch.ones_like(rays_o[:, :1])
            ## the far plane is the depth of the vertices, since what we want is the accumulated
            ## opacity along the path from camera origin to the vertices
            far = torch.FloatTensor(depth) * torch.ones_like(rays_o[:, :1])
            results = f([nerf_fine], embeddings,
                        torch.cat([rays_o, rays_d, near, far], 1).cuda(),
                        args.N_samples,
                        0,
                        args.chunk,
                        dataset.white_back)
            opacity = results['opacity_coarse'].cpu().numpy()[:, np.newaxis] # (N_vertices, 1)
            opacity = np.nan_to_num(opacity, 1)

            non_occluded = np.ones_like(non_occluded_sum) * 0.1/depth # weight by inverse depth
                                                                    # near=more confident in color
            non_occluded += opacity < args.occ_threshold

            v_color_sum += colors * non_occluded
            non_occluded_sum += non_occluded

    # Step 3. combine the output and write to file
    if use_vertex_normal:
        v_colors = rgb.cpu().numpy() * 255.0
    else: ## the combined color is the average color among all views
        v_colors = v_color_sum/non_occluded_sum
    v_colors = v_colors.astype(np.uint8)
    v_colors.dtype = [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    vertices_.dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    vertex_all = np.empty(N_vertices, vertices_.dtype.descr+v_colors.dtype.descr)
    for prop in vertices_.dtype.names:
        vertex_all[prop] = vertices_[prop][:, 0]
    for prop in v_colors.dtype.names:
        vertex_all[prop] = v_colors[prop][:, 0]

    face = np.empty(len(triangles), dtype=[('vertex_indices', 'i4', (3,))])
    face['vertex_indices'] = triangles

    PlyData([PlyElement.describe(vertex_all, 'vertex'),
             PlyElement.describe(face, 'face')]).write(f'{save_path}/mesh_colored_{sigma_threshold}_{N}.ply')

    print('Done!')


def train():

    parser = config_parser()
    args = parser.parse_args()

    # Load data
    K = None
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'local_blender':
        images, poses, render_poses, meta, masks, gt_depths, i_split = load_local_blender_data(args.datadir, args.res, args.testskip, args.max_ind)
        K = meta['intrinsic_mat']
        hwf = [meta['height'], meta['width'], meta['fx']]
        print('Loaded local blender', images.shape, poses.shape, render_poses.shape, K, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = args.near
        far = args.far

        if args.white_bkgd:
            # images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
            binary_masks = np.where(masks > 0, 1, 0)
            binary_masks = np.repeat(binary_masks[..., :, :, np.newaxis], 3, axis=3)
            images = images[..., :3] * binary_masks + (1. - binary_masks)

        else:
            images = images[...,:3]

    elif args.dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res, args.testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    elif args.dataset_type == 'draco':
        images, poses, render_poses, meta, gt_depths, masks, i_split = load_draco_data(args.datadir, args.res, args.testskip)
        K = meta['intrinsic_mat']
        hwf = [meta['height'], meta['width'], meta['fx']]
        print('Loaded draco', images.shape, poses.shape, render_poses.shape, K, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = args.near
        far = args.far

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            if args.gt_register:
                rgbs, disps, depths = render_path(poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor, gt_depths=gt_depths)
            else:
                extract_mesh(args.N_samples, args.x_range, args.y_range, args.z_range, args.sigma_threshold, render_kwargs_train['network_query_fn'], render_kwargs_train['network_fn'], near, far, testsavedir, render_kwargs_test)
                rgbs, disps, depths = render_path(poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)

            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

    if args.wand_en:
            wandb.init(project="NeRF",
                       entity="rrc_3d",
                       name=expname)

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)

        if N_rand:
            print('shuffle rays')
            np.random.shuffle(rays_rgb)

        print('done')
        i_batch = 0

    # Move training data to GPU
    if use_batching:
        images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)


    N_iters = args.iters + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    
    if not N_rand:
        N_rand = H * W

    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()

        # Sample random ray batch
        if use_batching:
            # Random over all images
            batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                # print("Shuffle data after an epoch!")
                # rand_idx = torch.randperm(rays_rgb.shape[0])
                # rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:
            # Random from one image
            # img_i = np.random.choice(i_train)
            img_i = i % len(i_train)
            target = images[img_i]
            target = torch.Tensor(target).to(device)
            pose = poses[img_i, :3,:4]
            if args.semantic_en:
                target_sem = masks[img_i]
                target_sem = torch.Tensor(target_sem).to(device)

            if not (i % len(i_train)) and (i / len(i_train) > 0):
                print("Completed %d epochs!" % (i // len(i_train)))

            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                # target_s = target.reshape((H * W, 3))
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                if args.semantic_en:
                    target_sem_s = target_sem[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        #####  Core optimization loop  #####
        rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)

        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        trans = extras['raw'][...,-1]
        loss = args.rgb_wt * img_loss
        psnr = mse2psnr(img_loss)
        semantic_loss = 0
        if args.semantic_en:
            semantic_loss = mask2entropy(extras['semantic_map'], target_sem_s.type(torch.LongTensor).cuda())
            loss = loss + args.semantic_wt * semantic_loss

        rays_sparsity_loss = sigmas2loss(extras['raw'][..., 3], args.rays_sparsity_scale)
        loss = loss + args.rays_sparsity_wt * rays_sparsity_loss

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + args.rgb_wt * img_loss0
            psnr0 = mse2psnr(img_loss0)

            semantic_loss0 = 0
            if 'semantic0' in extras:
                semantic_loss0 = mask2entropy(extras['semantic0'], target_sem_s.type(torch.LongTensor).cuda())
                loss = loss + args.semantic_wt * semantic_loss0

        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time()-time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        # print("Training rgb info: ", type(rgb), rgb.shape, target_s.shape)
        # print(np.unique(rgb.detach().cpu().numpy(), return_counts=True))
        # print(np.unique(target_s.detach().cpu().numpy(), return_counts=True))
        # tqdm.write(f"Info: {rgb.shape}, {target_s.shape}, \nUnique GT Values: {np.unique(target_s.detach().cpu().numpy(), return_counts=True)} \nUnique Rendered Values: {np.unique(rgb.detach().cpu().numpy(), return_counts=True)}")
        if args.wand_en:
            render_H = int(N_rand ** 0.5)
            render_W = int(N_rand ** 0.5)
            if N_rand == H * W:
                render_H = H
                render_W = W

            log_dict = {
                "Train Loss": loss.item(),
                "Train PSNR": psnr.item(),
                "Rendered vs GT Train Image": [wandb.Image(rgb.detach().cpu().numpy().reshape((render_H, render_W, 3))), wandb.Image(target_s.detach().cpu().numpy().reshape((render_H, render_W, 3)))]
                }

            if args.semantic_en:
                semantic_gt = target_sem_s.detach().cpu().numpy().reshape((render_H, render_W))
                semantic_gt = labels_to_pallette(semantic_gt, tensor = False)

                semantic_pred = torch.nn.Softmax(dim=1)(extras['semantic_map']).max(dim=1).indices
                semantic_pred = semantic_pred.detach().cpu().numpy().reshape((render_H, render_W))
                semantic_pred = labels_to_pallette(semantic_pred, tensor = False)

                log_dict["Train RGB Loss"] = (img_loss + img_loss0).item()
                log_dict["Train Semantic Loss"] = (semantic_loss + semantic_loss0).item()
                log_dict["Train Rays Sparsity Loss"] = (rays_sparsity_loss).item()
                log_dict["Rendered vs GT Train Semantic Mask"] = [wandb.Image(semantic_pred), wandb.Image(semantic_gt)]

            wandb.log(log_dict)

        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i%args.i_video==0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps, depths = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

            # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     with torch.no_grad():
            #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        if i%args.i_testset==0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')


    
        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")

            if i%args.i_img==0:
                # Log a rendered validation view
                img_i=np.random.choice(i_val)
                target = images[img_i]
                pose = poses[img_i, :3,:4]
                if args.semantic_en:
                    target_sem = masks[img_i]
                    target_sem = torch.Tensor(target_sem).to(device)

                with torch.no_grad():
                    rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, c2w=pose,
                                                        **render_kwargs_test)

                img_loss = img2mse(rgb, torch.tensor(target))
                loss = args.rgb_wt * img_loss
                psnr = mse2psnr(img_loss)
                semantic_loss = 0
                if args.semantic_en:
                    semantic_loss = mask2entropy(extras['semantic_map'].reshape((H * W, -1)), target_sem.type(torch.LongTensor).cuda().reshape((H * W)))
                    loss = loss + args.semantic_wt * semantic_loss

                if 'rgb0' in extras:
                    img_loss0 = img2mse(extras['rgb0'], torch.tensor(target))
                    loss = loss + args.rgb_wt * img_loss0
                    psnr0 = mse2psnr(img_loss0)
                    
                    semantic_loss0 = 0
                    if args.semantic_en:
                        semantic_loss0 = mask2entropy(extras['semantic0'].reshape((H * W, -1)), target_sem.type(torch.LongTensor).cuda().reshape((H * W)))
                        loss = loss + args.semantic_wt * semantic_loss0

                rays_sparsity_loss = sigmas2loss(extras['sigma_map'], args.rays_sparsity_scale)
                loss = loss + args.rays_sparsity_wt * rays_sparsity_loss

                tqdm.write(f"[TRAIN] Iter: {i} Validation Loss: {loss.item()}  Validation PSNR: {psnr.item()}")
                if args.wand_en:
                    log_dict = {
                        "Validation Loss": loss.item(),
                        "Validation PSNR": psnr.item(),
                        "Rendered vs GT Image": [wandb.Image(rgb.cpu().numpy().reshape((H, W, 3))), wandb.Image(target)],
                        "Disparity": [wandb.Image(disp.cpu().numpy())],
                        "Opacity": [wandb.Image(acc.cpu().numpy())],
                        "Depth": [wandb.Image(extras['depth_map'].cpu().numpy())],
                        }

                    if args.semantic_en:
                        semantic_gt = target_sem.detach().cpu().numpy().reshape((H, W))
                        semantic_gt = labels_to_pallette(semantic_gt, tensor = False)

                        semantic_pred = torch.nn.Softmax(dim=1)(extras['semantic_map'].reshape((H * W, -1))).max(dim=1).indices
                        semantic_pred = semantic_pred.detach().cpu().numpy().reshape((H, W))
                        semantic_pred = labels_to_pallette(semantic_pred, tensor = False)

                        log_dict["Validation RGB Loss"] = (img_loss + img_loss0).item()
                        log_dict["Validation Semantic Loss"] = (semantic_loss + semantic_loss0).item()
                        log_dict["Validation Rays Sparsity Loss"] = (rays_sparsity_loss).item()
                        log_dict["Rendered vs GT Semantic Mask"] = [wandb.Image(semantic_pred), wandb.Image(semantic_gt)]

                    wandb.log(log_dict)

        """
            print(expname, i, psnr.numpy(), loss.numpy(), global_step.numpy())
            print('iter time {:.05f}'.format(dt))

            with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
                tf.contrib.summary.scalar('loss', loss)
                tf.contrib.summary.scalar('psnr', psnr)
                tf.contrib.summary.histogram('tran', trans)
                if args.N_importance > 0:
                    tf.contrib.summary.scalar('psnr0', psnr0)


            if i%args.i_img==0:

                # Log a rendered validation view to Tensorboard
                img_i=np.random.choice(i_val)
                target = images[img_i]
                pose = poses[img_i, :3,:4]
                with torch.no_grad():
                    rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose,
                                                        **render_kwargs_test)

                psnr = mse2psnr(img2mse(rgb, target))

                with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):

                    tf.contrib.summary.image('rgb', to8b(rgb)[tf.newaxis])
                    tf.contrib.summary.image('disp', disp[tf.newaxis,...,tf.newaxis])
                    tf.contrib.summary.image('acc', acc[tf.newaxis,...,tf.newaxis])

                    tf.contrib.summary.scalar('psnr_holdout', psnr)
                    tf.contrib.summary.image('rgb_holdout', target[tf.newaxis])


                if args.N_importance > 0:

                    with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                        tf.contrib.summary.image('rgb0', to8b(extras['rgb0'])[tf.newaxis])
                        tf.contrib.summary.image('disp0', extras['disp0'][tf.newaxis,...,tf.newaxis])
                        tf.contrib.summary.image('z_std', extras['z_std'][tf.newaxis,...,tf.newaxis])
        """

        global_step += 1
        # del batch, batch_rays, target_s, img_loss, rgb, disp, acc, extras


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(device_idx)

    train()
