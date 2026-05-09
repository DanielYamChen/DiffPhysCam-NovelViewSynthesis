# Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import time
import argparse
import json
import multiprocessing
import glob
import re

import numpy as np
from math import pi
import torch
import nvdiffrast.torch as dr
import xatlas
# import igl
import random
import multiprocessing
import shutil
import matplotlib.pyplot as plt
from datetime import datetime

torch_random_seed = 5678
torch.manual_seed(torch_random_seed)
random.seed(torch_random_seed)
np.random.seed(torch_random_seed)

# Import data readers / generators
from dataset import DatasetMesh, DatasetNERF, DatasetLLFF, DatasetCustom, InOrderBatchSampler

# Import topology / geometry trainers
from geometry.dmtet import DMTetGeometry
from geometry.dlmesh import DLMesh

import render.renderutils as ru
from render import obj
from render import material
from render import util
from render import mesh
from render import texture
from render import mlptexture
from render import light
from render import render
from render.mesh import Mesh

from denoiser.denoiser import BilateralDenoiser
from models.cam_model import PhysDiffCamera
from models import networks

RADIUS = 3.0

# Enable to debug back-prop anomalies
#torch.autograd.set_detect_anomaly(True)

###############################################################################
# Mix background into a dataset image
###############################################################################

@torch.no_grad()
def prepare_batch(target, train_res, bg_type):

    target['model_view_trnsfrm'] = target['model_view_trnsfrm'].cuda()
    target['mvp_trnsfrm'] = target['mvp_trnsfrm'].cuda()
    target['cam_posi'] = target['cam_posi'].cuda()
    target['img'] = target['img'].cuda()

    ## rescale batch of target images to align with train-resolutions
    if (train_res[0] != target['img'].shape[1] or train_res[1] != target['img'].shape[2]):
        target['img'] = util.scale_img_nhwc(target['img'], train_res)
        target['resolution'] = train_res

    assert (len(target['img'].shape) == 4), "Image shape should be [n, h, w, c]"
    if bg_type == 'checker':
        background = torch.tensor(util.checkerboard(target['img'].shape[1:3], 8), dtype=torch.float32, device='cuda')[None, ...]
    elif bg_type == 'black':
        background = torch.zeros(target['img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
    elif bg_type == 'white':
        background = torch.ones(target['img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
    elif bg_type == 'reference':
        background = target['img'][..., 0:3]
    elif bg_type == 'random':
        background = torch.rand(target['img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
    else:
        assert False, "Unknown background type %s" % bg_type

    target['background'] = background
    target['img'] = torch.cat((torch.lerp(background, target['img'][..., 0:3], target['img'][..., 3:4]), target['img'][..., 3:4]), dim=-1)

    return target

###############################################################################
# Utility functions for material
###############################################################################

def initial_guess_material(geometry, mlp, FLAGS, init_mat=None):
    kd_min, kd_max = torch.tensor(FLAGS.kd_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.kd_max, dtype=torch.float32, device='cuda')
    ks_min, ks_max = torch.tensor(FLAGS.ks_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.ks_max, dtype=torch.float32, device='cuda')
    nrm_min, nrm_max = torch.tensor(FLAGS.nrm_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.nrm_max, dtype=torch.float32, device='cuda')
    if mlp:
        mlp_min = torch.cat((kd_min[0:3], ks_min), dim=0)
        mlp_max = torch.cat((kd_max[0:3], ks_max), dim=0)
        mlp_map_opt = mlptexture.MLPTexture3D(geometry.getAABB(), channels=6, min_max=[mlp_min, mlp_max])
        mat = {'kd_ks' : mlp_map_opt}
    else:
        # Setup Kd, Ks albedo, and specular textures
        assert (init_mat is not None), "Initial material is not provided"
        
        kd_map_opt = texture.create_trainable(init_mat['kd'], FLAGS.texture_res, not FLAGS.custom_mip, [kd_min, kd_max])
        ks_map_opt = texture.create_trainable(init_mat['ks'], FLAGS.texture_res, not FLAGS.custom_mip, [ks_min, ks_max])
        normal_map_opt = texture.create_trainable(init_mat['normal'], FLAGS.texture_res, not FLAGS.custom_mip, [nrm_min, nrm_max])

        mat = {
            'kd'     : kd_map_opt,
            'ks'     : ks_map_opt,
            'normal' : normal_map_opt
        }

    mat['bsdf'] = FLAGS.bsdf

    mat['no_perturbed_nrm'] = FLAGS.no_perturbed_nrm

    return mat

###############################################################################
# Validation & testing
###############################################################################

def validate_itr(glctx, batch_targets, ref_mesh, geometry, opt_material, lgt, FLAGS, denoiser, iter=0, phys_cam=None, defocus_net=None):
    result_dict = {}
    with torch.no_grad():
        opt_mesh = geometry.getMesh(opt_material)
        
        ## debug
        # print(f"\n opt_mesh={opt_mesh} \n")
        # print(f"\n batch_targets['cam_trnsfrm']={torch.linalg.inv(batch_targets['model_view_trnsfrm'][0])} \n")
        # print(f"\n batch_targets['cam_posi']={batch_targets['cam_posi']} \n")
        # print(f"\n batch_targets['envmap_name']={batch_targets['envmap_name']} \n")

        buffers = render.render_mesh(
            FLAGS,
            glctx,
            opt_mesh,
            batch_targets['mvp_trnsfrm'],
            batch_targets['cam_posi'],
            batch_targets['light'] if (lgt is None) else lgt,
            batch_targets['resolution'],
            batch_targets['defocus_mtrx_name'] if ('defocus_mtrx_name' in batch_targets) else None,
            batch_targets['cam_ctrl_params'],
            spp=batch_targets['spp'],
            num_layers=FLAGS.layers,
            background=batch_targets['background'],
            optix_ctx=geometry.optix_ctx,
            denoiser=denoiser,
            phys_cam=phys_cam,
            defocus_net=defocus_net,
        )
        ## debug
        # temp = buffers['shaded'].detach().cpu().numpy()
        # plt.imshow(temp[0, :, :, :3])
        # plt.show()
        # plt.imshow(temp[0, :, :, 3], cmap='gray')
        # plt.show()

        ## mask out rendered pixels outside of ROI
        buffers['shaded'][0, :, :, 0:4][buffers['shaded'][0, :, :, 3] < 0.99] = 0.
        # print(buffers['shaded'][0, :, :, 3] > 0.99)
        num_effect_pxs = np.sum((buffers['shaded'][0, :, :, 3] > 0.99).detach().cpu().numpy())

        result_dict['ref'] = util.rgb_to_srgb(batch_targets['img'][0, :, :, :])
        result_dict['opt'] = util.rgb_to_srgb(buffers['shaded'][0, :, :, :])
        result_image = torch.cat([result_dict['opt'], result_dict['ref']], axis=1)

        if (FLAGS.display is not None):
            white_bg = torch.ones_like(batch_targets['background'])
            for layer in FLAGS.display:
                if ('latlong' in layer and layer['latlong']):
                    result_dict['light_image'] = lgt.generate_image(FLAGS.display_res)
                    result_dict['light_image'] = util.rgb_to_srgb(result_dict['light_image'] / (1 + result_dict['light_image']))
                    result_image = torch.cat([result_image, result_dict['light_image']], axis=1)
                
                elif ('bsdf' in layer):
                    img = render.render_mesh(
                        FLAGS,
                        glctx,
                        opt_mesh,
                        batch_targets['mvp_trnsfrm'],
                        batch_targets['cam_posi'],
                        batch_targets['light'] if lgt is None else lgt,
                        batch_targets['resolution'],
                        batch_targets['defocus_mtrx_name'] if ('defocus_mtrx_name' in batch_targets) else None,
                        batch_targets['cam_ctrl_params'],
                        spp=batch_targets['spp'],
                        num_layers=FLAGS.layers,
                        background=white_bg,
                        bsdf=layer['bsdf'],
                        optix_ctx=geometry.optix_ctx,
                        phys_cam=phys_cam,
                        defocus_net=defocus_net,
                    )['shaded']
                    
                    if (layer['bsdf'] == 'kd'):
                        result_dict[layer['bsdf']] = util.rgb_to_srgb(img[..., 0:3])[0]
                    else:
                        result_dict[layer['bsdf']] = img[0, ..., 0:3]
                    
                    result_image = torch.cat([result_image, result_dict[layer['bsdf']]], axis=1)
                    
                    if (ref_mesh is not None):
                        img = render.render_mesh(
                            FLAGS,
                            glctx,
                            ref_mesh,
                            batch_targets['mvp_trnsfrm'],
                            batch_targets['cam_posi'],
                            batch_targets['light'],
                            batch_targets['resolution'],
                            batch_targets['defocus_mtrx_name'] if ('defocus_mtrx_name' in batch_targets) else None,
                            batch_targets['cam_ctrl_params'],
                            spp=batch_targets['spp'],
                            num_layers=FLAGS.layers,
                            background=white_bg,
                            bsdf=layer['bsdf'],
                            optix_ctx=geometry.optix_ctx,
                            phys_cam=phys_cam,
                            defocus_net=defocus_net,
                        )['shaded']

                        if (layer['bsdf'] == 'kd'):
                            result_dict[layer['bsdf'] + "_ref"] = util.rgb_to_srgb(img[..., 0:3])[0]
                        else:
                            result_dict[layer['bsdf'] + "_ref"] = img[0, ..., 0:3]
                        
                        result_image = torch.cat([result_image, result_dict[layer['bsdf'] + "_ref"]], axis=1)
                
                elif ('normals' in layer and not FLAGS.no_perturbed_nrm):
                    result_image = torch.cat([result_image, (buffers['perturbed_nrm'][0, ...,0:3] + 1.0) * 0.5], axis=1)
                
                elif ('diffuse_light' in layer):
                    result_image = torch.cat([result_image, util.rgb_to_srgb(buffers['diffuse_light'][..., 0:3])[0]], axis=1)
                
                elif ('specular_light' in layer):
                    result_image = torch.cat([result_image, util.rgb_to_srgb(buffers['specular_light'][..., 0:3])[0]], axis=1)


        return result_image, result_dict, num_effect_pxs


def validate(glctx, geometry, opt_material, lgt, dataset_valid, out_dir, FLAGS, denoiser, phys_cam=None, defocus_net=None):

    # ==============================================================================================
    #  Validation loop
    # ==============================================================================================
    img_cnt = 0
    mse_values = []
    psnr_values = []

    # Hack validation to use high sample count and no denoiser
    _n_samples = FLAGS.n_samples
    _denoiser = denoiser
    FLAGS.n_samples = 32
    denoiser = None

    dataloader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=1, collate_fn=dataset_valid.collate)

    os.makedirs(out_dir, exist_ok=True)
    lines = []
    print("Running validation")
    for it, target in enumerate(dataloader_valid):

        # Mix validation background
        target = prepare_batch(target, FLAGS.train_res, FLAGS.background)
        if (FLAGS.add_defocus_net):
            defocus_net.eval()
        result_image, result_dict, num_effect_pxs = validate_itr(
            glctx, target, dataset_valid.getMesh(), geometry, opt_material, lgt, FLAGS, denoiser,
            phys_cam=phys_cam, defocus_net=defocus_net,
        )
        if (FLAGS.add_defocus_net):
            defocus_net.train()
        
        # Compute metrics
        opt = torch.clamp(result_dict['opt'][:, :, 0:3], 0.0, 1.0) 
        ref = torch.clamp(result_dict['ref'][:, :, 0:3], 0.0, 1.0)

        mse = torch.nn.functional.mse_loss(opt, ref, size_average=None, reduce=None, reduction='sum').item()
        mse = mse / num_effect_pxs
        mse_values.append(float(mse))
        psnr = util.mse_to_psnr(mse)
        psnr_values.append(float(psnr))

        lines.append(str("%d, %1.8f, %1.8f \n" % (it, mse, psnr)))

        for k in result_dict.keys():
            if ((k == "ref") and (FLAGS.save_gt_test_imgs == False)):
                continue
            else:
                np_img = result_dict[k].detach().cpu().numpy()
                util.save_image(out_dir + '/' + ('val_%s_%06d.png' % (k, it)), np_img)

    avg_mse = np.mean(np.array(mse_values))
    avg_psnr = np.mean(np.array(psnr_values))
    lines.append(str("AVERAGES: %1.4f, %2.3f\n" % (avg_mse, avg_psnr)))
    
    print("MSE,      PSNR")
    print("%1.8f, %2.3f" % (avg_mse, avg_psnr))
    
    with open(os.path.join(out_dir, 'metrics.txt'), 'w') as fout:
        fout.write('ID, MSE, PSNR\n')
        for line in lines:
            fout.write(line)

        

    # Restore sample count and denoiser
    FLAGS.n_samples = _n_samples
    denoiser = _denoiser

    return avg_psnr

###############################################################################
# Main shape fitter function / optimization loop
###############################################################################



# def lambda_rule(epoch):
#     return (1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1))



#----------------------------------------------------------------------------
# Main function.
#----------------------------------------------------------------------------

if __name__ == "__main__":
    
    ############################
    ######## Parameters ########
    ############################
    
    ## Parameter order: Arguments --> Defaults in code --> Config file
    
    parser = argparse.ArgumentParser(description='nvdiffrecmc')
    parser.add_argument('-i', '--iter', type=int, default=5000)
    parser.add_argument('-b', '--batch', type=int, default=1)
    parser.add_argument('-s', '--spp', type=int, default=1)
    parser.add_argument('-l', '--layers', type=int, default=1)
    parser.add_argument('-r', '--train-res', type=int, default=[512, 512])
    parser.add_argument('-dr', '--display-res', type=int, default=None)
    parser.add_argument('-tr', '--texture-res', nargs=2, type=int, default=[1024, 1024])
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.01)
    parser.add_argument('-mip', '--custom-mip', action='store_true', default=False)
    parser.add_argument('-bg', '--background', default='checker', choices=['black', 'white', 'checker', 'reference'])
    parser.add_argument('-o', '--out-dir', type=str, default=None)
    parser.add_argument('--config', type=str, default=None, help='Config file')
    parser.add_argument('-bm', '--base-mesh', type=str, default=None)
    parser.add_argument('--validate', type=bool, default=True)
    
    ## Render specific arguments
    parser.add_argument('--n_samples', type=int, default=4)
    parser.add_argument('--bsdf', type=str, default='pbr', choices=['pbr', 'diffuse', 'white'])
    
    ## Denoiser specific arguments
    parser.add_argument('--denoiser', default='bilateral', choices=['none', 'bilateral'])
    parser.add_argument('--denoiser_demodulate', type=bool, default=True)
    
    ## Customized arguments
    parser.add_argument('--scene', required=True, type=str, default="", help="Name of the scene to be optimized")
    parser.add_argument("--out_dir", type=str, required=True, default="", help="Output directory for logs and results")
    parser.add_argument("--base_mesh", type=str, required=True, default=None, help="Path to the mesh")
    parser.add_argument("--add_phys_cam", action='store_true', help="Whether to add the physics-based camera model into the optimization")
    parser.add_argument("--save_gt_test_imgs", action='store_true', help="Whether to save ground truth test images for visual comparison")
    parser.add_argument('--defocus_type', type=str, default=None, help="which defocus type to run: gaussian, uniform")
    parser.add_argument('--cond', type=str, default=None, help="which condition of using the physics-based camera: full, wo_expsr, wo_defocus")
    parser.add_argument('--test_set_name', type=str, default=None, help="name of the test set to be used for validation")
    
    FLAGS = parser.parse_args()
    
    if ("Real" == FLAGS.scene[:4]):
        
        if (FLAGS.out_dir[-5:] == "sugar"):
            FLAGS.config = "configs/real_scenes_sugar.json"    
        
        else:
            FLAGS.config = "configs/real_scenes.json"
    
    elif ("Sim" == FLAGS.scene[:3]):
        FLAGS.config = "configs/sim_scenes.json"
    
    FLAGS.mtl_override        = None        # Override material of model
    FLAGS.dmtet_grid          = 64          # Resolution of initial tet grid. We provide 64 and 128 resolution grids. 
                                            #    Other resolutions can be generated with https://github.com/crawforddoran/quartet
                                            #    We include examples in data/tets/generate_tets.py
    FLAGS.envlight            = None        # HDR environment probe
    FLAGS.env_scale           = 1.0         # Env map intensity multiplier
    FLAGS.probe_res           = 256         # Env map probe resolution
    FLAGS.learn_lighting      = False       # Enable optimization of env lighting
    FLAGS.display             = None        # Configure validation window/display. E.g. [{"bsdf" : "kd"}, {"bsdf" : "ks"}]
    FLAGS.transparency        = False       # Enabled transparency through depth peeling
    FLAGS.lock_light          = True        # Disable light optimization in the second pass
    FLAGS.lock_pos            = False       # Disable vertex position optimization in the second pass
    FLAGS.sdf_regularizer     = 0.2         # Weight for sdf regularizer.
    FLAGS.laplace             = "relative"  # Mesh Laplacian ["absolute", "relative"]
    FLAGS.laplace_scale       = 3000.0      # Weight for Laplace regularizer. Default is relative with large weight
    FLAGS.pre_load            = True        # Pre-load entire dataset into memory for faster training
    FLAGS.no_perturbed_nrm    = False       # Disable normal map
    FLAGS.decorrelated        = False       # Use decorrelated sampling in forward and backward passes
    FLAGS.kd_min              = [ 0.0,  0.0,  0.0,  0.0]
    FLAGS.kd_max              = [ 1.0,  1.0,  1.0,  1.0]
    FLAGS.ks_min              = [ 0.0,  0.05, 0.00]
    FLAGS.ks_max              = [ 0.0,  1.00, 1.00]
    FLAGS.nrm_min             = [-1.0, -1.0,  0.0]
    FLAGS.nrm_max             = [ 1.0,  1.0,  1.0]
    FLAGS.clip_max_norm       = 0.0
    FLAGS.cam_near_far        = [0.001, 1000.0] # [m]
    
    ## Customized flags
    FLAGS.fix_envlight        = False       # Whether to have only one environment map or more than one environment maps to use
    FLAGS.add_defocus_net     = False
    FLAGS.log_interval = 20
    FLAGS.save_interval = 20
    FLAGS.rand_seed = torch_random_seed
    FLAGS.mesh_offset = [0., 0., 0.]
    FLAGS.gt_img_dir = "imgs_wo_ground/"
    FLAGS.ref_mesh = f"../DiffPhysCam_Data/NovelViewSynthesis_Data/{FLAGS.scene}/"
    FLAGS.envlight = f"../DiffPhysCam_Data/NovelViewSynthesis_Data/{FLAGS.scene}/envmaps/"
    
    FLAGS.use_base_material = True
    
    ######################
    ######## Main ########
    ######################
    
    ## Load config from json file and override default and argument-provided flags
    if FLAGS.config is not None:
        
        data = json.load(open(FLAGS.config, 'r'))
        
        for key in data:
            FLAGS.__dict__[key] = data[key]
    
    if FLAGS.display_res is None:
        FLAGS.display_res = FLAGS.train_res
    
    print("Config / Flags:")
    print("---------")
    for key in FLAGS.__dict__.keys():
        print(f"{key} : {FLAGS.__dict__[key]}")
    
    print("---------")
    
    
    os.makedirs(FLAGS.out_dir, exist_ok=True)
      
    glctx         = dr.RasterizeGLContext() # Context for training
    glctx_display = glctx if FLAGS.batch_size < 16 else dr.RasterizeGLContext() # Context for display
    
    #### initialize the physics-based camera if added ####
    
    assert not(FLAGS.add_phys_cam and FLAGS.add_defocus_net), "phys_cam and defocus_net cannot be added simultaneously"
    
    if (FLAGS.add_phys_cam):
        
        assert(FLAGS.cond is not None), "Condition for using the physics-based camera is not specified ..."

        if (FLAGS.cond != "wo_defocus"):
            assert(FLAGS.defocus_type is not None), "defocus_type for using the physics-based camera is not specified ..."
    
        #### Paths ####
        if (FLAGS.cond != "wo_defocus"):
            phys_cam_model_params_path = f"../CameraCalibrateExp/phys_cam_model_params_defocus_{FLAGS.defocus_type}.json"
        else:
            phys_cam_model_params_path = f"../CameraCalibrateExp/phys_cam_model_params_defocus_gaussian.json"
            
        FLAGS.defocus_mtrx_base_dir = f"../DiffPhysCam_Data/NovelViewSynthesis_Data/{FLAGS.scene}/defocus_matrices_{FLAGS.defocus_type}_wo_ground/"
    
        noise_amp = 0.40
        near_clip = 0.005 # [m]
        far_clip = 100 # [m]
    
        phys_cam = PhysDiffCamera(FLAGS.train_res[0], FLAGS.train_res[1], torch_random_seed, 'cuda')
        with open(phys_cam_model_params_path) as json_file:
            cam_params = json.load(json_file)
    
        pixel_size = cam_params["pixel_size"] # [m]
        rgb_QEs = np.array(cam_params["rgb_QEs"], dtype=float)
        
        gain_params = cam_params["gain_params"]
        gain_params["max_CoC"] = 15 # [px]
        
        noise_params = cam_params["noise_params"]
        noise_params["noise_gains"] = noise_amp * np.array(noise_params["noise_gains"], dtype=float)
        noise_params["STD_reads"] = noise_amp * np.array(noise_params["STD_reads"], dtype=float)
        
        if (FLAGS.scene == "RealScene01"):
            print("customized cam model params for Real Scene 1 ....")

            ## Lens parameters (Arducam LN042 5mm lens)
            focal_length = 0.00572951691782118 # [m]
            hFOV = 1.1278099154119037 # [rad]

            max_scene_light = 10000 * 0.10 / np.sum(np.array([0.825, 0., 0.825])**2) # [lux = lumen/m^2]
            # max_scene_light = 1080 # [lux = lumen/m^2]

        elif (FLAGS.scene == "RealScene02"):
            print("customized cam model params for Real Scene 2 ....")

            ## Lens parameters (Arducam LN042 5mm lens)
            focal_length = 0.005641539774724099 # [m]
            hFOV = 1.1475922691663856 # [rad]

            max_scene_light = 10000 * 0.10 / np.sum(np.array([0.825, 0., 0.825])**2) # [lux = lumen/m^2]
            # max_scene_light = 1080 # [lux = lumen/m^2]
    
        elif ("SimScene" in FLAGS.scene):
            print("Customized cam model params for Sim Scenes ....")
            
            focal_length = 0.005 # [m]
            hFOV = 55.7 * pi / 180.0 # [rad]

            max_scene_light = 400        
    
        sensor_width = 2 * focal_length * np.tan(hFOV / 2) # [m]

        phys_cam.SetModelParameters(sensor_width, pixel_size, max_scene_light, rgb_QEs, gain_params, noise_params)
        phys_cam.BuildVignetMask(sensor_width, focal_length) 
        
        if (FLAGS.cond == "full"):
            phys_cam.artifact_switches = {
                "vignetting": True,
                "defocus_blur": True,
                "aggregate": True,
                "add_noise": False,
                "expsr2dv": True,
            }
        elif (FLAGS.cond == "wo_defocus"):
            phys_cam.artifact_switches = {
                "vignetting": True,
                "defocus_blur": False,
                "aggregate": True,
                "add_noise": False,
                "expsr2dv": True,
            }
        elif (FLAGS.cond == "wo_expsr"):
            phys_cam.artifact_switches = {
                "vignetting": False,
                "defocus_blur": True,
                "aggregate": False,
                "add_noise": False,
                "expsr2dv": False,
            }
        else:
            assert False, "Unknown condition with adding the DiffPhysCam ..."
    
        defocus_net = None
    
    elif (FLAGS.add_defocus_net):
        defocus_net = networks.define_G(4, 3, 64, "unet_1024", gpu_ids=[0], use_dropout=True)
        phys_cam = None
    
    else:
        phys_cam = None
        defocus_net = None
    
    ## log
    if (FLAGS.add_phys_cam):
        print()
        print("phys_cam added:")
        for key in phys_cam.artifact_switches.keys():
            print(f"{key} : {phys_cam.artifact_switches[key]}")

        print()
    
    #### Set up random seed
    random.seed(FLAGS.rand_seed)
    np.random.seed(FLAGS.rand_seed)
    torch.manual_seed(FLAGS.rand_seed)
    
    # ==============================================================================================
    #  Create data pipeline
    # ==============================================================================================
    if (os.path.splitext(FLAGS.ref_mesh)[1] == '.obj'):
        ref_mesh      = mesh.load_mesh(FLAGS.ref_mesh, FLAGS.mtl_override)
        dataset_valid = DatasetMesh(ref_mesh, glctx_display, RADIUS, FLAGS, validate=True, phys_cam=phys_cam)
    
    elif os.path.isdir(FLAGS.ref_mesh):
        if (any(sub in FLAGS.ref_mesh.split(os.sep)[-2] for sub in ['custom', 'RealScene', 'SimScene'])):
            dataset_valid = DatasetCustom(os.path.join(FLAGS.ref_mesh, FLAGS.test_set_name), FLAGS, num_samples=None)
        
        else:
            assert False, "Invalid dataset format"
    
    else:
        print("Invalid dataset format", FLAGS.ref_mesh)
        assert False, "Invalid dataset format"
    
    # ==============================================================================================
    #  Setup denoiser
    # ==============================================================================================
    
    denoiser = None
    if (FLAGS.denoiser == 'bilateral'):
        denoiser = BilateralDenoiser().cuda()
    else:
        assert (FLAGS.denoiser == 'none'), "Invalid denoiser %s" % FLAGS.denoiser
    
    # ==============================================================================================
    #  Train with fixed topology (mesh)
    # ==============================================================================================

    # Load initial guess mesh from file
    base_mesh = mesh.load_mesh(FLAGS.base_mesh)
    geometry = DLMesh(base_mesh, FLAGS)
    base_mesh.v_pos = base_mesh.v_pos.clone().detach().requires_grad_(True)
    base_material = base_mesh.material if (FLAGS.use_base_material is True) else None
    mat = initial_guess_material(geometry, False, FLAGS, init_mat=base_material)

    mat['no_perturbed_nrm'] = False
    print(f"mat['no_perturbed_nrm'] = {mat['no_perturbed_nrm']}")

    # ==============================================================================================
    #  Validate
    # ==============================================================================================
    validate(
        glctx_display,
        geometry,
        mat,
        None,
        dataset_valid,
        FLAGS.out_dir,
        FLAGS,
        denoiser,
        phys_cam=phys_cam,
        defocus_net=defocus_net,
    )


