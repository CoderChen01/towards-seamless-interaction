# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2023 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: mica@tue.mpg.de

import argparse
import os
from pathlib import Path

from yacs.config import CfgNode as CN

cfg = CN()
# mica
abs_mica_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
cfg.mica_dir = abs_mica_dir
cfg.device = "cuda"
cfg.device_id = "0"
cfg.pretrained_model_path = os.path.join(cfg.mica_dir, "data/pretrained", "mica.tar")
cfg.output_dir = ""
cfg.dataset_making = False
cfg.video_file = ""

cfg.model = CN()
cfg.model.testing = True
cfg.model.name = "mica"
cfg.model.topology_path = os.path.join("./data/FLAME2020", "head_template.obj")
cfg.model.flame_model_path = os.path.join("./data/FLAME2020", "generic_model.pkl")
cfg.model.flame_lmk_embedding_path = os.path.join(
    "./data/FLAME2020", "landmark_embedding.npy"
)
cfg.model.n_shape = 300
cfg.model.layers = 8
cfg.model.hidden_layers_size = 256
cfg.model.mapping_layers = 3
cfg.model.use_pretrained = True
cfg.model.arcface_pretrained_model = (
    "/scratch/is-rg-ncs/models_weights/arcface-torch/backbone100.pth"
)

cfg.dataset = CN()
cfg.dataset.training_data = ["LYHM"]
cfg.dataset.eval_data = ["FLORENCE"]
cfg.dataset.batch_size = 2
cfg.dataset.K = 4
cfg.dataset.n_train = 100000
cfg.dataset.num_workers = 4
cfg.dataset.root = "/datasets/MICA/"

cfg.mask_weights = CN()
cfg.mask_weights.face = 150.0
cfg.mask_weights.nose = 50.0
cfg.mask_weights.lips = 50.0
cfg.mask_weights.forehead = 50.0
cfg.mask_weights.lr_eye_region = 50.0
cfg.mask_weights.eye_region = 50.0

cfg.mask_weights.whole = 1.0
cfg.mask_weights.ears = 0.01
cfg.mask_weights.eyes = 0.01

cfg.running_average = 7

cfg.train = CN()
cfg.train.use_mask = False
cfg.train.max_epochs = 50
cfg.train.max_steps = 100000
cfg.train.lr = 1e-4
cfg.train.arcface_lr = 1e-3
cfg.train.weight_decay = 0.0
cfg.train.lr_update_step = 100000000
cfg.train.log_dir = "logs"
cfg.train.log_steps = 10
cfg.train.vis_dir = "train_images"
cfg.train.vis_steps = 200
cfg.train.write_summary = True
cfg.train.checkpoint_steps = 1000
cfg.train.checkpoint_epochs_steps = 2
cfg.train.val_steps = 1000
cfg.train.val_vis_dir = "val_images"
cfg.train.eval_steps = 5000
cfg.train.reset_optimizer = False
cfg.train.val_save_img = 5000
cfg.test_dataset = "now"

# ---------------------------------------------------------------------------- #

cfg.flame_geom_path = "./data/FLAME2020/generic_model.pkl"
cfg.flame_template_path = "./data/uv_template.obj"
cfg.flame_lmk_path = "./data/landmark_embedding.npy"
cfg.tex_space_path = "./data/FLAME2020/FLAME_texture.npz"

cfg.num_shape_params = 300
cfg.num_exp_params = 100
cfg.tex_params = 140
cfg.actor = ""
cfg.config_name = ""
cfg.kernel_size = 7
cfg.sigma = 9.0
cfg.keyframes = [0, 1]
cfg.bbox_scale = 2.5
cfg.fps = 25
cfg.begin_frames = 1
cfg.end_frames = 0
cfg.image_size = [512, 512]  # height, width
# cfg.image_size = [256, 256]  # height, width
cfg.rotation_lr = 0.2
cfg.translation_lr = 0.003
cfg.raster_update = 8
cfg.pyr_levels = [
    [1.0, 160],
    [0.25, 40],
    [0.5, 40],
    [1.0, 70],
]  # Gaussian pyramid levels (scaling, iters per level) first level is only the sparse term!
# cfg.pyr_levels = [[1.0, 160], [0.5, 40]]  # Gaussian pyramid levels (scaling, iters per level) first level is only the sparse term!
cfg.optimize_shape = True
cfg.optimize_jaw = True
cfg.crop_image = True
cfg.save_folder = "./output/"

# Weights
cfg.w_pho = 350
cfg.w_lmks = 7000
cfg.w_lmks_68 = 1000
cfg.w_lmks_lid = 1000
cfg.w_lmks_mouth = 15000
cfg.w_lmks_iris = 1000
cfg.w_lmks_oval = 2000

cfg.w_exp = 0.02
cfg.w_shape = 0.3
cfg.w_tex = 0.04
cfg.w_jaw = 0.05


def get_cfg_defaults():
    return cfg.clone()


def update_cfg(cfg, cfg_file):
    cfg.merge_from_file(cfg_file)
    return cfg.clone()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="Configuration file", required=True)

    args = parser.parse_args()
    print(args, end="\n\n")

    cfg = get_cfg_defaults()
    if args.cfg is not None:
        cfg_file = args.cfg
        cfg = update_cfg(cfg, args.cfg)
        cfg.cfg_file = cfg_file

    cfg.config_name = Path(args.cfg).stem

    return cfg


def parse_cfg(cfg_file):
    cfg = get_cfg_defaults()
    cfg = update_cfg(cfg, cfg_file)
    cfg.cfg_file = cfg_file

    cfg.config_name = Path(cfg_file).stem

    return cfg
