# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import os
import sys
import time
from shutil import copy, copytree

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data.nuscenes_dataloader import TrainDatasetMultiSeq
from model import MotionNet


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {avg" + self.fmt + "}"
        return fmtstr.format(**self.__dict__)


def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path


use_weighted_loss = True  # Whether to set different weights for different grid cell categories for loss computation
pred_adj_frame_distance = True  # Whether to predict the relative offset between frames

height_feat_size = 13  # The size along the height dimension
cell_category_num = 5  # The number of object categories (including the background)

out_seq_len = 20  # The number of future frames we are going to predict
trans_matrix_idx = (
    1  # Among N transformation matrices (N=2 in our experiment), which matrix is used for alignment (see paper)
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d", "--data", default=None, type=str, help="The path to the preprocessed sparse BEV training data"
)
parser.add_argument(
    "--resume", default="", type=str, help="The path to the saved model that is loaded to resume training"
)
parser.add_argument("--batch", default=8, type=int, help="Batch size")
parser.add_argument("--nepoch", default=45, type=int, help="Number of epochs")
parser.add_argument("--nworker", default=4, type=int, help="Number of workers")

parser.add_argument(
    "--reg_weight_bg_tc", default=0.1, type=float, help="Weight of background temporal consistency term"
)
parser.add_argument("--reg_weight_fg_tc", default=2.5, type=float, help="Weight of instance temporal consistency")
parser.add_argument("--reg_weight_sc", default=15.0, type=float, help="Weight of spatial consistency term")

parser.add_argument("--use_bg_tc", action="store_true", help="Whether to use background temporal consistency loss")
parser.add_argument("--use_fg_tc", action="store_true", help="Whether to use foreground loss in st.")
parser.add_argument("--use_sc", action="store_true", help="Whether to use spatial consistency loss")

parser.add_argument("--nn_sampling", action="store_true", help="Whether to use nearest neighbor sampling in bg_tc loss")
parser.add_argument("--log", action="store_true", help="Whether to log")
parser.add_argument("--logpath", default="", help="The path to the output log file")

args = parser.parse_args()
print(args)

need_log = args.log
BATCH_SIZE = args.batch
num_epochs = args.nepoch
num_workers = args.nworker

reg_weight_bg_tc = args.reg_weight_bg_tc  # The weight of background temporal consistency term
reg_weight_fg_tc = args.reg_weight_fg_tc  # The weight of foreground temporal consistency term
reg_weight_sc = args.reg_weight_sc  # The weight of spatial consistency term

use_bg_temporal_consistency = args.use_bg_tc
use_fg_temporal_consistency = args.use_fg_tc
use_spatial_consistency = args.use_sc

use_nn_sampling = args.nn_sampling


def main():
    start_epoch = 1
    # Whether to log the training information
    if need_log:
        logger_root = args.logpath if args.logpath != "" else "logs"
        time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")

        if args.resume == "":
            model_save_path = check_folder(logger_root)
            model_save_path = check_folder(os.path.join(model_save_path, "train_multi_seq"))
            model_save_path = check_folder(os.path.join(model_save_path, time_stamp))

            log_file_name = os.path.join(model_save_path, "log.txt")
            saver = open(log_file_name, "w")
            saver.write("GPU number: {}\n".format(torch.cuda.device_count()))
            saver.flush()

            # Logging the details for this experiment
            saver.write("command line: {}\n".format(" ".join(sys.argv[0:])))
            saver.write(args.__repr__() + "\n\n")
            saver.flush()

            # Copy the code files as logs
            copytree("nuscenes-devkit", os.path.join(model_save_path, "nuscenes-devkit"))
            copytree("data", os.path.join(model_save_path, "data"))
            python_files = [f for f in os.listdir(".") if f.endswith(".py")]
            for f in python_files:
                copy(f, model_save_path)
        else:
            model_save_path = args.resume  # eg, "logs/train_multi_seq/1234-56-78-11-22-33"

            log_file_name = os.path.join(model_save_path, "log.txt")
            saver = open(log_file_name, "a")
            saver.write("GPU number: {}\n".format(torch.cuda.device_count()))
            saver.flush()

            # Logging the details for this experiment
            saver.write("command line: {}\n".format(" ".join(sys.argv[1:])))
            saver.write(args.__repr__() + "\n\n")
            saver.flush()

    # Specify gpu device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_num = torch.cuda.device_count()
    print("device number", device_num)

    voxel_size = (0.25, 0.25, 0.4)
    area_extents = np.array([[-32.0, 32.0], [-32.0, 32.0], [-3.0, 2.0]])

    trainset = TrainDatasetMultiSeq(
        dataset_root=args.data,
        future_frame_skip=0,
        voxel_size=voxel_size,
        area_extents=area_extents,
        num_category=cell_category_num,
    )

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    print("Training dataset size:", len(trainset))

    model = MotionNet(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size)
    model = nn.DataParallel(model)
    model = model.to(device)

    if use_weighted_loss:
        criterion = nn.SmoothL1Loss(reduction="none")
    else:
        criterion = nn.SmoothL1Loss(reduction="sum")
    optimizer = optim.Adam(model.parameters(), lr=0.0016)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40], gamma=0.5)

    if args.resume != "":
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        print("Load model from {}, at epoch {}".format(args.resume, start_epoch - 1))

    for epoch in range(start_epoch, num_epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        print("Epoch {}, learning rate {}".format(epoch, lr))

        if need_log:
            saver.write("epoch: {}, lr: {}\t".format(epoch, lr))
            saver.flush()

        scheduler.step()
        model.train()

        loss_disp, loss_class, loss_motion, loss_bg_tc, loss_sc, loss_fg_tc = train(
            model, criterion, trainloader, optimizer, device, epoch
        )

        if need_log:
            saver.write(
                "{}\t{}\t{}\t{}\t{}\t{}\n".format(loss_disp, loss_class, loss_motion, loss_bg_tc, loss_fg_tc, loss_sc)
            )
            saver.flush()

        # save model
        if need_log and (epoch % 5 == 0 or epoch == num_epochs or epoch == 1 or epoch > 20):
            save_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": loss_disp.avg,
            }
            torch.save(save_dict, os.path.join(model_save_path, "epoch_" + str(epoch) + ".pth"))

    if need_log:
        saver.close()


def train(model, criterion, trainloader, optimizer, device, epoch):
    running_loss_bg_tc = AverageMeter("bg_tc", ":.7f")  # background temporal consistency error
    running_loss_fg_tc = AverageMeter("fg_tc", ":.7f")  # foreground temporal consistency error
    running_loss_sc = AverageMeter("sc", ":.7f")  # spatial consistency error
    running_loss_disp = AverageMeter("Disp", ":.6f")  # for motion prediction error
    running_loss_class = AverageMeter("Obj_Cls", ":.6f")  # for cell classification error
    running_loss_motion = AverageMeter("Motion_Cls", ":.6f")  # for state estimation error

    for i, data in enumerate(trainloader, 0):
        (
            padded_voxel_points,
            all_disp_field_gt,
            all_valid_pixel_maps,
            non_empty_map,
            pixel_cat_map_gt,
            trans_matrices,
            motion_gt,
            pixel_instance_map,
            num_past_frames,
            num_future_frames,
        ) = data

        # Move to GPU/CPU
        padded_voxel_points = padded_voxel_points.view(-1, num_past_frames[0].item(), 256, 256, height_feat_size)
        padded_voxel_points = padded_voxel_points.to(device)

        # Make prediction
        disp_pred, class_pred, motion_pred = model(padded_voxel_points)

        # Compute and back-propagate the losses
        loss_disp, loss_class, loss_motion, loss_bg_tc, loss_sc, loss_fg_tc = compute_and_bp_loss(
            optimizer,
            device,
            num_future_frames[0].item(),
            all_disp_field_gt,
            all_valid_pixel_maps,
            pixel_cat_map_gt,
            disp_pred,
            criterion,
            non_empty_map,
            class_pred,
            motion_gt,
            motion_pred,
            trans_matrices,
            pixel_instance_map,
        )

        if not all((loss_disp, loss_class, loss_motion)):
            print(
                "{}, \t{}, \tat epoch {}, \titerations {} [empty occupy map]".format(
                    running_loss_disp, running_loss_class, epoch, i
                )
            )
            continue

        running_loss_bg_tc.update(loss_bg_tc)
        running_loss_fg_tc.update(loss_fg_tc)
        running_loss_sc.update(loss_sc)
        running_loss_disp.update(loss_disp)
        running_loss_class.update(loss_class)
        running_loss_motion.update(loss_motion)
        print(
            "[{}/{}]\t{}, \t{}, \t{}, \t{}, \t{}, \t{}".format(
                epoch,
                i,
                running_loss_disp,
                running_loss_class,
                running_loss_motion,
                running_loss_bg_tc,
                running_loss_sc,
                running_loss_fg_tc,
            )
        )

    return (
        running_loss_disp,
        running_loss_class,
        running_loss_motion,
        running_loss_bg_tc,
        running_loss_sc,
        running_loss_fg_tc,
    )


# Compute and back-propagate the loss
def compute_and_bp_loss(
    optimizer,
    device,
    future_frames_num,
    all_disp_field_gt,
    all_valid_pixel_maps,
    pixel_cat_map_gt,
    disp_pred,
    criterion,
    non_empty_map,
    class_pred,
    motion_gt,
    motion_pred,
    trans_matrices,
    pixel_instance_map,
):
    optimizer.zero_grad()

    # Compute the displacement loss
    all_disp_field_gt = all_disp_field_gt.view(-1, future_frames_num, 256, 256, 2)
    gt = all_disp_field_gt[:, -future_frames_num:, ...].contiguous()
    gt = gt.view(-1, gt.size(2), gt.size(3), gt.size(4))
    gt = gt.permute(0, 3, 1, 2).to(device)

    all_valid_pixel_maps = all_valid_pixel_maps.view(-1, future_frames_num, 256, 256)
    valid_pixel_maps = all_valid_pixel_maps[:, -future_frames_num:, ...].contiguous()
    valid_pixel_maps = valid_pixel_maps.view(-1, valid_pixel_maps.size(2), valid_pixel_maps.size(3))
    valid_pixel_maps = torch.unsqueeze(valid_pixel_maps, 1)
    valid_pixel_maps = valid_pixel_maps.to(device)

    valid_pixel_num = torch.nonzero(valid_pixel_maps).size(0)
    if valid_pixel_num == 0:
        return [None] * 6

    # ---------------------------------------------------------------------
    # -- Generate the displacement w.r.t. the keyframe
    if pred_adj_frame_distance:
        disp_pred = disp_pred.view(-1, future_frames_num, disp_pred.size(-3), disp_pred.size(-2), disp_pred.size(-1))

        # Compute temporal consistency loss
        if use_bg_temporal_consistency:
            bg_tc_loss = background_temporal_consistency_loss(
                disp_pred, pixel_cat_map_gt, non_empty_map, trans_matrices
            )

        if use_fg_temporal_consistency or use_spatial_consistency:
            (
                instance_spatio_temp_loss,
                instance_spatial_loss_value,
                instance_temporal_loss_value,
            ) = instance_spatial_temporal_consistency_loss(disp_pred, pixel_instance_map)

        for c in range(1, disp_pred.size(1)):
            disp_pred[:, c, ...] = disp_pred[:, c, ...] + disp_pred[:, c - 1, ...]
        disp_pred = disp_pred.view(-1, disp_pred.size(-3), disp_pred.size(-2), disp_pred.size(-1))

    # ---------------------------------------------------------------------
    # -- Compute the masked displacement loss
    pixel_cat_map_gt = pixel_cat_map_gt.view(-1, 256, 256, cell_category_num)

    if use_weighted_loss:  # Note: have also tried focal loss, but did not observe noticeable improvement
        pixel_cat_map_gt_numpy = pixel_cat_map_gt.numpy()
        pixel_cat_map_gt_numpy = np.argmax(pixel_cat_map_gt_numpy, axis=-1) + 1
        cat_weight_map = np.zeros_like(pixel_cat_map_gt_numpy, dtype=np.float32)
        weight_vector = [0.005, 1.0, 1.0, 1.0, 1.0]  # [bg, car & bus, ped, bike, other]
        for k in range(len(weight_vector)):
            mask = pixel_cat_map_gt_numpy == (k + 1)
            cat_weight_map[mask] = weight_vector[k]

        cat_weight_map = cat_weight_map[:, np.newaxis, np.newaxis, ...]  # (batch, 1, 1, h, w)
        cat_weight_map = torch.from_numpy(cat_weight_map).to(device)
        map_shape = cat_weight_map.size()

        loss_disp = criterion(gt * valid_pixel_maps, disp_pred * valid_pixel_maps)
        loss_disp = loss_disp.view(map_shape[0], -1, map_shape[-3], map_shape[-2], map_shape[-1])
        loss_disp = torch.sum(loss_disp * cat_weight_map) / valid_pixel_num
    else:
        loss_disp = criterion(gt * valid_pixel_maps, disp_pred * valid_pixel_maps) / valid_pixel_num

    # ---------------------------------------------------------------------
    # -- Compute the grid cell classification loss
    non_empty_map = non_empty_map.view(-1, 256, 256)
    non_empty_map = non_empty_map.to(device)
    pixel_cat_map_gt = pixel_cat_map_gt.permute(0, 3, 1, 2).to(device)

    log_softmax_probs = F.log_softmax(class_pred, dim=1)

    if use_weighted_loss:
        map_shape = cat_weight_map.size()
        cat_weight_map = cat_weight_map.view(map_shape[0], map_shape[-2], map_shape[-1])  # (bs, h, w)
        loss_class = torch.sum(-pixel_cat_map_gt * log_softmax_probs, dim=1) * cat_weight_map
    else:
        loss_class = torch.sum(-pixel_cat_map_gt * log_softmax_probs, dim=1)
    loss_class = torch.sum(loss_class * non_empty_map) / torch.nonzero(non_empty_map).size(0)

    # ---------------------------------------------------------------------
    # -- Compute the speed level classification loss
    motion_gt = motion_gt.view(-1, 256, 256, 2)
    motion_gt_numpy = motion_gt.numpy()
    motion_gt = motion_gt.permute(0, 3, 1, 2).to(device)
    log_softmax_motion_pred = F.log_softmax(motion_pred, dim=1)

    if use_weighted_loss:
        motion_gt_numpy = np.argmax(motion_gt_numpy, axis=-1) + 1
        motion_weight_map = np.zeros_like(motion_gt_numpy, dtype=np.float32)
        weight_vector = [0.005, 1.0]  # [static, moving]
        for k in range(len(weight_vector)):
            mask = motion_gt_numpy == (k + 1)
            motion_weight_map[mask] = weight_vector[k]

        motion_weight_map = torch.from_numpy(motion_weight_map).to(device)
        loss_speed = torch.sum(-motion_gt * log_softmax_motion_pred, dim=1) * motion_weight_map
    else:
        loss_speed = torch.sum(-motion_gt * log_softmax_motion_pred, dim=1)
    loss_motion = torch.sum(loss_speed * non_empty_map) / torch.nonzero(non_empty_map).size(0)

    # ---------------------------------------------------------------------
    # -- Sum up all the losses
    if use_bg_temporal_consistency and (use_fg_temporal_consistency or use_spatial_consistency):
        loss = loss_disp + loss_class + loss_motion + reg_weight_bg_tc * bg_tc_loss + instance_spatio_temp_loss
    elif use_bg_temporal_consistency:
        loss = loss_disp + loss_class + loss_motion + reg_weight_bg_tc * bg_tc_loss
    elif use_spatial_consistency or use_fg_temporal_consistency:
        loss = loss_disp + loss_class + loss_motion + instance_spatio_temp_loss
    else:
        loss = loss_disp + loss_class + loss_motion
    loss.backward()
    optimizer.step()

    if use_bg_temporal_consistency:
        bg_tc_loss_value = bg_tc_loss.item()
    else:
        bg_tc_loss_value = -1

    if use_spatial_consistency or use_fg_temporal_consistency:
        sc_loss_value = instance_spatial_loss_value
        fg_tc_loss_value = instance_temporal_loss_value
    else:
        sc_loss_value = -1
        fg_tc_loss_value = -1

    return loss_disp.item(), loss_class.item(), loss_motion.item(), bg_tc_loss_value, sc_loss_value, fg_tc_loss_value


def background_temporal_consistency_loss(disp_pred, pixel_cat_map_gt, non_empty_map, trans_matrices):
    """
    disp_pred: Should be relative displacement between adjacent frames. shape (batch * 2, sweep_num, 2, h, w)
    pixel_cat_map_gt: Shape (batch, 2, h, w, cat_num)
    non_empty_map: Shape (batch, 2, h, w)
    trans_matrices: Shape (batch, 2, sweep_num, 4, 4)
    """
    criterion = nn.SmoothL1Loss(reduction="sum")

    non_empty_map_numpy = non_empty_map.numpy()
    pixel_cat_maps = pixel_cat_map_gt.numpy()
    max_prob = np.amax(pixel_cat_maps, axis=-1)
    filter_mask = max_prob == 1.0
    pixel_cat_maps = np.argmax(pixel_cat_maps, axis=-1) + 1  # category starts from 1 (background), etc
    pixel_cat_maps = pixel_cat_maps * non_empty_map_numpy * filter_mask  # (batch, 2, h, w)

    trans_matrices = trans_matrices.numpy()
    device = disp_pred.device

    pred_shape = disp_pred.size()
    disp_pred = disp_pred.view(-1, 2, pred_shape[1], pred_shape[2], pred_shape[3], pred_shape[4])

    seq_1_pred = disp_pred[:, 0]  # (batch, sweep_num, 2, h, w)
    seq_2_pred = disp_pred[:, 1]

    seq_1_absolute_pred_list = list()
    seq_2_absolute_pred_list = list()

    seq_1_absolute_pred_list.append(seq_1_pred[:, 1])
    for i in range(2, pred_shape[1]):
        seq_1_absolute_pred_list.append(seq_1_pred[:, i] + seq_1_absolute_pred_list[i - 2])

    seq_2_absolute_pred_list.append(seq_2_pred[:, 0])
    for i in range(1, pred_shape[1] - 1):
        seq_2_absolute_pred_list.append(seq_2_pred[:, i] + seq_2_absolute_pred_list[i - 1])

    # ----------------- Compute the consistency loss -----------------
    # Compute the transformation matrices
    # First, transform the coordinate
    transformed_disp_pred_list = list()

    trans_matrix_global = trans_matrices[:, 1]  # (batch, sweep_num, 4, 4)
    trans_matrix_global = trans_matrix_global[:, trans_matrix_idx, 0:3]  # (batch, 3, 4)  # <---
    trans_matrix_global = trans_matrix_global[:, :, (0, 1, 3)]  # (batch, 3, 3)
    trans_matrix_global[:, 2] = np.array([0.0, 0.0, 1.0])

    # --- Move pixel coord to global and rescale; then rotate; then move back to local pixel coord
    translate_to_global = np.array([[1.0, 0.0, -120.0], [0.0, 1.0, -120.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    scale_global = np.array([[0.25, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    trans_global = scale_global @ translate_to_global
    inv_trans_global = np.linalg.inv(trans_global)

    trans_global = np.expand_dims(trans_global, axis=0)
    inv_trans_global = np.expand_dims(inv_trans_global, axis=0)
    trans_matrix_total = inv_trans_global @ trans_matrix_global @ trans_global

    # --- Generate grid transformation matrix, so as to use Pytorch affine_grid and grid_sample function
    w, h = pred_shape[-2], pred_shape[-1]
    resize_m = np.array([[2 / w, 0.0, -1], [0.0, 2 / h, -1], [0.0, 0.0, 1]], dtype=np.float32)
    inverse_m = np.linalg.inv(resize_m)
    resize_m = np.expand_dims(resize_m, axis=0)
    inverse_m = np.expand_dims(inverse_m, axis=0)

    grid_trans_matrix = resize_m @ trans_matrix_total @ inverse_m  # (batch, 3, 3)
    grid_trans_matrix = grid_trans_matrix[:, :2].astype(np.float32)
    grid_trans_matrix = torch.from_numpy(grid_trans_matrix)

    # --- For displacement field
    trans_matrix_translation_global = np.eye(trans_matrix_total.shape[1])
    trans_matrix_translation_global = np.expand_dims(trans_matrix_translation_global, axis=0)
    trans_matrix_translation_global = np.repeat(trans_matrix_translation_global, grid_trans_matrix.shape[0], axis=0)
    trans_matrix_translation_global[:, :, 2] = trans_matrix_global[:, :, 2]  # only translation
    trans_matrix_translation_total = inv_trans_global @ trans_matrix_translation_global @ trans_global

    grid_trans_matrix_disp = resize_m @ trans_matrix_translation_total @ inverse_m
    grid_trans_matrix_disp = grid_trans_matrix_disp[:, :2].astype(np.float32)
    grid_trans_matrix_disp = torch.from_numpy(grid_trans_matrix_disp).to(device)

    disp_rotate_matrix = trans_matrix_global[:, 0:2, 0:2].astype(np.float32)  # (batch, 2, 2)
    disp_rotate_matrix = torch.from_numpy(disp_rotate_matrix).to(device)

    for i in range(len(seq_1_absolute_pred_list)):

        # --- Start transformation for displacement field
        curr_pred = seq_1_absolute_pred_list[i]  # (batch, 2, h, w)

        # First, rotation
        curr_pred = curr_pred.permute(0, 2, 3, 1).contiguous()  # (batch, h, w, 2)
        curr_pred = curr_pred.view(-1, h * w, 2)
        curr_pred = torch.bmm(curr_pred, disp_rotate_matrix)
        curr_pred = curr_pred.view(-1, h, w, 2)
        curr_pred = curr_pred.permute(0, 3, 1, 2).contiguous()  # (batch, 2, h, w)

        # Next, translation
        curr_pred = curr_pred.permute(0, 1, 3, 2).contiguous()  # swap x and y axis
        curr_pred = torch.flip(curr_pred, dims=[2])

        grid = F.affine_grid(grid_trans_matrix_disp, curr_pred.size())
        if use_nn_sampling:
            curr_pred = F.grid_sample(curr_pred, grid, mode="nearest")
        else:
            curr_pred = F.grid_sample(curr_pred, grid)

        curr_pred = torch.flip(curr_pred, dims=[2])
        curr_pred = curr_pred.permute(0, 1, 3, 2).contiguous()

        transformed_disp_pred_list.append(curr_pred)

    # --- Start transformation for category map
    pixel_cat_map = pixel_cat_maps[:, 0]  # (batch, h, w)
    pixel_cat_map = torch.from_numpy(pixel_cat_map.astype(np.float32))
    pixel_cat_map = pixel_cat_map[:, None, :, :]  # (batch, 1, h, w)
    trans_pixel_cat_map = pixel_cat_map.permute(0, 1, 3, 2)  # (batch, 1, h, w), swap x and y axis
    trans_pixel_cat_map = torch.flip(trans_pixel_cat_map, dims=[2])

    grid = F.affine_grid(grid_trans_matrix, pixel_cat_map.size())
    trans_pixel_cat_map = F.grid_sample(trans_pixel_cat_map, grid, mode="nearest")

    trans_pixel_cat_map = torch.flip(trans_pixel_cat_map, dims=[2])
    trans_pixel_cat_map = trans_pixel_cat_map.permute(0, 1, 3, 2)

    # --- Compute the loss, using smooth l1 loss
    adj_pixel_cat_map = pixel_cat_maps[:, 1]
    adj_pixel_cat_map = torch.from_numpy(adj_pixel_cat_map.astype(np.float32))
    adj_pixel_cat_map = torch.unsqueeze(adj_pixel_cat_map, dim=1)

    mask_common = trans_pixel_cat_map == adj_pixel_cat_map
    mask_common = mask_common.float()
    non_empty_map_gpu = non_empty_map.to(device)
    non_empty_map_gpu = non_empty_map_gpu[:, 1:2, :, :]  # select the second sequence, keep dim
    mask_common = mask_common.to(device)
    mask_common = mask_common * non_empty_map_gpu

    loss_list = list()
    for i in range(len(seq_1_absolute_pred_list)):
        trans_seq_1_pred = transformed_disp_pred_list[i]  # (batch, 2, h, w)
        seq_2_pred = seq_2_absolute_pred_list[i]  # (batch, 2, h, w)

        trans_seq_1_pred = trans_seq_1_pred * mask_common
        seq_2_pred = seq_2_pred * mask_common

        num_non_empty_cells = torch.nonzero(mask_common).size(0)
        if num_non_empty_cells != 0:
            loss = criterion(trans_seq_1_pred, seq_2_pred) / num_non_empty_cells
            loss_list.append(loss)

    res_loss = torch.mean(torch.stack(loss_list, 0))

    return res_loss


# We name it instance spatial-temporal consistency loss because it involves each instance
def instance_spatial_temporal_consistency_loss(disp_pred, pixel_instance_map):
    device = disp_pred.device

    pred_shape = disp_pred.size()
    disp_pred = disp_pred.view(-1, 2, pred_shape[1], pred_shape[2], pred_shape[3], pred_shape[4])

    seq_1_pred = disp_pred[:, 0]  # (batch, sweep_num, 2, h, w)
    seq_2_pred = disp_pred[:, 1]

    pixel_instance_map = pixel_instance_map.numpy()
    batch = pixel_instance_map.shape[0]

    spatial_loss = 0.0
    temporal_loss = 0.0
    counter = 0
    criterion = nn.SmoothL1Loss()

    for i in range(batch):
        curr_batch_instance_maps = pixel_instance_map[i]
        seq_1_instance_map = curr_batch_instance_maps[0]
        seq_2_instance_map = curr_batch_instance_maps[1]

        seq_1_instance_ids = np.unique(seq_1_instance_map)
        seq_2_instance_ids = np.unique(seq_2_instance_map)

        common_instance_ids = np.intersect1d(seq_1_instance_ids, seq_2_instance_ids, assume_unique=True)

        seq_1_batch_pred = seq_1_pred[i]  # (sweep_num, 2, h, w)
        seq_2_batch_pred = seq_2_pred[i]

        for h in common_instance_ids:
            if h == 0:  # do not consider the background instance
                continue

            seq_1_mask = np.where(seq_1_instance_map == h)
            seq_1_idx_x = torch.from_numpy(seq_1_mask[0]).to(device)
            seq_1_idx_y = torch.from_numpy(seq_1_mask[1]).to(device)
            seq_1_selected_cells = seq_1_batch_pred[:, :, seq_1_idx_x, seq_1_idx_y]

            seq_2_mask = np.where(seq_2_instance_map == h)
            seq_2_idx_x = torch.from_numpy(seq_2_mask[0]).to(device)
            seq_2_idx_y = torch.from_numpy(seq_2_mask[1]).to(device)
            seq_2_selected_cells = seq_2_batch_pred[:, :, seq_2_idx_x, seq_2_idx_y]

            seq_1_selected_cell_num = seq_1_selected_cells.size(2)
            seq_2_selected_cell_num = seq_2_selected_cells.size(2)

            # for spatial loss
            if use_spatial_consistency:
                tmp_seq_1 = 0
                if seq_1_selected_cell_num > 1:
                    tmp_seq_1 = criterion(seq_1_selected_cells[:, :, :-1], seq_1_selected_cells[:, :, 1:])

                tmp_seq_2 = 0
                if seq_2_selected_cell_num > 1:
                    tmp_seq_2 = criterion(seq_2_selected_cells[:, :, :-1], seq_2_selected_cells[:, :, 1:])

                spatial_loss += tmp_seq_1 + tmp_seq_2

            if use_fg_temporal_consistency:
                seq_1_mean = torch.mean(seq_1_selected_cells, dim=2)
                seq_2_mean = torch.mean(seq_2_selected_cells, dim=2)
                temporal_loss += criterion(seq_1_mean, seq_2_mean)

            counter += 1

    if counter != 0:
        spatial_loss = spatial_loss / counter
        temporal_loss = temporal_loss / counter

    total_loss = reg_weight_sc * spatial_loss + reg_weight_fg_tc * temporal_loss

    spatial_loss_value = 0 if type(spatial_loss) == float else spatial_loss.item()
    temporal_loss_value = 0 if type(temporal_loss) == float else temporal_loss.item()

    return total_loss, spatial_loss_value, temporal_loss_value


if __name__ == "__main__":
    main()
