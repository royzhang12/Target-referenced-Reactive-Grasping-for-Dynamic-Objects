import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
import math
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import pointnet2_utils

from einops import repeat

def get_loss_refine(end_points, criterion):
    # criterion = nn.L1Loss()
    loss, end_points = compute_tf_loss(end_points ,criterion)

    end_points['loss/overall_loss'] = loss
    return loss, end_points

def compute_gt(end_points):

    selected_grasps = end_points['selected_grasps']
    rot_pred = end_points['rot_pred']
    trans_pred = end_points['translation_pred']

    b, c, n = selected_grasps.shape    
    b//=5
    selected_grasps = selected_grasps.view(b, 5, c, n).permute(0, 3, 1, 2).contiguous()

    

    object_poses = end_points['object_poses']
    cloud_seg = end_points['cloud_segs']

    seed_idxs = end_points['seed_idxs'].to(torch.int32)
    B, L, _ = seed_idxs.shape
    seed_idxs = seed_idxs.view(-1, _)

    seed_inds = end_points['fp2_inds']

    seed_idxs = pointnet2_utils.gather_operation(seed_inds.unsqueeze(1).to(torch.float32), seed_idxs).long().squeeze(1)

    object_transform = end_points['object_transform']
    object_indices = end_points['object_indices']
    # print (len(object_indices), object_indices[0])
    # print (object_poses[0][10])

    assert len(object_indices) == len(object_transform)

    for batch_id in range(b):
        for i in range(5):
            indices = object_indices[batch_id*5 + i][0].cpu().item()
            # print (object_transform[batch_id*5 + i].shape)
            object_transform[batch_id*5 + i][0] = object_transform[batch_id*5 + i][indices]
            # print (indices)

    obj_transforms =[]
    obj_poses = []
    for batch_id in range(b):
        obj_transform = []
        obj_pose = []
        for i in range(5):
            seed_seg = cloud_seg[batch_id*5 + 0]
            seed_seg_ = seed_seg[seed_idxs[batch_id*5 + 0]]

            object_transform_ = object_transform[batch_id*5 + i]
            object_transform_ = object_transform_[seed_seg_]

            obj_pose_ = object_poses[batch_id*5 + i]
            obj_pose_ = obj_pose_[seed_seg_]

            obj_transform.append(object_transform_)
            obj_pose.append(obj_pose_)
        obj_transform = torch.stack(obj_transform, dim=0)
        obj_transforms.append(obj_transform)
        obj_pose = torch.stack(obj_pose, dim=0)
        obj_poses.append(obj_pose)
    obj_transforms = torch.stack(obj_transforms, dim=0)
    obj_poses = torch.stack(obj_poses, dim=0)

    grasps_rot = selected_grasps[:, :, :, 3:].contiguous().view(b, n, 5, 3, 3)
    grasps_translation = selected_grasps[:, :, :, :3].unsqueeze(-1)
    grasps = torch.cat([grasps_rot, grasps_translation], dim=-1)
    
    to_stack = torch.tensor([0., 0., 0., 1])
    to_stack = repeat(to_stack, 'c -> b n l c', b=grasps.shape[0], n=grasps.shape[1], l=grasps.shape[2]).unsqueeze(-2)
        
    grasps = torch.cat([grasps, to_stack.to(grasps.device)], dim=-2)
        
    select_grasp_first_frame = grasps[:, :, 0]
    select_grasp_first_frame = repeat(select_grasp_first_frame, 'b n c m -> b n l c m', l=5)

    first_frame_transformed_poses = torch.matmul(obj_transforms.permute(0,2,1,3,4).contiguous().to(select_grasp_first_frame.dtype), 
                        select_grasp_first_frame)
    
    first_frame_rot_tf = first_frame_transformed_poses[:, :, 1:, :3, :3]
    first_frame_trans_tf = first_frame_transformed_poses[:, :, 1:, :3, -1]

    rot_pred = rot_pred.contiguous().view(b, n, 4, 3, 3)
    trans_pred = trans_pred.contiguous().view(b, n, 4, 3)
    return end_points, first_frame_rot_tf.detach(), first_frame_trans_tf.detach(), rot_pred, trans_pred


def compute_loss(gt, pred, criterion):

    if len(gt.shape) == 5:
        b, n, l, a, c = gt.shape
    else:
        b, n, l, a = gt.shape

    pred = pred.contiguous().view(b*n, l, -1)
    gt = gt.contiguous().view(b*n, l, -1).detach()

    loss = criterion(pred, gt)

    return loss

def compute_tf_loss(end_points, criterion):

    end_points, rot_gt, trans_gt, rot_pred, trans_pred = compute_gt(end_points)

    loss1 = compute_loss(rot_gt, rot_pred, criterion)
    loss2 = compute_loss(trans_gt, trans_pred, criterion)

    end_points['losses/rot_correspondence_loss'] = loss1
    end_points['losses/trans_correspondence_loss'] = loss2
    loss = loss1 + loss2	

    return loss, end_points