import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from loss_utils import GRASP_MAX_WIDTH, GRASP_MAX_TOLERANCE, THRESH_GOOD, THRESH_BAD,\
                       transform_point_cloud, generate_grasp_views,\
                       batch_viewpoint_params_to_matrix, huber_loss

def get_loss(end_points):
    loss, end_points = compute_correspondence_loss(end_points)
    end_points['loss/overall_loss'] = loss
    return loss, end_points

def compute_gt_grasp_correspondence(end_points):
    grasp_preds = end_points['batch_grasp_preds']
    seed_inds = end_points['fp2_inds']

    B, Ns, _ = grasp_preds.size()
    B //= 2
    grasp_preds = grasp_preds.view(B, 2, Ns, 17)
    seed_inds = seed_inds.view(B, 2, seed_inds.shape[1]).long()

    grasp_trans_1, grasp_trans_2 = grasp_preds[:,0,:,13:16], grasp_preds[:,1,:,13:16] #(B, Ns, 3)
    grasp_rot_1, grasp_rot_2 = grasp_preds[:,0,:,4:13], grasp_preds[:,1,:,4:13] # (B, Ns, 9)

    seed_poses1 = []
    seed_poses2 = []
    seed_segs1 = []
    seed_segs2 = []
    for batch_id in range(B):
        cloud_seg1, cloud_seg2 = end_points['cloud_segs'][batch_id*2], end_points['cloud_segs'][batch_id*2+1] # (point_num, )
        object_poses1, object_poses2 = end_points["object_poses"][batch_id*2], end_points["object_poses"][batch_id*2+1]  # (89, 4, 4)
        batch_seed_inds1 = seed_inds[batch_id][0]  # (Ns, )
        batch_seed_inds2 = seed_inds[batch_id][1]  # (Ns, )

        batch_seed_segs1 = cloud_seg1[batch_seed_inds1]  # (Ns, )
        batch_seed_segs2 = cloud_seg2[batch_seed_inds2]  # (Ns, )

        batch_seed_poses1 = object_poses1[batch_seed_segs1]  # (Ns, 4, 4)
        batch_seed_poses2 = object_poses2[batch_seed_segs2]  # (Ns, 4, 4)

        seed_poses1.append(batch_seed_poses1)
        seed_poses2.append(batch_seed_poses2)

        seed_segs1.append(batch_seed_segs1)
        seed_segs2.append(batch_seed_segs2)

    # print (seed_segs1)

    seed_poses1 = torch.stack(seed_poses1, dim=0).contiguous()  # (B, Ns, 4, 4)
    seed_poses2 = torch.stack(seed_poses2, dim=0).contiguous()  # (B, Ns, 4, 4)

    seed_segs1 = torch.stack(seed_segs1, dim=0)  # (B, Ns)
    seed_segs2 = torch.stack(seed_segs2, dim=0)  # (B, Ns)

    seed_segs1 = seed_segs1.unsqueeze(2)  # (B, Ns, 1)
    seed_segs2 = seed_segs2.unsqueeze(1)  # (B, 1, Ns)
    seg_correspondence = torch.eq(seed_segs1, seed_segs2)  # (B, Ns, Ns)
    seg_correspondence = seg_correspondence & (seed_segs1 > 0)
    end_points['batch_grasp_seg_correspondence'] = seg_correspondence.detach()

    seed_poses_inv1 = torch.inverse(seed_poses1)
    seed_poses_inv2 = torch.inverse(seed_poses2)

    grasp_trans_1 = torch.matmul(seed_poses_inv1[:, :, :3, :3], grasp_trans_1.unsqueeze(-1)).squeeze(-1)
    grasp_trans_1 = grasp_trans_1 + seed_poses_inv1[:, :, :3, 3]

    grasp_trans_2 = torch.matmul(seed_poses_inv2[:, :, :3, :3], grasp_trans_2.unsqueeze(-1)).squeeze(-1)
    grasp_trans_2 = grasp_trans_2 + seed_poses_inv2[:, :, :3, 3]

    ## translation correspondence
    grasp_trans_1 = grasp_trans_1.unsqueeze(2)  # (B, Ns, 1, 3)
    grasp_trans_2 = grasp_trans_2.unsqueeze(1)  # (B, 1, Ns, 3)
    trans_correspondence = torch.norm(grasp_trans_1-grasp_trans_2, dim=3) # (B, Ns, Ns)
    end_points['batch_grasp_trans_correspondence'] = trans_correspondence.detach()

    grasp_rot_1, grasp_rot_2 = grasp_rot_1.view(B, Ns, 3, 3), grasp_rot_2.view(B, Ns, 3, 3)
    grasp_rot_1 = torch.matmul(seed_poses_inv1[:, :, :3, :3], grasp_rot_1)  # (B, Ns, 3, 3)
    grasp_rot_2 = torch.matmul(seed_poses_inv2[:, :, :3, :3], grasp_rot_2)  # (B, Ns, 3, 3)

    ## rotation correspondence
    grasp_rot_1 = grasp_rot_1.unsqueeze(2) # (B, Ns, 1, 3, 3)
    grasp_rot_2 = grasp_rot_2.unsqueeze(1) # (B, 1, Ns, 3, 3)
    rot_correspondence = torch.matmul(grasp_rot_1, grasp_rot_2.transpose(3,4)) # (B, Ns, Ns, 3, 3)
    rot_correspondence = torch.diagonal(rot_correspondence, dim1=3, dim2=4).sum(dim=3) # (B, Ns, Ns)
    rot_correspondence = torch.clamp((rot_correspondence-1)/2, -1, 1) # (B, Ns, Ns)
    rot_correspondence = torch.acos(rot_correspondence) # (B, Ns, Ns)
    end_points['batch_grasp_rot_correspondence'] = rot_correspondence.detach()

    return seg_correspondence, trans_correspondence, rot_correspondence, end_points

def compute_supervised_contrastive_loss(corres_pred, corres_label, training_mask, tau):
    corres_pred_max, _ = torch.max(corres_pred, dim=2, keepdims=True)
    corres_pred = (corres_pred - corres_pred_max) / tau

    all_exp = torch.sum(torch.exp(corres_pred), dim=2, keepdims=True)  # (B, Ns, 1)
    log_exp = torch.log(all_exp + 1e-6) - corres_pred # (B, Ns, Ns)
    pos_cnt = torch.sum(corres_label.float(), dim=2) # (B, Ns)
    corres_loss = torch.sum(log_exp * corres_label, dim=-2) / (pos_cnt + 1e-6) # (B, Ns)

    target_mask = torch.sum(training_mask.float(), dim=2) > 0 # (B, Ns)
    target_mask = (target_mask & (pos_cnt.int() > 0))
    corres_loss = torch.sum(corres_loss * target_mask, dim=1) / (torch.sum(target_mask, dim=1) + 1e-6)
    corres_loss = torch.mean(corres_loss)

    return corres_loss


def compute_correspondence_loss(end_points, gamma=0.1, thre=0.1, tau=0.1, symmetric=False):
    corres_s, corres_t, corres_r, end_points = compute_gt_grasp_correspondence(end_points)
    training_mask = corres_s
    corres_label = (corres_t / GRASP_MAX_WIDTH) + 0.4 * (corres_r / np.pi) # (B, Ns, Ns)


    # corres_label = torch.where(training_mask&(corres_label<thre), \
    #                         torch.ones_like(corres_label), torch.zeros_like(corres_label))  # (B, Ns, Ns)

    # print (torch.nonzero(training_mask&(corres_label<thre)).shape, corres_label.shape)

    corres_label_trans = (corres_t / GRASP_MAX_WIDTH)
    corres_label_rot  = 0.4 * (corres_r / np.pi)
    corres_label = torch.where(training_mask&((corres_label_trans <= 0.3) & (corres_label_rot <= 0.1)), \
                            torch.ones_like(corres_label_trans), torch.zeros_like(corres_label_trans))  # (B, Ns, Ns)

    print (torch.nonzero(training_mask&((corres_label_trans <= 0.3) & (corres_label_rot <= 0.1))).shape)
    corres_pred = end_points['grasp_correspondance'] # (B, Ns, Ns)
    corres_pred = torch.clamp(corres_pred, -1., 1.)
    corres_loss1 = compute_supervised_contrastive_loss(corres_pred, corres_label, training_mask, tau)
    if symmetric:
        corres_pred = corres_pred.transpose(1,2).contiguous()
        corres_label = corres_label.transpose(1,2).contiguous()
        training_mask = training_mask.transpose(1,2).contiguous()
        corres_loss2 = compute_correspondence_loss(corres_pred, corres_label, training_mask, tau)
        corres_loss = corres_loss1 + corres_loss2
    else:
        corres_loss = corres_loss1

    end_points['losses/stage4_grasp_correspondence_loss'] = corres_loss

    
    return corres_loss, end_points