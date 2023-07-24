import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from refine_backbone import RefineBackbone
from backbone import Pointnet2Backbone
from modules import ApproachNet, CloudCrop, OperationNet, ToleranceNet
from loss import get_loss
from loss_utils import GRASP_MAX_WIDTH, GRASP_MAX_TOLERANCE
from label_generation import process_grasp_labels, match_grasp_view_and_label, batch_viewpoint_params_to_matrix
from att_gnn import KeypointEncoder, AttentionalGNN
from pointnet2_utils import CylinderQueryAndGroup
from einops import repeat

class GraspNetStage1(nn.Module):
    def __init__(self, input_feature_dim=0, num_view=300):
        super().__init__()
        self.backbone = Pointnet2Backbone(input_feature_dim)
        self.vpmodule = ApproachNet(num_view, 256)

    def forward(self, end_points):
        pointcloud = end_points['point_clouds']
        seed_features, seed_xyz, end_points = self.backbone(pointcloud, end_points)
        end_points = self.vpmodule(seed_xyz, seed_features, end_points)
        return end_points


class GraspNetStage2(nn.Module):
    def __init__(self, num_angle=12, num_depth=4, cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=True):
        super().__init__()
        self.num_angle = num_angle
        self.num_depth = num_depth
        self.is_training = is_training
        self.crop = CloudCrop(64, 3, cylinder_radius, hmin, hmax_list)
        self.operation = OperationNet(num_angle, num_depth)
        self.tolerance = ToleranceNet(num_angle, num_depth)
    
    def forward(self, end_points):
        pointcloud = end_points['input_xyz']
        if self.is_training:
            grasp_top_views_rot, _, _, _, end_points = match_grasp_view_and_label(end_points)
            seed_xyz = end_points['batch_grasp_point']
        else:
            grasp_top_views_rot = end_points['grasp_top_view_rot']
            seed_xyz = end_points['fp2_xyz']

        # print (seed_xyz.shape, pointcloud.shape)
        vp_features, vp_seed_features = self.crop(seed_xyz, pointcloud, grasp_top_views_rot)
        end_points = self.operation(vp_features, end_points)
        end_points = self.tolerance(vp_features, end_points)
        end_points['grasp_features'] = vp_features
        end_points['vp_seed_features'] = vp_seed_features

        return end_points

class GraspNet(nn.Module):
    def __init__(self, input_feature_dim=0, num_view=300, num_angle=12, num_depth=4, cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=True):
        super().__init__()
        self.is_training = is_training
        self.view_estimator = GraspNetStage1(input_feature_dim, num_view)
        self.grasp_generator = GraspNetStage2(num_angle, num_depth, cylinder_radius, hmin, hmax_list, is_training)

    def forward(self, end_points):
        end_points = self.view_estimator(end_points)
        if self.is_training:
            end_points = process_grasp_labels(end_points)
        end_points = self.grasp_generator(end_points)
        return end_points

class CorresNet(nn.Module):
    def __init__(self, input_feature_dim, num_sample=64, cylinder_radius=0.05, hmin=-0.02, hmax=0.04, cosine=True):
        super().__init__()
        self.feature_dim = input_feature_dim

        self.conv1 = nn.Conv1d(input_feature_dim*4, input_feature_dim*3, 1, bias=True)
        self.conv2 = nn.Conv1d(input_feature_dim*3, input_feature_dim*2, 1, bias=True)
        self.conv3 = nn.Conv1d(input_feature_dim*2, input_feature_dim, 1, bias=True)
        self.bn1 = nn.BatchNorm1d(input_feature_dim*3)
        self.bn2 = nn.BatchNorm1d(input_feature_dim*2)

        self.cosine	 = cosine
        self.crop = CloudCrop(256, 3, cylinder_radius=0.05, hmin=hmin, hmax_list=[0.06], use_xyz=True, rotate_xyz=True, use_feat=True)

        self.grasp_pose_encoder =  KeypointEncoder(
            input_feature_dim * 4, [128, 256, 512])
        
        self.gnn = AttentionalGNN(
            feature_dim=input_feature_dim, layer_names=['self', 'cross'] * 4)

        self.final_proj = nn.Conv1d(
            input_feature_dim, input_feature_dim,
            kernel_size=1, bias=True)

    def forward(self, end_points):
        pointcloud = end_points['input_xyz']
        colors = end_points['cloud_colors']
        seed_xyz = end_points['fp2_xyz']
        # seed_inds = end_points['fp2_inds'].long()
        seed_features = end_points['fp2_features']
        # grasp_features = end_points['grasp_features'][:,:,:,-1] # (B*2, C, Ns)
        if 'batch_grasp_preds' in end_points.keys():
            grasp_preds = end_points['batch_grasp_preds']
        else:
            grasp_preds, end_points = pred_decode(end_points, remove_background=False) # (B*2, Ns, 17)
        grasp_translation = grasp_preds[:,:,13:16].transpose(1,2).contiguous() # (B*2, 3, Ns)
        grasp_rotation = grasp_preds[:,:,4:13].transpose(1,2).contiguous() # (B*2, 9, Ns)
        grasp_scores = grasp_preds[:, :, 0].contiguous()
        
        global_seed_features = end_points['sa4_features']

        grasp_pose_feature = torch.cat([grasp_translation, grasp_rotation], dim=1)

        grasp_pose_feature = self.grasp_pose_encoder(grasp_pose_feature)

        colors = colors.transpose(1, 2).contiguous()
        B, Ns, _ = grasp_preds.size()
        approaching = grasp_preds[:,:,4:13].contiguous().view(B, Ns, 3, 3).contiguous()
        # color_features, _ = self.crop(seed_xyz, pointcloud, approaching, colors)
        global_seed_features = F.max_pool1d(
            global_seed_features, kernel_size=global_seed_features.shape[-1]
        ).squeeze(-1) # (batch_size, mlps[-1], num_seed*num_depth, 1)
        global_seed_features = repeat(global_seed_features, 'b c -> b c ns', ns=Ns)
        
        local_seed_features, _, color_features, _ = self.crop(seed_xyz, pointcloud, approaching, colors)
        local_seed_features = local_seed_features[:, :, :, -1]

        # color_features, _ = self.color_crop(seed_xyz, pointcloud, approaching, colors)
        color_features = color_features[:, :, :, -1]
        
        # print (local_seed_features.shape)
        end_points['local_grasp_features'] = local_seed_features
        end_points['local_color_features'] = color_features
        end_points['grasp_pose_feature'] = grasp_pose_feature

        # get grasp features
        grasp_features = grasp_pose_feature + torch.cat([global_seed_features,  seed_features, local_seed_features, color_features], dim=1) # (B*2, C+C+C+12, Ns)
        grasp_features = F.relu(self.bn1(self.conv1(grasp_features)), inplace=True)
        grasp_features = F.relu(self.bn2(self.conv2(grasp_features)), inplace=True)
        grasp_features = self.conv3(grasp_features)

        # compute grasp correspondence (cosine distance)
        B, C, Ns = grasp_features.size()
        B //= 2
        grasp_features = grasp_features.contiguous().view(B, 2, C, Ns)
        grasp_features_1 = grasp_features[:,0] 
        grasp_features_2 = grasp_features[:,1] 

        if self.cosine:

            match_desc1 = match_desc1.unsqueeze(3) # (B, C, Ns, 1)
            match_desc2 = match_desc2.unsqueeze(2) # (B, C, 1, Ns)

            match_desc_norm_1 = torch.norm(match_desc1, dim=1) + 1e-6 # (B, Ns, 1)
            match_desc_norm_2 = torch.norm(match_desc2, dim=1) + 1e-6 # (B, 1, Ns)

            match_desc_dot = torch.sum(match_desc1*match_desc2, dim=1)

            scores = match_desc_dot / (match_desc_norm_1*match_desc_norm_2) # (B, C, Ns, Ns)

        else:

            match_desc1, match_desc2 = self.gnn(grasp_features_1, grasp_features_2)

            match_desc1, match_desc2 = self.final_proj(match_desc1), self.final_proj(match_desc2)

            match_desc1, match_desc2 = match_desc1 / (self.feature_dim**.5), match_desc2 / (self.feature_dim**.5)  

            scores = torch.einsum('bdn,bdm->bnm', match_desc1, match_desc2)
        
        end_points['grasp_correspondance'] = scores

        return end_points

def pred_decode(end_points, remove_background=True):
    batch_size = len(end_points['point_clouds'])
    grasp_preds = []
    grasp_angles = []
    for i in range(batch_size):
        ## load predictions
        objectness_score = end_points['objectness_score'][i].float()
        grasp_score = end_points['grasp_score_pred'][i].float()
        grasp_center = end_points['fp2_xyz'][i].float()
        approaching = -end_points['grasp_top_view_xyz'][i].float()
        grasp_angle_class_score = end_points['grasp_angle_cls_pred'][i]
        grasp_width = 1.2 * end_points['grasp_width_pred'][i]
        grasp_width = torch.clamp(grasp_width, min=0, max=GRASP_MAX_WIDTH)
        grasp_tolerance = end_points['grasp_tolerance_pred'][i]

        ## slice preds by angle
        # grasp angle
        grasp_angle_class = torch.argmax(grasp_angle_class_score, 0)
        grasp_angle = grasp_angle_class.float() / 12 * np.pi
        # grasp score & width & tolerance
        grasp_angle_class_ = grasp_angle_class.unsqueeze(0)
        grasp_score = torch.gather(grasp_score, 0, grasp_angle_class_).squeeze(0)
        grasp_width = torch.gather(grasp_width, 0, grasp_angle_class_).squeeze(0)
        grasp_tolerance = torch.gather(grasp_tolerance, 0, grasp_angle_class_).squeeze(0)

        ## slice preds by score/depth
        # grasp depth
        grasp_depth_class = torch.argmax(grasp_score, 1, keepdims=True)
        grasp_depth = (grasp_depth_class.float()+1) * 0.01
        # grasp score & angle & width & tolerance
        grasp_score = torch.gather(grasp_score, 1, grasp_depth_class)
        grasp_angle = torch.gather(grasp_angle, 1, grasp_depth_class)
        grasp_width = torch.gather(grasp_width, 1, grasp_depth_class)
        grasp_tolerance = torch.gather(grasp_tolerance, 1, grasp_depth_class)
        grasp_score = grasp_score * grasp_tolerance / GRASP_MAX_TOLERANCE

        ## slice preds by objectness
        if remove_background:
            objectness_pred = torch.argmax(objectness_score, 0)
            objectness_mask = (objectness_pred==1)
            grasp_score = grasp_score[objectness_mask]
            grasp_width = grasp_width[objectness_mask]
            grasp_depth = grasp_depth[objectness_mask]
            approaching = approaching[objectness_mask]
            grasp_angle = grasp_angle[objectness_mask]
            grasp_center = grasp_center[objectness_mask]
            grasp_tolerance = grasp_tolerance[objectness_mask]

        ## convert to rotation matrix
        Ns = grasp_angle.size(0)
        approaching_ = approaching.contiguous().view(Ns, 3)
        grasp_angle_ = grasp_angle.contiguous().view(Ns)
        rotation_matrix = batch_viewpoint_params_to_matrix(approaching_, grasp_angle_)
        rotation_matrix = rotation_matrix.contiguous().view(Ns, 9)

        # merge preds
        grasp_height = 0.02 * torch.ones_like(grasp_score)
        obj_ids = -1 * torch.ones_like(grasp_score)
        grasp_preds.append(torch.cat([grasp_score, grasp_width, grasp_height, grasp_depth, rotation_matrix, grasp_center, obj_ids], axis=-1))
        grasp_angles.append(grasp_angle)

    if not remove_background:
        grasp_preds = torch.stack(grasp_preds, dim=0)
        grasp_angles = torch.stack(grasp_angles, dim=0)
        end_points['batch_grasp_preds'] = grasp_preds
        end_points['batch_grasp_angles'] = grasp_angles

    return grasp_preds, end_points

