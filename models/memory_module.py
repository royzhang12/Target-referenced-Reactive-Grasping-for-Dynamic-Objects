import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
from pointnet2_modules import PointnetSAModuleVotes, PointnetFPModule
import pointnet2_utils
from pointnet2_utils import CylinderQueryAndGroup
import pytorch_utils as pt_utils
from einops import repeat
from att_gnn import KeypointEncoder, AttentionalGNN
from modules import ApproachNet, CloudCrop, OperationNet, ToleranceNet
from temporal_model import Temporal_Model
from pytorch3d.transforms import matrix_to_euler_angles, euler_angles_to_matrix, matrix_to_rotation_6d, rotation_6d_to_matrix

class GraspProcess(nn.Module):
    def __init__(self):
        super().__init__()

    def get_fine_seed_features(self, fine_features, end_points):
        
        B, C, Ns = fine_features.shape
        seed_idxs = end_points['seed_idxs']
        seed_idxs = seed_idxs.view(B, -1).to(torch.int32)
        

        fine_features = pointnet2_utils.gather_operation(fine_features, seed_idxs)
        
        
        end_points['selected_grasp_features'] = pointnet2_utils.gather_operation(end_points['local_grasp_features'], seed_idxs)
        end_points['selected_color_features'] = pointnet2_utils.gather_operation(end_points['local_color_features'], seed_idxs)
        end_points['selected_grasp_pose_features'] = pointnet2_utils.gather_operation(end_points['grasp_pose_feature'], seed_idxs)
  
        end_points['fine_seed_features'] = fine_features

        seed_xyz = end_points['fp2_xyz']
        seed_xyz = seed_xyz.permute(0,2,1).contiguous()
        seed_xyz = pointnet2_utils.gather_operation(seed_xyz, seed_idxs)
        seed_xyz = seed_xyz.permute(0,2,1).contiguous()
        
        end_points['selected_seed_xyzs'] = seed_xyz

        return end_points

    def get_grasp(self, end_points):

        seed_idxs = end_points['seed_idxs'].to(torch.int32)
        B, L, _ = seed_idxs.shape
        seed_idxs = seed_idxs.view(-1, _)

        batch_grasp_preds = end_points['batch_grasp_preds'].permute(0,2,1).contiguous()
        grasps = pointnet2_utils.gather_operation(batch_grasp_preds, seed_idxs)
        grasp_translation = grasps[:,13:16,:]
        grasp_rotation = grasps[:,4:13, :]
        grasps =  torch.cat([grasp_translation, grasp_rotation], dim=1)

        end_points['selected_grasps'] = grasps
        return end_points
    
    def get_color_features(self, end_points):

        seed_idxs = end_points['seed_idxs'].to(torch.int32)
        B, L, _ = seed_idxs.shape
        seed_idxs = seed_idxs.view(-1, _)

        color_features = end_points['color_features']
        color_features = pointnet2_utils.gather_operation(color_features, seed_idxs)

        end_points['selected_colors'] = color_features

        return end_points


    def forward(self, end_points):
        
        fine_features = end_points['fp2_features']

        end_points = self.get_fine_seed_features(fine_features, end_points)
        end_points = self.get_grasp(end_points)
        
        return end_points

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
  
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
  
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1), :], 
                         requires_grad=False)
        return self.dropout(x)

class MemoryGnn(nn.Module):
    
    def __init__(self, frame_size = 5):
        super().__init__()

        self.fine_group = GraspProcess()

        input_feature_dim = 256

        self.frame_size = frame_size

        self.feature_dim = input_feature_dim


        self.conv1 = nn.Conv1d(input_feature_dim*8 , input_feature_dim*6, 1, bias=True)
        self.conv2 = nn.Conv1d(input_feature_dim*6, input_feature_dim*4, 1, bias=True)
        self.conv3 = nn.Conv1d(input_feature_dim*4, input_feature_dim*3, 1, bias=True)
        self.conv4 = nn.Conv1d(input_feature_dim*3, input_feature_dim*2, 1, bias=True)
        self.bn1 = nn.BatchNorm1d(input_feature_dim*6)
        self.bn2 = nn.BatchNorm1d(input_feature_dim*4)
        self.bn3 = nn.BatchNorm1d(input_feature_dim*3)

        self.temporal = Temporal_Model(512, 12, 8, 64, 512, dropout = 0.)


        self.final_proj = nn.Conv1d(
            self.feature_dim*2, self.feature_dim*2,
            kernel_size=1, bias=True)

        self.to_rot = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(inplace=True),
                    nn.Linear(256, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, 6)
                )

        self.temporal_embedding = PositionalEncoding(512, 0., 16)
        self.local_grasp_pcd_group = CylinderQueryAndGroup(
                radius = 0.06, hmin=-0.02, hmax=0.06, nsample=512, use_xyz=True, rotate_xyz=True, normalize_xyz=True
            )
    
    def forward(self, end_points):

    
        end_points = self.fine_group(end_points)

        selected_grasps = end_points['selected_grasps']
        selected_seed_features = end_points['fine_seed_features'] 

        global_seed_features = end_points['sa4_features']

        global_seed_features = F.max_pool1d(
            global_seed_features, kernel_size=global_seed_features.shape[-1]
        ).squeeze(-1) # (batch_size, mlps[-1], num_seed*num_depth, 1)
        
        pointcloud = end_points['input_xyz']
        seed_xyz = end_points['selected_seed_xyzs']
        colors = end_points['cloud_colors']
        colors = colors.transpose(1, 2).contiguous()
        B, _, N = selected_grasps.shape
        
        
        selected_grasp_features = end_points['selected_grasp_features']
        selected_color_features = end_points['selected_color_features']
        selected_grasp_pose_features = end_points['selected_grasp_pose_features']
        
        
        B, C, N = selected_grasp_features.shape
        B //= self.frame_size
        
        global_seed_features = repeat(global_seed_features, 'b c -> b c n', n=N)

        grasp_features = selected_grasp_pose_features + torch.cat([selected_grasp_features, selected_color_features, selected_seed_features, global_seed_features], dim=1)
        grasp_features = grasp_features.contiguous().view(B, self.frame_size, -1, N)
        grasp_features_cond = grasp_features[:, 0]
        grasp_features_cond = repeat(grasp_features_cond, 'b c n -> b l c n', l=self.frame_size-1)
        grasp_features = torch.cat([grasp_features_cond, grasp_features[:, 1:]], dim=-2)
        grasp_features = grasp_features.contiguous().view(B*(self.frame_size-1), -1, N)

        grasp_features = F.relu(self.bn1(self.conv1(grasp_features)), inplace=True)
        grasp_features = F.relu(self.bn2(self.conv2(grasp_features)), inplace=True)
        grasp_features = F.relu(self.bn3(self.conv3(grasp_features)), inplace=True)
        grasp_features = self.conv4(grasp_features)

        grasp_features = grasp_features.contiguous().view(B, self.frame_size-1, -1, N).permute(0, 3, 2, 1).contiguous().view(B*N, -1, self.frame_size-1).permute(0,2,1).contiguous()

        grasp_features = self.temporal_embedding(grasp_features)

        grasp_features = self.temporal(grasp_features)

        grasp_features = self.final_proj(grasp_features.permute(0,2,1).contiguous()).permute(0,2,1).contiguous()


        rot_pred = self.to_rot(grasp_features)
        rot_pred = rotation_6d_to_matrix(rot_pred)

        translation_pred = self.to_translation(grasp_features)

        end_points['translation_pred'] = translation_pred 
        end_points['rot_pred'] = rot_pred
        
        return end_points

